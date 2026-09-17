#include "azure/azcache_provider/azcache_provider_loader.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <algorithm>

#include "common/exception/exception.h"
#include "utils/dylib/dylib.h"
#include "utils/logging/logging.h"
#include "utils/env/env.h"

namespace runai::llm::streamer::impl::azure
{

// Static members
std::weak_ptr<CacheLibHandle> AzCacheProviderLoader::s_shared_handle;
std::mutex AzCacheProviderLoader::s_handle_mutex;

namespace
{

// Known cache provider package/library names for auto-discovery
constexpr const char* CACHE_PROVIDER_PACKAGE = "dacs_client";
constexpr const char* CACHE_PROVIDER_LIB = "libStorageDirect.so";

/**
 * Use dladdr on a symbol known to live in libstreamerazure.so to locate
 * the site-packages directory, then look for a known cache provider .so.
 *
 * Layout: <site-packages>/runai_model_streamer/libstreamer/lib/libstreamerazure.so
 *         <site-packages>/<CACHE_PROVIDER_PACKAGE>/<CACHE_PROVIDER_LIB>
 */
std::string autodiscover_cache_lib()
{
    Dl_info info;
    auto fn_ptr = reinterpret_cast<void*>(&autodiscover_cache_lib);
    if (!dladdr(fn_ptr, &info) || !info.dli_fname)
    {
        LOG(DEBUG) << "AzCacheProvider: dladdr failed — cannot auto-discover";
        return {};
    }

    std::error_code ec;
    auto azure_so = std::filesystem::weakly_canonical(info.dli_fname, ec);
    if (ec)
    {
        LOG(DEBUG) << "AzCacheProvider: canonical path failed for "
                   << info.dli_fname << ": " << ec.message();
        return {};
    }

    // libstreamerazure.so → lib/ → libstreamer/ → runai_model_streamer/ → site-packages/
    auto site_packages = azure_so.parent_path().parent_path().parent_path().parent_path();
    auto candidate = site_packages / CACHE_PROVIDER_PACKAGE / CACHE_PROVIDER_LIB;

    if (std::filesystem::exists(candidate, ec) && !ec)
    {
        return candidate.string();
    }

    LOG(DEBUG) << "AzCacheProvider: auto-discovery checked " << candidate.string()
               << " — not found" << (ec ? ": " + ec.message() : "");
    return {};
}

CacheMode parse_cache_mode(const std::string& val)
{
    std::string lower = val;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return std::tolower(c); });

    if (lower == "0")
    {
        return CacheMode::Disabled;
    }
    if (lower == "1")
    {
        return CacheMode::Required;
    }
    // "auto" or any other value → auto mode
    return CacheMode::Auto;
}

} // anonymous namespace

// --- CacheLibHandle implementation ---

CacheLibHandle::CacheLibHandle(const std::string& path, CacheMode mode)
    : lib_handle(nullptr),
      read_fn(nullptr),
      close_fn(nullptr),
      lib_path(path)
{
    LOG(INFO) << "AzCacheProvider: loading cache library: " << lib_path;

    try
    {
        lib_handle = utils::Dylib::dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    }
    catch (...)
    {
        if (mode == CacheMode::Required)
        {
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }
        return;
    }

    // ABI version check
    runai_cache_abi_version_fn version_fn = nullptr;
    try
    {
        version_fn = utils::Dylib::dlsym<runai_cache_abi_version_fn>(lib_handle, RUNAI_CACHE_ABI_VERSION_SYMBOL);
    }
    catch (...)
    {
        if (mode == CacheMode::Required)
        {
            LOG(ERROR) << "AzCacheProvider: '" << lib_path << "' does not export "
                       << RUNAI_CACHE_ABI_VERSION_SYMBOL << " — incompatible library";
            utils::Dylib::dlclose(lib_handle);
            lib_handle = nullptr;
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }
        LOG(WARNING) << "AzCacheProvider: '" << lib_path << "' does not export "
                     << RUNAI_CACHE_ABI_VERSION_SYMBOL << " — skipping (pre-versioning build)";
        utils::Dylib::dlclose(lib_handle);
        lib_handle = nullptr;
        return;
    }

    uint32_t lib_version = version_fn();
    if (lib_version != RUNAI_CACHE_ABI_VERSION)
    {
        if (mode == CacheMode::Required)
        {
            LOG(ERROR) << "AzCacheProvider: ABI version mismatch — expected "
                       << RUNAI_CACHE_ABI_VERSION << ", got " << lib_version;
            utils::Dylib::dlclose(lib_handle);
            lib_handle = nullptr;
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }
        LOG(WARNING) << "AzCacheProvider: ABI version mismatch — expected "
                     << RUNAI_CACHE_ABI_VERSION << ", got " << lib_version << " — skipping";
        utils::Dylib::dlclose(lib_handle);
        lib_handle = nullptr;
        return;
    }

    // Resolve blob_read (required)
    try
    {
        read_fn = utils::Dylib::dlsym<blob_read_fn>(lib_handle, BLOB_READ_SYMBOL);
    }
    catch (...)
    {
        if (mode == CacheMode::Required)
        {
            utils::Dylib::dlclose(lib_handle);
            lib_handle = nullptr;
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }
        utils::Dylib::dlclose(lib_handle);
        lib_handle = nullptr;
        return;
    }

    // Resolve shutdown (optional — don't throw if missing)
    try
    {
        close_fn = utils::Dylib::dlsym<shutdown_fn>(lib_handle, SHUTDOWN_SYMBOL);
    }
    catch (...)
    {
        close_fn = nullptr; // not exported, that's fine
    }

    LOG(INFO) << "AzCacheProvider: cache provider loaded successfully from " << lib_path
              << " (shutdown: " << (close_fn ? "yes" : "no") << ")";
}

CacheLibHandle::~CacheLibHandle()
{
    if (!lib_handle)
    {
        return;
    }

    // Call shutdown() for graceful shutdown before releasing the handle
    if (close_fn)
    {
        LOG(INFO) << "AzCacheProvider: calling shutdown() for graceful shutdown";
        try
        {
            close_fn();
        }
        catch (...)
        {
            LOG(ERROR) << "AzCacheProvider: shutdown() threw an exception";
        }
    }

    // Note: we intentionally do NOT call dlclose() here.
    lib_handle = nullptr;

    LOG(INFO) << "AzCacheProvider: released cache provider handle";
}

// --- AzCacheProviderLoader implementation ---

std::shared_ptr<AzCacheProviderLoader> AzCacheProviderLoader::from_env()
{
    CacheProviderConfig config;
    config.mode = CacheMode::Auto;

    // Parse mode from RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED
    std::string enabled_val;
    if (utils::try_getenv<std::string>(RUNAI_AZURE_CACHE_ENABLED_ENV, enabled_val))
    {
        config.mode = parse_cache_mode(enabled_val);
        if (config.mode == CacheMode::Disabled)
        {
            LOG(INFO) << "AzCacheProvider: disabled via " << RUNAI_AZURE_CACHE_ENABLED_ENV << "=" << enabled_val;
        }
    }

    // Parse explicit library path from RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB
    std::string lib_path;
    if (utils::try_getenv<std::string>(RUNAI_AZURE_CACHE_LIB_ENV, lib_path))
    {
        if (lib_path.empty())
        {
            LOG(WARNING) << "AzCacheProvider: " << RUNAI_AZURE_CACHE_LIB_ENV
                         << " is set to empty string — ignoring";
        }
        else
        {
            config.lib_path = lib_path;
        }
    }

    // Auto-discover if no explicit path and not disabled
    if (config.lib_path.empty() && config.mode != CacheMode::Disabled)
    {
        std::string discovered = autodiscover_cache_lib();
        if (!discovered.empty())
        {
            LOG(INFO) << "AzCacheProvider: auto-discovered cache library: " << discovered;
            config.lib_path = discovered;
        }
    }

    return std::make_shared<AzCacheProviderLoader>(config);
}

AzCacheProviderLoader::AzCacheProviderLoader(const CacheProviderConfig& config)
    : _handle(nullptr),
      _enabled(false)
{
    if (config.mode == CacheMode::Disabled)
    {
        return;
    }

    if (config.lib_path.empty())
    {
        if (config.mode == CacheMode::Required)
        {
            LOG(ERROR) << "AzCacheProvider: mode=1 (required) but no cache library path configured. "
                       << "Set " << RUNAI_AZURE_CACHE_LIB_ENV << " or install a cache provider package.";
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }
        LOG(DEBUG) << "AzCacheProvider: no cache library configured — cache disabled";
        return;
    }

    // Acquire or create the shared library handle
    {
        std::lock_guard<std::mutex> lock(s_handle_mutex);
        _handle = s_shared_handle.lock();

        if (_handle && _handle->lib_handle && _handle->lib_path == config.lib_path)
        {
            // Reuse existing handle for same library
            LOG(DEBUG) << "AzCacheProvider: reusing shared cache library handle for " << config.lib_path;
        }
        else
        {
            // Create new handle (first loader or path changed)
            _handle = std::make_shared<CacheLibHandle>(config.lib_path, config.mode);
            s_shared_handle = _handle;
        }
    }

    if (_handle && _handle->read_fn)
    {
        _enabled = true;
    }
    LOG(INFO) << "AzCacheProvider: cache provider is " << (_enabled ? "enabled" : "disabled");
}

AzCacheProviderLoader::~AzCacheProviderLoader()
{
    // Release under mutex so destruction of CacheLibHandle (which calls shutdown())
    // cannot race with a new loader trying to lock() the weak_ptr and recreate.
    std::lock_guard<std::mutex> lock(s_handle_mutex);
    _handle.reset();
}

bool AzCacheProviderLoader::read(
    const std::string& account,
    const std::string& container,
    const std::string& blob,
    char* buffer,
    size_t offset,
    size_t length)
{
    if (!_enabled || !_handle || !_handle->read_fn)
    {
        return false;
    }

    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};
    ssize_t bytes_read = _handle->read_fn(
        account.c_str(), container.c_str(), blob.c_str(),
        buffer, offset, length, error_buf, sizeof(error_buf));

    if (bytes_read < 0 || static_cast<size_t>(bytes_read) != length)
    {
        if (error_buf[0] != '\0')
        {
            LOG(ERROR) << "AzCacheProvider: cache read failed for "
                       << account << "/" << container << "/" << blob
                       << " offset=" << offset << " length=" << length
                       << ": " << error_buf;
        }
        else
        {
            LOG(ERROR) << "AzCacheProvider: cache read failed for "
                       << account << "/" << container << "/" << blob
                       << " offset=" << offset << " length=" << length;
        }
        return false;
    }

    LOG(SPAM) << "AzCacheProvider: cache read " << length << " bytes from "
              << account << "/" << container << "/" << blob << " offset=" << offset;
    return true;
}

} // namespace runai::llm::streamer::impl::azure
