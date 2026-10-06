/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include <dlfcn.h>

#include "azure/azcache_provider/runai_azcache_provider.h"
#include "azure/azcache_provider/azcache_provider_loader.h"

namespace runai::llm::streamer::impl::azure::testing
{

namespace fs = std::filesystem;

class SimpleFileCacheTest : public ::testing::Test
{
protected:
    fs::path cache_dir_;
    fs::path so_path_;

    void SetUp() override
    {
        // Create a temp cache directory
        cache_dir_ = fs::temp_directory_path() / ("runai_cache_test_" + std::to_string(getpid()));
        fs::create_directories(cache_dir_);

        // Locate the example .so built by Bazel
        // Bazel puts genrule outputs relative to the runfiles directory
        const char* test_srcdir = getenv("TEST_SRCDIR");
        const char* test_workspace = getenv("TEST_WORKSPACE");
        if (test_srcdir && test_workspace)
        {
            so_path_ = fs::path(test_srcdir) / test_workspace / "azure/azcache_provider/libsimple_file_cache_test.so";
        }

        // Fallback: try relative path from working directory
        if (!fs::exists(so_path_))
        {
            so_path_ = "azure/azcache_provider/libsimple_file_cache_test.so";
        }
    }

    void TearDown() override
    {
        fs::remove_all(cache_dir_);
        unsetenv("RUNAI_CACHE_DIR");
    }

    // Write a test blob file into the cache directory
    void populate_cache(const std::string& container, const std::string& blob,
                        const std::vector<uint8_t>& data)
    {
        fs::path blob_path = cache_dir_ / container / blob;
        fs::create_directories(blob_path.parent_path());
        std::ofstream ofs(blob_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    // Directly dlopen the example .so and return the function pointer
    blob_read_fn load_cache_fn()
    {
        void* handle = dlopen(so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle)
        {
            ADD_FAILURE() << "dlopen failed: " << dlerror();
            return nullptr;
        }
        auto fn = reinterpret_cast<blob_read_fn>(
            dlsym(handle, BLOB_READ_SYMBOL));
        if (!fn)
        {
            ADD_FAILURE() << "dlsym failed: " << dlerror();
        }
        return fn;
    }

    // Verify the .so exports a compatible ABI version
    uint32_t load_abi_version()
    {
        void* handle = dlopen(so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle)
        {
            ADD_FAILURE() << "dlopen failed: " << dlerror();
            return 0;
        }
        auto version_fn = reinterpret_cast<runai_cache_abi_version_fn>(
            dlsym(handle, RUNAI_CACHE_ABI_VERSION_SYMBOL));
        if (!version_fn)
        {
            ADD_FAILURE() << "dlsym failed for " << RUNAI_CACHE_ABI_VERSION_SYMBOL << ": " << dlerror();
            return 0;
        }
        return version_fn();
    }
};

TEST_F(SimpleFileCacheTest, ReadFullBlob)
{
    // Verify ABI version is exported and matches expected
    uint32_t version = load_abi_version();
    ASSERT_EQ(version, RUNAI_CACHE_ABI_VERSION);

    // Populate cache with test data
    std::vector<uint8_t> data(1024);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    populate_cache("test-container", "model/weights.bin", data);

    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    auto cache_read = load_cache_fn();
    ASSERT_NE(cache_read, nullptr);

    // Read the full blob
    std::vector<uint8_t> buf(1024);
    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};
    ssize_t result = cache_read("test-account", "test-container", "model/weights.bin",
                                buf.data(), 0, 1024, error_buf, sizeof(error_buf));
    ASSERT_EQ(result, 1024) << (error_buf[0] ? error_buf : "no error detail");
    EXPECT_EQ(error_buf[0], '\0');
    EXPECT_EQ(buf, data);
}

TEST_F(SimpleFileCacheTest, ReadRange)
{
    // Populate cache with sequential bytes
    std::vector<uint8_t> data(4096);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<uint8_t>(i % 251);  // prime to avoid aliasing
    }
    populate_cache("models", "llm/shard-0001.safetensors", data);

    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    auto cache_read = load_cache_fn();
    ASSERT_NE(cache_read, nullptr);

    // Read a range from the middle
    size_t offset = 1000;
    size_t length = 512;
    std::vector<uint8_t> buf(length);
    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};

    ssize_t result = cache_read("test-account", "models", "llm/shard-0001.safetensors",
                                buf.data(), offset, length, error_buf, sizeof(error_buf));
    ASSERT_EQ(result, static_cast<ssize_t>(length)) << (error_buf[0] ? error_buf : "no error");
    EXPECT_EQ(error_buf[0], '\0');

    // Verify data matches the expected range
    for (size_t i = 0; i < length; ++i)
    {
        EXPECT_EQ(buf[i], data[offset + i]) << "mismatch at offset " << (offset + i);
    }
}

TEST_F(SimpleFileCacheTest, MissingBlobReturnsError)
{
    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    auto cache_read = load_cache_fn();
    ASSERT_NE(cache_read, nullptr);

    char buf[100];
    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};

    ssize_t result = cache_read("test-account", "no-such-container", "no-such-blob",
                                buf, 0, 100, error_buf, sizeof(error_buf));
    EXPECT_EQ(result, -1);
    EXPECT_NE(error_buf[0], '\0');
    EXPECT_NE(std::string(error_buf).find("open failed"), std::string::npos);
}

TEST_F(SimpleFileCacheTest, ShortReadReturnsError)
{
    // Populate cache with a small file
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    populate_cache("c", "small.bin", data);

    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    auto cache_read = load_cache_fn();
    ASSERT_NE(cache_read, nullptr);

    // Try to read more than the file contains
    char buf[100];
    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};

    ssize_t result = cache_read("test-account", "c", "small.bin", buf, 0, 100,
                                error_buf, sizeof(error_buf));
    EXPECT_EQ(result, -1);
    EXPECT_NE(error_buf[0], '\0');
    EXPECT_NE(std::string(error_buf).find("short read"), std::string::npos);
}

TEST_F(SimpleFileCacheTest, NullArgumentsReturnError)
{
    auto cache_read = load_cache_fn();
    ASSERT_NE(cache_read, nullptr);

    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};
    char buf[10];

    EXPECT_EQ(cache_read(nullptr, "c", "blob", buf, 0, 10, error_buf, sizeof(error_buf)), -1);
    error_buf[0] = '\0';

    EXPECT_EQ(cache_read("a", nullptr, "blob", buf, 0, 10, error_buf, sizeof(error_buf)), -1);
    error_buf[0] = '\0';

    EXPECT_EQ(cache_read("a", "c", nullptr, buf, 0, 10, error_buf, sizeof(error_buf)), -1);
    error_buf[0] = '\0';

    EXPECT_EQ(cache_read("a", "c", "b", nullptr, 0, 10, error_buf, sizeof(error_buf)), -1);
}

TEST_F(SimpleFileCacheTest, MultipleContainers)
{
    // Populate two containers with different data
    std::vector<uint8_t> data_a = {10, 20, 30, 40};
    std::vector<uint8_t> data_b = {50, 60, 70, 80};
    populate_cache("container-a", "file.bin", data_a);
    populate_cache("container-b", "file.bin", data_b);

    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    auto cache_read = load_cache_fn();
    ASSERT_NE(cache_read, nullptr);

    char buf[4];
    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};

    ASSERT_EQ(cache_read("test-account", "container-a", "file.bin", buf, 0, 4,
                         error_buf, sizeof(error_buf)), 4);
    EXPECT_EQ(memcmp(buf, data_a.data(), 4), 0);

    ASSERT_EQ(cache_read("test-account", "container-b", "file.bin", buf, 0, 4,
                         error_buf, sizeof(error_buf)), 4);
    EXPECT_EQ(memcmp(buf, data_b.data(), 4), 0);
}

TEST_F(SimpleFileCacheTest, PathTraversalRejected)
{
    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    auto cache_read = load_cache_fn();
    ASSERT_NE(cache_read, nullptr);

    char buf[10];
    char error_buf[RUNAI_CACHE_ERROR_BUF_SIZE] = {};

    // ".." in container
    EXPECT_EQ(cache_read("a", "../etc", "passwd", buf, 0, 10, error_buf, sizeof(error_buf)), -1);
    EXPECT_NE(std::string(error_buf).find("path traversal"), std::string::npos);
    error_buf[0] = '\0';

    // ".." in blob
    EXPECT_EQ(cache_read("a", "c", "../../etc/passwd", buf, 0, 10, error_buf, sizeof(error_buf)), -1);
    EXPECT_NE(std::string(error_buf).find("path traversal"), std::string::npos);
    error_buf[0] = '\0';

    // ".." embedded in path
    EXPECT_EQ(cache_read("a", "c", "sub/../../../etc/shadow", buf, 0, 10, error_buf, sizeof(error_buf)), -1);
    EXPECT_NE(std::string(error_buf).find("path traversal"), std::string::npos);
    error_buf[0] = '\0';

    // "..." (not traversal) should NOT be rejected — will fail with open error instead
    EXPECT_EQ(cache_read("a", "c", "...", buf, 0, 10, error_buf, sizeof(error_buf)), -1);
    EXPECT_EQ(std::string(error_buf).find("path traversal"), std::string::npos);
}

// --- Shared handle tests ---

TEST_F(SimpleFileCacheTest, SharedHandleMultipleLoadersSameLibrary)
{
    // Multiple loaders for the same library should share one CacheLibHandle
    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    config.lib_path = so_path_.string();

    auto loader1 = std::make_shared<AzCacheProviderLoader>(config);
    EXPECT_TRUE(loader1->is_enabled());

    auto loader2 = std::make_shared<AzCacheProviderLoader>(config);
    EXPECT_TRUE(loader2->is_enabled());

    // Both should work
    std::vector<uint8_t> data = {0xDE, 0xAD, 0xBE, 0xEF};
    populate_cache("shared-test", "blob.bin", data);
    char buf[4];
    EXPECT_TRUE(loader1->read("acct", "shared-test", "blob.bin", buf, 0, 4));
    EXPECT_TRUE(loader2->read("acct", "shared-test", "blob.bin", buf, 0, 4));

    // Destroy first — cache should still work via second
    loader1.reset();
    EXPECT_TRUE(loader2->read("acct", "shared-test", "blob.bin", buf, 0, 4));
}

TEST_F(SimpleFileCacheTest, ShutdownCalledOnLastDestruction)
{
    setenv("RUNAI_CACHE_DIR", cache_dir_.c_str(), 1);

    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    config.lib_path = so_path_.string();

    // Load the .so to check shutdown_called() helper
    void* handle = dlopen(so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();
    auto close_called_fn = reinterpret_cast<int(*)()>(dlsym(handle, "shutdown_called"));
    ASSERT_NE(close_called_fn, nullptr);
    auto reset_fn = reinterpret_cast<void(*)()>(dlsym(handle, "shutdown_reset"));
    ASSERT_NE(reset_fn, nullptr);

    // Reset state from any prior test (OS dedupes dlopen, so g_closed persists)
    reset_fn();
    EXPECT_EQ(close_called_fn(), 0);

    {
        auto loader1 = std::make_shared<AzCacheProviderLoader>(config);
        EXPECT_TRUE(loader1->is_enabled());

        auto loader2 = std::make_shared<AzCacheProviderLoader>(config);
        EXPECT_TRUE(loader2->is_enabled());

        // Destroy first loader — shutdown should NOT be called yet
        loader1.reset();
        EXPECT_EQ(close_called_fn(), 0);

        // Destroy last loader — shutdown SHOULD be called
        loader2.reset();
    }

    EXPECT_EQ(close_called_fn(), 1);
    dlclose(handle);
}

// Test the AzCacheProviderLoader with explicit config
TEST(CacheProviderLoaderTest, DisabledModeDoesNotLoad)
{
    CacheProviderConfig config;
    config.mode = CacheMode::Disabled;
    config.lib_path = "/nonexistent.so";

    AzCacheProviderLoader loader(config);
    EXPECT_FALSE(loader.is_enabled());
}

TEST(CacheProviderLoaderTest, AutoModeNoPathDisables)
{
    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    // lib_path empty

    AzCacheProviderLoader loader(config);
    EXPECT_FALSE(loader.is_enabled());
}

TEST(CacheProviderLoaderTest, AutoModeInvalidPathDisables)
{
    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    config.lib_path = "/nonexistent/path/to/lib.so";

    AzCacheProviderLoader loader(config);
    EXPECT_FALSE(loader.is_enabled());
}

TEST(CacheProviderLoaderTest, RequiredModeNoPathThrows)
{
    CacheProviderConfig config;
    config.mode = CacheMode::Required;
    // lib_path empty

    EXPECT_THROW(AzCacheProviderLoader loader(config), std::exception);
}

TEST(CacheProviderLoaderTest, RequiredModeInvalidPathThrows)
{
    CacheProviderConfig config;
    config.mode = CacheMode::Required;
    config.lib_path = "/nonexistent/path/to/lib.so";

    EXPECT_THROW(AzCacheProviderLoader loader(config), std::exception);
}

// Helper: compile a minimal .so from source code string
static std::string build_test_so(const std::string& source, const std::string& name)
{
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / ("abi_test_" + std::to_string(getpid()));
    fs::create_directories(dir);

    fs::path src_path = dir / (name + ".c");
    fs::path so_path = dir / (name + ".so");

    std::ofstream ofs(src_path);
    ofs << source;
    ofs.close();

    std::string cmd = "gcc -shared -fPIC -o " + so_path.string() + " " + src_path.string() + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret != 0) return "";
    return so_path.string();
}

// --- ABI version tests with CacheProviderLoader ---

TEST(CacheProviderLoaderAbiTest, AutoModeNoVersionSymbolDisables)
{
    // .so with blob_read but no cache_abi_version
    std::string src = R"(
#include <stddef.h>
#include <sys/types.h>
ssize_t blob_read(const char* a, const char* c, const char* b,
    void* buf, size_t offset, size_t length,
    char* err, size_t err_sz) { return (ssize_t)length; }
)";
    std::string path = build_test_so(src, "no_version");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    config.lib_path = path;

    AzCacheProviderLoader loader(config);
    EXPECT_FALSE(loader.is_enabled());

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(CacheProviderLoaderAbiTest, RequiredModeNoVersionSymbolThrows)
{
    // .so with blob_read but no cache_abi_version
    std::string src = R"(
#include <stddef.h>
#include <sys/types.h>
ssize_t blob_read(const char* a, const char* c, const char* b,
    void* buf, size_t offset, size_t length,
    char* err, size_t err_sz) { return (ssize_t)length; }
)";
    std::string path = build_test_so(src, "no_version_req");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Required;
    config.lib_path = path;

    EXPECT_THROW(AzCacheProviderLoader loader(config), std::exception);

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(CacheProviderLoaderAbiTest, AutoModeVersionMismatchDisables)
{
    // .so with version returning 999 (wrong)
    std::string src = R"(
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
uint32_t cache_abi_version(void) { return 999; }
ssize_t blob_read(const char* a, const char* c, const char* b,
    void* buf, size_t offset, size_t length,
    char* err, size_t err_sz) { return (ssize_t)length; }
)";
    std::string path = build_test_so(src, "bad_version");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    config.lib_path = path;

    AzCacheProviderLoader loader(config);
    EXPECT_FALSE(loader.is_enabled());

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(CacheProviderLoaderAbiTest, RequiredModeVersionMismatchThrows)
{
    // .so with version returning 999 (wrong)
    std::string src = R"(
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
uint32_t cache_abi_version(void) { return 999; }
ssize_t blob_read(const char* a, const char* c, const char* b,
    void* buf, size_t offset, size_t length,
    char* err, size_t err_sz) { return (ssize_t)length; }
)";
    std::string path = build_test_so(src, "bad_version_req");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Required;
    config.lib_path = path;

    EXPECT_THROW(AzCacheProviderLoader loader(config), std::exception);

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(CacheProviderLoaderAbiTest, AutoModeVersionMatchEnables)
{
    // .so with correct version and valid blob_read
    std::string src = R"(
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
uint32_t cache_abi_version(void) { return 1; }
ssize_t blob_read(const char* a, const char* c, const char* b,
    void* buf, size_t offset, size_t length,
    char* err, size_t err_sz) { return (ssize_t)length; }
)";
    std::string path = build_test_so(src, "good_version");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    config.lib_path = path;

    AzCacheProviderLoader loader(config);
    EXPECT_TRUE(loader.is_enabled());

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(CacheProviderLoaderAbiTest, RequiredModeVersionMatchEnables)
{
    // .so with correct version and valid blob_read
    std::string src = R"(
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
uint32_t cache_abi_version(void) { return 1; }
ssize_t blob_read(const char* a, const char* c, const char* b,
    void* buf, size_t offset, size_t length,
    char* err, size_t err_sz) { return (ssize_t)length; }
)";
    std::string path = build_test_so(src, "good_version_req");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Required;
    config.lib_path = path;

    AzCacheProviderLoader loader(config);
    EXPECT_TRUE(loader.is_enabled());

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(CacheProviderLoaderAbiTest, AutoModeVersionMatchNoBlobReadDisables)
{
    // .so with correct version but no blob_read symbol
    std::string src = R"(
#include <stdint.h>
uint32_t cache_abi_version(void) { return 1; }
)";
    std::string path = build_test_so(src, "no_blob_read");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Auto;
    config.lib_path = path;

    AzCacheProviderLoader loader(config);
    EXPECT_FALSE(loader.is_enabled());

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(CacheProviderLoaderAbiTest, RequiredModeVersionMatchNoBlobReadThrows)
{
    // .so with correct version but no blob_read symbol
    std::string src = R"(
#include <stdint.h>
uint32_t cache_abi_version(void) { return 1; }
)";
    std::string path = build_test_so(src, "no_blob_read_req");
    ASSERT_FALSE(path.empty());

    CacheProviderConfig config;
    config.mode = CacheMode::Required;
    config.lib_path = path;

    EXPECT_THROW(AzCacheProviderLoader loader(config), std::exception);

    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

} // namespace runai::llm::streamer::impl::azure::testing
