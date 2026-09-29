/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <type_traits>

#include "common/s3_wrapper/s3_wrapper.h"
#include "common/s3_credentials/s3_credentials.h"
#include "common/exception/exception.h"

#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::common::s3
{

std::mutex S3ClientWrapper::_backend_handle_mutex;

S3ClientWrapper::Params::Params(std::shared_ptr<StorageUri> uri, const Credentials & credentials, size_t chunk_bytesize,
                                unsigned concurrent_readers) :
    chunk_bytesize(chunk_bytesize),
    uri(uri),
    credentials(credentials),
    concurrent_readers(concurrent_readers)
{
    if (auto ep = credentials.endpoint(); ep.has_value()) // endpoint passed as parameter by user application (in credentials)
    {
        _endpoint = ep.value();
        LOG(DEBUG) <<"Using credentials endpoint " << ep.value();
    }
    else
    {
        std::string endpoint;
        bool override_endpoint = utils::try_getenv("AWS_ENDPOINT_URL", endpoint);
        bool override_endpoint_flag = utils::getenv<bool>("RUNAI_STREAMER_OVERRIDE_ENDPOINT_URL", true);
        if (override_endpoint) // endpoint passed as environment variable
        {
            LOG(DEBUG) << "direct override of url endpoint in client configuration";
            if (override_endpoint_flag)
            {
                _endpoint = endpoint;
            }
            LOG(DEBUG) <<"Using environment variable endpoint " << endpoint << (override_endpoint_flag ? " , using configuration parameter endpointOverride" : "");
        }
    }
}

const common::backend_api::ObjectClientConfig_t S3ClientWrapper::Params::to_config(std::vector<common::backend_api::ObjectConfigParam_t> & initial_params) const
{
    common::backend_api::ObjectClientConfig_t config;
    config.endpoint_url = _endpoint.empty() ? nullptr : _endpoint.c_str();

    credentials.to_object_client_config(initial_params);
    config.num_initial_params = initial_params.size();
    config.initial_params = initial_params.data();
    config.default_storage_chunk_size = chunk_bytesize;
    config.concurrent_readers = concurrent_readers;
    return config;
}

size_t S3ClientWrapper::max_inflight_bytes()
{
    constexpr size_t unbounded = static_cast<size_t>(-1);

    // _backend_handle is a live-wrapper invariant (set and dereferenced during construction, held by its own
    // shared_ptr), so it is non-null here - use it directly, like every other method. obj_get_backend_config
    // is a required entry point (resolved strictly in resolve_api), so it is present too. A plugin with no
    // bounded in-flight window (gcs / azure) returns a non-Success code for the key, which we map to unbounded
    // - that is the backend advertising "no limit", not a missing-symbol fallback.
    char buf[32] = {0};
    unsigned int len = sizeof(buf);
    auto ret = _backend_handle->api.get_backend_config(_backend_handle->backend_handle(), "max_inflight_bytes", buf, &len);
    if (ret != ResponseCode::Success)
    {
        return unbounded;
    }
    buf[sizeof(buf) - 1] = '\0';
    return static_cast<size_t>(std::strtoull(buf, nullptr, 10));
}

const utils::Semver min_glibc_semver = utils::Semver(description(static_cast<int>(ResponseCode::GlibcPrerequisite)));

S3ClientWrapper::BackendHandle::BackendHandle(const Params & params) :
    dylib_ptr(open_object_storage_impl(params))
{
    ASSERT(dylib_ptr != nullptr) << "Failed to open libstreamer shared library"; // should never happen

    resolve_api();

    auto ret = api.open_backend(&_backend_handle);
    if (ret != ResponseCode::Success)
    {
        LOG(ERROR) << "Failed to open object storage";
        throw Exception(ret);
    }
    LOG(DEBUG) << "Opened object storage";
}

void S3ClientWrapper::BackendHandle::resolve_api()
{
    // Resolve every plugin entry point once, here, instead of per call. dlsym takes the process-wide
    // dynamic-linker lock, so resolving on the hot read/response path would serialize all workers on it.
    //
    // Every entry point is required, so a missing one fails backend open - fail fast, before any client is
    // created or any read is submitted (this makes resolve_api a full-API conformance check on the plugin).
    // dlsym throws a bare std::exception for a missing symbol, which the client-build catch (in
    // ObjectStorageWorker) would only see as a generic UnknownError; wrap it as a typed ObjPluginLoadError so
    // the failure is attributed to the plugin (an incomplete / ABI-mismatched library).
    auto require = [this](auto & fn, const char * name)
    {
        using Fn = typename std::remove_reference<decltype(fn)>::type;
        try
        {
            fn = dylib_ptr->dlsym<Fn>(name);
        }
        catch (...)
        {
            LOG(ERROR) << "Object storage plugin is missing required symbol '" << name << "'";
            throw Exception(ResponseCode::ObjPluginLoadError);
        }
    };

    require(api.open_backend,                "obj_open_backend");
    require(api.close_backend,               "obj_close_backend");
    require(api.create_client,               "obj_create_client");
    require(api.remove_client,               "obj_remove_client");
    require(api.remove_all_clients,          "obj_remove_all_clients");
    require(api.cancel_all_reads,            "obj_cancel_all_reads");
    require(api.request_read,                "obj_request_read");
    require(api.wait_for_completions,        "obj_wait_for_completions");
    require(api.list_files,                  "obj_list_files");
    require(api.free_file_list,              "obj_free_file_list");
    require(api.get_backend_shutdown_policy, "obj_get_backend_shutdown_policy");
    require(api.get_backend_config,          "obj_get_backend_config");
}

S3ClientWrapper::BackendHandle::~BackendHandle()
{
    try
    {
        auto ret = api.close_backend(_backend_handle);
        if (ret != ResponseCode::Success)
        {
            LOG(ERROR) << "Failed to close object storage";
            throw Exception(ret);
        }
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while closing object storage";
    }
}

const ObjectPluginType ObjectPluginType::ObjStorageGCS(PluginID::GCS, obj_plugin_gcs_name, lib_streamer_gcs_so_name);
const ObjectPluginType ObjectPluginType::ObjStorageS3(PluginID::S3, obj_plugin_s3_name, lib_streamer_s3_so_name);
const ObjectPluginType ObjectPluginType::ObjStorageAzure(PluginID::AZURE, obj_plugin_azure_name, lib_streamer_azure_so_name);

const ObjectPluginType S3ClientWrapper::BackendHandle::get_libstreamers_plugin_type(const std::shared_ptr<common::s3::StorageUri> & uri) {
    if (uri != nullptr && uri->is_gcs()) {
        return ObjectPluginType::ObjStorageGCS;
    } else if (uri != nullptr && uri->is_azure()) {
        return ObjectPluginType::ObjStorageAzure;
    } else {
        return ObjectPluginType::ObjStorageS3;
    }
}

std::shared_ptr<utils::Dylib> S3ClientWrapper::BackendHandle::open_object_storage_impl(const Params & params)
{
    // lazy load the s3 library once
    const ObjectPluginType plugin_type = get_libstreamers_plugin_type(params.uri);
    try
    {
        return std::make_shared<utils::Dylib>(plugin_type.so_name().c_str());
    }
    catch (...)
    {
        LOG(ERROR) << "Failed to open storage backend for " << plugin_type.name() << ": " << plugin_type.so_name();
    }
    switch(plugin_type.id()) {
        case PluginID::GCS:
            throw Exception(ResponseCode::GCSNotSupported);
        case PluginID::S3:
            throw Exception(ResponseCode::S3NotSupported);
        case PluginID::AZURE:
            throw Exception(ResponseCode::AzureBlobNotSupported);
    }
    return nullptr;
}

common::backend_api::ObjectBackendHandle_t S3ClientWrapper::BackendHandle::backend_handle() const
{
    return _backend_handle;
}

void S3ClientWrapper::shutdown()
{
    try
    {
        LOG(DEBUG) << "Shutting down S3 client wrapper";
        std::shared_ptr<BackendHandle> handle = manage_backend_handle(Params(), ManageBackendHandleOp::CREATE);
        handle->api.remove_all_clients();

       // destroy according to backend shutdown policy
       manage_backend_handle(Params(), ManageBackendHandleOp::DESTROY);
    }
    catch(...)
    {
    }
}

void S3ClientWrapper::stop()
{
    try
    {
        std::shared_ptr<BackendHandle> handle = manage_backend_handle(Params(), ManageBackendHandleOp::CREATE);
        ASSERT(handle != nullptr) << "Backend handle is already closed";
        handle->api.cancel_all_reads();
    }
    catch(...)
    {
    }
}

S3ClientWrapper::S3ClientWrapper(const Params & params) :
    _backend_handle(manage_backend_handle(params, ManageBackendHandleOp::CREATE)),
    _s3_client(create_client(params))
{
    LOG(SPAM) << "Created client for uri " << *params.uri;
    ASSERT(_backend_handle != nullptr) << "Backend handle is alreday closed";
}

S3ClientWrapper::~S3ClientWrapper()
{
    try
    {
        ASSERT(_backend_handle != nullptr) << "Backend handle is alreday closed";
        _backend_handle->api.remove_client(_s3_client);
    }
    catch(...)
    {
        LOG(ERROR) << "Caught exception while deleting s3 client";
    }
}


std::shared_ptr<S3ClientWrapper::BackendHandle> S3ClientWrapper::manage_backend_handle(const Params & params, ManageBackendHandleOp op)
{
    std::unique_lock<std::mutex> lock(_backend_handle_mutex);
    LOG(SPAM) << "Managing backend handle " << (op == ManageBackendHandleOp::CREATE ? "CREATE" : "DESTROY");

    static auto __backend_handle = std::make_shared<BackendHandle>(params);
    if (__backend_handle == nullptr && op == ManageBackendHandleOp::CREATE)
    {
        LOG(DEBUG) << "Backend handle is not initialized";
        __backend_handle = std::make_shared<BackendHandle>(params);
    }

    if (op == ManageBackendHandleOp::DESTROY && __backend_handle != nullptr)
    {
        const auto shutdown_policy = get_backend_shutdown_policy(__backend_handle);
        if (shutdown_policy == common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN)
        {
            LOG(DEBUG) << "Destroying backend handle on streamer shutdown";
            ASSERT(__backend_handle != nullptr) << "Backend handle is not initialized";
            __backend_handle = nullptr;
        }
        else if (shutdown_policy == common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT)
        {
            LOG(DEBUG) << "Object storage backend will be closed on process exit";
        }
        else
        {
            LOG(ERROR) << "Unknown object storage backend shutdown policy";
        }
    }
    return __backend_handle;
}

void * S3ClientWrapper::create_client(const Params & params)
{
    auto start = std::chrono::steady_clock::now();

    common::backend_api::ObjectClientHandle_t client;

    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
    const auto config = params.to_config(initial_params);

    auto ret = _backend_handle->api.create_client(_backend_handle->backend_handle(), &config, &client);
    if (ret != ResponseCode::Success)
    {
        LOG(ERROR) << "Failed to create S3 client for uri " << *params.uri << " and endpoint " << config.endpoint_url;
        throw Exception(ret);
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    LOG(SPAM) << "created s3 client in " << duration.count() << " ms";
    return client;
}

ResponseCode S3ClientWrapper::async_read(const Params & params, common::backend_api::ObjectRequestId_t request_id, const Range & range, char * buffer)
{
    return _backend_handle->api.request_read(_s3_client, params.uri->uri.c_str(), range.to_backend_api_range(), buffer, request_id);
}

common::ResponseCode S3ClientWrapper::async_read_response(std::vector<backend_api::ObjectCompletionEvent_t> & event_buffer, unsigned max_events_to_retrieve)
{
    if (max_events_to_retrieve == 0)
    {
        LOG(WARNING) << "Max events to retrieve is 0";
        return common::ResponseCode::Success;
    }

    event_buffer.resize(max_events_to_retrieve);
    unsigned int out_num_events_retrieved;
    auto ret = _backend_handle->api.wait_for_completions(_s3_client, event_buffer.data(), max_events_to_retrieve, &out_num_events_retrieved, common::backend_api::OBJECT_WAIT_MODE_BLOCK);

    if (ret == common::ResponseCode::Success)
    {
        ASSERT(out_num_events_retrieved >= 0 && out_num_events_retrieved <= max_events_to_retrieve);
        event_buffer.resize(out_num_events_retrieved);
    }
    return ret;
}

common::ResponseCode S3ClientWrapper::list_files(
    const char* prefix,
    int is_recursive,
    common::backend_api::ObjectFileEntry_t** out_entries,
    unsigned* out_num_entries)
{
    return _backend_handle->api.list_files(_s3_client, prefix, is_recursive, out_entries, out_num_entries);
}

void S3ClientWrapper::free_file_list(
    common::backend_api::ObjectFileEntry_t* entries,
    unsigned num_entries)
{
    _backend_handle->api.free_file_list(entries, num_entries);
}

common::backend_api::ObjectShutdownPolicy_t S3ClientWrapper::get_backend_shutdown_policy(std::shared_ptr<S3ClientWrapper::BackendHandle> handle)
{
    ASSERT(handle != nullptr) << "object storage backend handle is not initialized";
    return handle->api.get_backend_shutdown_policy();
}

}; // namespace runai::llm::streamer::common::s3
