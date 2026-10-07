/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "common/range/range.h"
#include "common/response_code/response_code.h"
#include "common/storage_uri/storage_uri.h"
#include "common/backend_api/object_storage/object_storage.h"
#include "common/backend_api/response/response.h"
#include "common/s3_credentials/s3_credentials.h"

#include "utils/dylib/dylib.h"
#include "utils/semver/semver.h"

namespace runai::llm::streamer::common::s3
{

static const std::string lib_streamer_s3_so_name = "libstreamers3.so";
static const std::string lib_streamer_gcs_so_name = "libstreamergcs.so";
static const std::string lib_streamer_azure_so_name = "libstreamerazure.so";
static const std::string obj_plugin_s3_name = "s3";
static const std::string obj_plugin_gcs_name = "gcs";
static const std::string obj_plugin_azure_name = "azure";

enum struct PluginID {
    GCS,
    S3,
    AZURE
};

/**
 * Type-safe enum for encapsulating plugin information (name, shared library)
 */
class ObjectPluginType {
private:
    PluginID _id;
    std::string _name;
    std::string _so_name;

    ObjectPluginType(PluginID id, std::string name, std::string so_name)
        : _id(id), _name(name), _so_name(so_name) {}

public:
    static const ObjectPluginType ObjStorageGCS;
    static const ObjectPluginType ObjStorageS3;
    static const ObjectPluginType ObjStorageAzure;

    std::string name() const { return _name; }
    std::string so_name() const { return _so_name; }
    constexpr PluginID id() const { return _id; }

    bool operator==(const ObjectPluginType& other) const {
        return _id == other._id;
    }
};

struct S3ClientWrapper
{
      struct Params
      {
         Params()
         {}

         // concurrent_readers has no default: the streamer resolves it once, in Config, and every
         // caller states it. A plugin must never have to decide it.
         Params(std::shared_ptr<StorageUri> uri, const Credentials & credentials, size_t chunk_bytesize,
                unsigned concurrent_readers);

         Params(std::shared_ptr<StorageUri> uri, size_t chunk_bytesize, unsigned concurrent_readers) :
             Params(uri, Credentials(), chunk_bytesize, concurrent_readers)
         {}

         bool valid() const { return (uri.get() != nullptr); }

         size_t chunk_bytesize;
         std::shared_ptr<StorageUri> uri;
         Credentials credentials;

         // How many clients the caller will run at once.
         unsigned concurrent_readers;
         const common::backend_api::ObjectClientConfig_t to_config(std::vector<common::backend_api::ObjectConfigParam_t> & initial_params) const;

       private:
         std::string _endpoint;
      };

      // Plugin C-ABI entry points. Resolved once (see resolve_api), because dlsym takes the process-wide
      // dynamic-linker lock: resolving per call would serialize every worker on the hot read/response path.
      // The pointers live and die with the owning BackendHandle's dylib_ptr, so a teardown+recreate that
      // swaps to a different plugin re-resolves against the correct library (no stale/dangling pointer).
      struct Api
      {
         common::ResponseCode (*open_backend)(common::backend_api::ObjectBackendHandle_t*) = nullptr;
         common::ResponseCode (*close_backend)(common::backend_api::ObjectBackendHandle_t) = nullptr;
         common::ResponseCode (*create_client)(common::backend_api::ObjectBackendHandle_t, const common::backend_api::ObjectClientConfig_t*, common::backend_api::ObjectClientHandle_t*) = nullptr;
         common::backend_api::ResponseCode_t (*remove_client)(common::backend_api::ObjectClientHandle_t) = nullptr;
         common::backend_api::ResponseCode_t (*remove_all_clients)() = nullptr;
         common::backend_api::ResponseCode_t (*cancel_all_reads)() = nullptr;
         common::ResponseCode (*request_read)(common::backend_api::ObjectClientHandle_t, const char*, common::backend_api::ObjectRange_t, char*, common::backend_api::ObjectRequestId_t) = nullptr;
         common::ResponseCode (*wait_for_completions)(common::backend_api::ObjectClientHandle_t, common::backend_api::ObjectCompletionEvent_t*, unsigned int, unsigned int*, common::backend_api::ObjectWaitMode_t) = nullptr;
         common::backend_api::ResponseCode_t (*list_files)(common::backend_api::ObjectClientHandle_t, const char*, int, common::backend_api::ObjectFileEntry_t**, unsigned*) = nullptr;
         void (*free_file_list)(common::backend_api::ObjectFileEntry_t*, unsigned) = nullptr;
         common::backend_api::ObjectShutdownPolicy_t (*get_backend_shutdown_policy)() = nullptr;
         common::backend_api::ResponseCode_t (*get_backend_config)(common::backend_api::ObjectBackendHandle_t, const char*, char*, unsigned int*) = nullptr;
      };

      struct BackendHandle
      {
         BackendHandle(const Params & params);

         ~BackendHandle();

         common::backend_api::ObjectBackendHandle_t backend_handle() const;

         static const ObjectPluginType get_libstreamers_plugin_type(const std::shared_ptr<common::s3::StorageUri> & uri);

         std::shared_ptr<utils::Dylib> open_object_storage_impl(const Params & params);

         std::shared_ptr<utils::Dylib> dylib_ptr;

         // Entry points into dylib_ptr, resolved once by the constructor and used for the handle's lifetime.
         Api api;

       private:
         // Resolve every entry point from dylib_ptr into `api`. All are required: a missing one throws and
         // fails backend open (fail fast, before any client or read), so this doubles as a full-API
         // conformance check on the plugin. Called by the constructor, before obj_open_backend.
         void resolve_api();

         common::backend_api::ObjectBackendHandle_t _backend_handle;
      };

      S3ClientWrapper(const Params & params);
      ~S3ClientWrapper();

      // request to read a continous range into a buffer
      // the range is divided into sub ranges, which will generate response whenever a full sub range is fully read
      // ranges - list of sub ranges
      // chunk_bytesize - size of chunk for reading in multi parts (minimal size is 5 MB)

      common::ResponseCode async_read(const Params & params, backend_api::ObjectRequestId_t request_id, const Range & ranges, char * buffer);
      common::ResponseCode async_read_response(std::vector<backend_api::ObjectCompletionEvent_t> & event_buffer, unsigned max_events_to_retrieve);

      // In-flight window (bytes) the backend advertises for submission throttling, via
      // obj_get_backend_config("max_inflight_bytes"). Returns SIZE_MAX (unbounded) when
      // the plugin does not provide the key (e.g. gcs/azure).
      size_t max_inflight_bytes();

      common::ResponseCode list_files(const char* prefix, int is_recursive,
                                      backend_api::ObjectFileEntry_t** out_entries,
                                      unsigned* out_num_entries);
      void free_file_list(backend_api::ObjectFileEntry_t* entries, unsigned num_entries);

      // stop - stops the responder of each S3 client, in order to notify callers which sent a request and are waiting for a response
      //        required for stopping the threadpool workers, which are bloking on the client responder
      static void stop();

      // destroy S3 all clients
      static void shutdown();

      static constexpr size_t min_chunk_bytesize = 5 * 1024 * 1024;
      static constexpr size_t default_chunk_bytesize = 8 * 1024 * 1024;

 private:
      void * create_client(const Params & params);
      enum class ManageBackendHandleOp
      {
         CREATE,
         DESTROY,
      };
      static std::shared_ptr<BackendHandle> manage_backend_handle(const Params & params, ManageBackendHandleOp op);
      static common::backend_api::ObjectShutdownPolicy_t get_backend_shutdown_policy(std::shared_ptr<BackendHandle> handle);

 private:
      static std::mutex _backend_handle_mutex;
      std::shared_ptr<BackendHandle> _backend_handle;

      // Handle to s3 client
      void * _s3_client;
};

}; //namespace runai::llm::streamer::common::s3
