#include "s3/s3_mock/s3_mock.h"

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <atomic>

#include "common/s3_credentials/s3_credentials.h"

#include "utils/logging/logging.h"
#include "utils/random/random.h"
#include "utils/env/env.h"

namespace runai::llm::streamer::common::s3
{

std::set<common::backend_api::ObjectClientHandle_t> __mock_clients;
std::map<common::backend_api::ObjectClientHandle_t /* client */, std::set<common::backend_api::ObjectRequestId_t /* request id */>> __mock_client_requests;
std::set<common::backend_api::ObjectClientHandle_t> __mock_unused;
// config (credentials + endpoint) received by the last obj_create_client; endpoint_url is stored under the
// key "endpoint_url". Lets tests assert credentials arrive from runai_file_streamer_set_credentials all the way here.
std::map<std::string, std::string> __mock_last_client_config;
unsigned __mock_response_time_ms = 0;
std::mutex __mutex;
std::atomic<bool> __stopped(false);
std::atomic<bool> __opened(false);
// in-flight window (bytes) reported by obj_get_backend_config("max_inflight_bytes").
// default SIZE_MAX = unbounded (no window); set via runai_mock_s3_set_inflight_window.
std::atomic<size_t> __mock_max_inflight_bytes{static_cast<size_t>(-1)};
// peak per-client in-flight (submitted-but-not-completed) request count, for window tests
std::atomic<size_t> __mock_max_concurrent{0};
std::atomic<size_t> __mock_requests{0};
// requests whose path matched __mock_failing_path at submission time; their completion
// events are reported with FileAccessError instead of the global response code, for
// per-file error-isolation tests. __mock_failing_path is a substring match; empty = none.
// Keyed by (client, request_id): request ids (chunk handles) are only unique per client
// (each worker owns a client and a per-worker handle counter), so a flat request-id set would
// cross-attribute the mark when two clients reuse the same handle number.
std::string __mock_failing_path;
std::set<std::pair<common::backend_api::ObjectClientHandle_t, common::backend_api::ObjectRequestId_t>> __mock_failing_requests;
// Per-attempt completion overrides for deterministic retry tests. The same chunk handle may appear more
// than once sequentially; the override is consumed when its completion is returned, before a retry submits
// that handle again.
unsigned __mock_read_failures_remaining = 0;
common::ResponseCode __mock_read_failure_code = common::ResponseCode::RetryableFileAccessError;
size_t __mock_total_read_requests = 0;
std::map<std::pair<common::backend_api::ObjectClientHandle_t, common::backend_api::ObjectRequestId_t>, common::ResponseCode> __mock_completion_overrides;
// peak number of completion events returned by a single obj_wait_for_completions call,
// for batch (max_responses > 1) tests
std::atomic<size_t> __mock_max_events_per_wait{0};
// when set, obj_wait_for_completions appends a FinishedError sentinel event (handle 0) once
// the ready requests are drained - mimicking the azure/gcs plugins, which pop their responder
// up to max_events and emit the SharedQueue "finished" sentinel. Used to test that the worker
// tolerates the sentinel rather than treating handle 0 as an unknown completion.
std::atomic<bool> __mock_append_finished_sentinel{false};
common::backend_api::ObjectShutdownPolicy_t __shutdown_policy = common::backend_api::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT;

std::vector<std::pair<std::string, size_t>> __mock_files;
common::backend_api::ResponseCode_t __mock_list_files_response = common::ResponseCode::Success;
int __mock_last_list_files_is_recursive = -1;

void runai_s3_mock_set_backend_shutdown_policy(common::backend_api::ObjectShutdownPolicy_t policy)
{
    __shutdown_policy = policy;
}

common::backend_api::ResponseCode_t obj_open_backend(common::backend_api::ObjectBackendHandle_t* out_backend_handle)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);

    auto response_code = common::response_code_from(utils::getenv<int>("RUNAI_STREAMER_S3_MOCK_RESPONSE_CODE", static_cast<int>(common::ResponseCode::Success)));
    if (response_code != common::ResponseCode::Success)
    {
        LOG(ERROR) << "S3 mock backend not opened";
        return response_code;
    }

    if (__opened)
    {
        LOG(ERROR) << "S3 mock backend already opened";
        return common::ResponseCode::UnknownError;
    }

    __opened = true;
    return response_code;
}

common::backend_api::ResponseCode_t obj_close_backend(common::backend_api::ObjectBackendHandle_t backend_handle)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);

    // RUNAI_STREAMER_S3_MOCK_RESPONSE_CODE deliberately does NOT apply here. It exists to inject read
    // failures, and a test that sets it for the duration of a read is very likely to still have it set when
    // the streamer is destroyed - which used to make the streamer's own teardown fail to close the backend,
    // leaving the mock open while the wrapper believed it closed. Every later obj_open_backend then failed
    // with "already opened" for the rest of the process. Failure injection must not break the lifecycle.

    if (!__opened)
    {
        LOG(ERROR) << "S3 mock backend not opened";
        return common::ResponseCode::UnknownError;
    }

    __opened = false;
    return common::ResponseCode::Success;
}

common::backend_api::ObjectShutdownPolicy_t obj_get_backend_shutdown_policy()
{
    return __shutdown_policy;
}

common::backend_api::ResponseCode_t obj_get_backend_config(common::backend_api::ObjectBackendHandle_t /* backend_handle */,
                                                           const char* key,
                                                           char* out_value_buffer,
                                                           unsigned int* in_out_buffer_len)
{
    if (!key || !out_value_buffer || !in_out_buffer_len)
    {
        return common::ResponseCode::UnknownError;
    }

    if (std::strcmp(key, "max_inflight_bytes") == 0)
    {
        const std::string value = std::to_string(__mock_max_inflight_bytes.load());
        const unsigned int needed = static_cast<unsigned int>(value.size()) + 1;   // including null terminator
        if (*in_out_buffer_len < needed)
        {
            *in_out_buffer_len = needed;
            return common::ResponseCode::UnknownError;
        }
        std::memcpy(out_value_buffer, value.c_str(), needed);
        *in_out_buffer_len = static_cast<unsigned int>(value.size());
        return common::ResponseCode::Success;
    }

    return common::ResponseCode::UnknownError;
}

void runai_mock_s3_set_response_time_ms(unsigned milliseconds)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    __mock_response_time_ms = milliseconds;
}

void runai_mock_s3_set_inflight_window(size_t bytes)
{
    __mock_max_inflight_bytes.store(bytes);
}

size_t runai_mock_s3_requests()
{
    return __mock_requests.load();
}

size_t runai_mock_s3_max_concurrent()
{
    return __mock_max_concurrent.load();
}

void runai_mock_s3_set_failing_path(const char* substr)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    __mock_failing_path = (substr != nullptr) ? substr : "";
}

void runai_mock_s3_set_read_failures(unsigned count, common::backend_api::ResponseCode_t response_code)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    __mock_read_failures_remaining = count;
    __mock_read_failure_code = response_code;
}

size_t runai_mock_s3_total_read_requests()
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    return __mock_total_read_requests;
}

size_t runai_mock_s3_max_events_per_wait()
{
    return __mock_max_events_per_wait.load();
}

void runai_mock_s3_set_append_finished_sentinel(bool enabled)
{
    __mock_append_finished_sentinel.store(enabled);
}

common::backend_api::ResponseCode_t obj_create_client(
    common::backend_api::ObjectBackendHandle_t backend_handle,
    const common::backend_api::ObjectClientConfig_t* client_initial_config,
    common::backend_api::ObjectClientHandle_t* out_client_handle)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);

    // capture the credentials/config this client is created with, so tests can verify they arrived from
    // runai_file_streamer_set_credentials through the C++ layer to the plugin. Copy the strings (the config array is the
    // caller's temporary).
    __mock_last_client_config.clear();
    if (client_initial_config != nullptr)
    {
        if (client_initial_config->endpoint_url != nullptr)
        {
            __mock_last_client_config["endpoint_url"] = client_initial_config->endpoint_url;
        }
        for (unsigned i = 0; i < client_initial_config->num_initial_params; ++i)
        {
            const auto & param = client_initial_config->initial_params[i];
            if (param.key != nullptr)
            {
                __mock_last_client_config[param.key] = (param.value != nullptr) ? param.value : "";
            }
        }
    }

    do
    {
        *out_client_handle = reinterpret_cast<common::backend_api::ObjectClientHandle_t>(utils::random::number());
    } while (__mock_clients.count(*out_client_handle) || __mock_unused.count(*out_client_handle));

    __mock_clients.insert(*out_client_handle);

    if (__mock_client_requests.find(*out_client_handle) != __mock_client_requests.end())
    {
        LOG(ERROR) << "Client " << *out_client_handle << " already exists";
        return common::ResponseCode::UnknownError;
    }

    __mock_client_requests[*out_client_handle] = {};

    LOG(DEBUG) << "created client " << *out_client_handle << " - mock size is " << __mock_client_requests.size();
    return common::ResponseCode::Success;
}

common::backend_api::ResponseCode_t obj_remove_client(common::backend_api::ObjectClientHandle_t client_handle)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);

    try
    {
        ASSERT(client_handle) << "No client";
        ASSERT(__mock_client_requests.find(client_handle) != __mock_client_requests.end()) << "Client " << client_handle << " not found";
        __mock_client_requests.erase(client_handle);
        __mock_unused.insert(client_handle);
        LOG(DEBUG) << "Removed S3 client " << client_handle << " - mock size is " << __mock_client_requests.size();
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << "Failed to remove object storage client " << client_handle;
        return common::ResponseCode::UnknownError;
    }
    return common::ResponseCode::Success;
}

common::ResponseCode get_response_code(void * client)
{
    try
    {
        auto response_code = common::response_code_from(utils::getenv<int>("RUNAI_STREAMER_S3_MOCK_RESPONSE_CODE", static_cast<int>(common::ResponseCode::Success)));
        return response_code;
    }
    catch(const std::exception& e)
    {
    }
    return common::ResponseCode::UnknownError;
}

common::backend_api::ResponseCode_t obj_request_read(
    common::backend_api::ObjectClientHandle_t client_handle,
    const char* path,
    common::backend_api::ObjectRange_t range,
    char* destination_buffer,
    common::backend_api::ObjectRequestId_t request_id)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);

    ++__mock_requests;

    if (!__mock_clients.count(client_handle) || __mock_unused.count(client_handle))
    {
        LOG(ERROR) << "Mock client " << client_handle << " not found or unused";
        return common::ResponseCode::UnknownError;
    }

    if (__stopped)
    {
        LOG(DEBUG) <<"Mock s3 is stopped";
        return common::ResponseCode::FinishedError;
    }

    auto r = get_response_code(client_handle);

    ASSERT(__mock_client_requests.find(client_handle) != __mock_client_requests.end()) << "Client " << client_handle << " not found";
    __mock_client_requests[client_handle].insert(request_id);
    ++__mock_total_read_requests;

    if (__mock_read_failures_remaining > 0)
    {
        --__mock_read_failures_remaining;
        __mock_completion_overrides[{client_handle, request_id}] = __mock_read_failure_code;
    }

    // mark this request to fail at completion if its path matches the configured failing path
    if (!__mock_failing_path.empty() && path != nullptr && std::string(path).find(__mock_failing_path) != std::string::npos)
    {
        __mock_failing_requests.insert({client_handle, request_id});
    }

    // track peak per-client in-flight (submitted-but-not-completed) for window tests
    const size_t inflight = __mock_client_requests[client_handle].size();
    if (inflight > __mock_max_concurrent.load())
    {
        __mock_max_concurrent.store(inflight);
    }

    return r;
}

common::backend_api::ResponseCode_t obj_wait_for_completions(common::backend_api::ObjectClientHandle_t client_handle,
                                                              common::backend_api::ObjectCompletionEvent_t* event_buffer,
                                                              unsigned int max_events_to_retrieve,
                                                              unsigned int* out_num_events_retrieved,
                                                              common::backend_api::ObjectWaitMode_t wait_mode)
{
    if (out_num_events_retrieved == nullptr)
    {
        LOG(ERROR) << "output parameter out_num_events_retrieved is null";
        return common::ResponseCode::UnknownError;
    }

    if (event_buffer == nullptr)
    {
        LOG(ERROR) << "output parameter event_buffer is null";
        return common::ResponseCode::UnknownError;
    }

    const auto guard = std::unique_lock<std::mutex>(__mutex);

    if (!__mock_clients.count(client_handle) || __mock_unused.count(client_handle))
    {
        LOG(ERROR) << "Mock client " << client_handle << " not found or unused";
        return common::ResponseCode::UnknownError;
    }

    if (__mock_response_time_ms)
    {
        unsigned counter = 100;
        while (!__stopped && counter > 0)
        {
            LOG(DEBUG) << "Sleeping for " << __mock_response_time_ms << " milliseconds";
            ::usleep(10 * __mock_response_time_ms);
            --counter;
        }
    }

    if (__stopped)
    {
        return common::ResponseCode::FinishedError;
    }

    auto r = get_response_code(client_handle);

    if (__mock_client_requests.find(client_handle) == __mock_client_requests.end())
    {
        LOG(ERROR) << "Mock client " << client_handle << " not found";
        return common::ResponseCode::UnknownError;
    }

    auto & client_requests = __mock_client_requests[client_handle];

    *out_num_events_retrieved = 0;
    for (auto it = client_requests.begin(); it != client_requests.end() && *out_num_events_retrieved < max_events_to_retrieve; )
    {
        // A deterministic per-attempt override wins, then a path-marked failure, then the global code.
        const auto key = std::make_pair(client_handle, *it);
        auto override_it = __mock_completion_overrides.find(key);
        auto code = r;
        if (override_it != __mock_completion_overrides.end())
        {
            code = override_it->second;
            __mock_completion_overrides.erase(override_it);
        }
        else if (__mock_failing_requests.erase(key) != 0)
        {
            code = common::ResponseCode::FileAccessError;
        }
        event_buffer[*out_num_events_retrieved].request_id = *it;
        event_buffer[*out_num_events_retrieved].response_code = code;
        it = client_requests.erase(it);
        ++*out_num_events_retrieved;
    }
    // track the peak count of real completions per wait (before any sentinel), for batch tests
    if (*out_num_events_retrieved > __mock_max_events_per_wait.load())
    {
        __mock_max_events_per_wait.store(*out_num_events_retrieved);
    }

    // mimic azure/gcs: once ready requests are drained, append a FinishedError sentinel event
    // (handle 0) if the buffer still has room - the worker must tolerate this
    if (__mock_append_finished_sentinel.load() && *out_num_events_retrieved < max_events_to_retrieve)
    {
        event_buffer[*out_num_events_retrieved].request_id = 0;
        event_buffer[*out_num_events_retrieved].response_code = common::ResponseCode::FinishedError;
        ++*out_num_events_retrieved;
    }

    if (*out_num_events_retrieved == 0)
    {
        LOG(DEBUG) << "No more ranges to return";
        r = common::ResponseCode::FinishedError;
    }

    return r;
}

int runai_mock_s3_clients()
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);

    return __mock_clients.size();
}

common::backend_api::ResponseCode_t obj_remove_all_clients()
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    if (__mock_clients.size() == __mock_unused.size())
    {
        __mock_clients.clear();
        __mock_unused.clear();
        __mock_client_requests.clear();
    }
    return common::ResponseCode::Success;
}

common::backend_api::ResponseCode_t obj_cancel_all_reads()
{
    __stopped = true;
    LOG(DEBUG) << "Stopped S3 clients ";
    return common::ResponseCode::Success;
}

void runai_mock_s3_set_files(const char** paths, const size_t* sizes, unsigned count)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    __mock_files.clear();
    for (unsigned i = 0; i < count; ++i)
    {
        __mock_files.emplace_back(paths[i], sizes[i]);
    }
}

void runai_mock_s3_set_list_files_response(common::backend_api::ResponseCode_t response)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    __mock_list_files_response = response;
}

int runai_mock_s3_last_list_files_is_recursive()
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);
    return __mock_last_list_files_is_recursive;
}

common::backend_api::ResponseCode_t obj_list_files(
    common::backend_api::ObjectClientHandle_t client_handle,
    const char* prefix,
    int is_recursive,
    common::backend_api::ObjectFileEntry_t** out_entries,
    unsigned* out_num_entries)
{
    const auto guard = std::unique_lock<std::mutex>(__mutex);

    __mock_last_list_files_is_recursive = is_recursive;

    if (!__mock_clients.count(client_handle) || __mock_unused.count(client_handle))
    {
        LOG(ERROR) << "Mock client " << client_handle << " not found or unused";
        return common::ResponseCode::UnknownError;
    }

    if (__mock_list_files_response != common::ResponseCode::Success)
    {
        return __mock_list_files_response; // out_entries left untouched on error
    }

    *out_num_entries = static_cast<unsigned>(__mock_files.size());
    if (__mock_files.empty())
    {
        *out_entries = nullptr;
        return common::ResponseCode::Success;
    }

    auto* entries = new common::backend_api::ObjectFileEntry_t[__mock_files.size()];
    for (size_t i = 0; i < __mock_files.size(); ++i)
    {
        entries[i].path = ::strdup(__mock_files[i].first.c_str());
        entries[i].size = __mock_files[i].second;
    }
    *out_entries = entries;
    return common::ResponseCode::Success;
}

void obj_free_file_list(common::backend_api::ObjectFileEntry_t* entries, unsigned num_entries)
{
    if (entries == nullptr)
    {
        return;
    }
    for (unsigned i = 0; i < num_entries; ++i)
    {
        ::free(entries[i].path);
    }
    delete[] entries;
}

const char* runai_mock_s3_last_client_config_value(const char* key)
{
    // Return a pointer into thread-local storage, not into __mock_last_client_config: the map is
    // only stable while __mutex is held, so a raw pointer into it would dangle once the lock is
    // released (a later obj_create_client / cleanup could rehash or clear it). The copy is made
    // under the lock and stays valid until this thread's next call.
    thread_local std::string value;

    const auto guard = std::unique_lock<std::mutex>(__mutex);
    if (key == nullptr)
    {
        return nullptr;
    }
    auto it = __mock_last_client_config.find(key);
    if (it == __mock_last_client_config.end())
    {
        return nullptr;
    }
    value = it->second;
    return value.c_str();
}

void runai_mock_s3_cleanup()
{
    runai_mock_s3_set_response_time_ms(0);
    __stopped = false;
    __mock_max_inflight_bytes.store(static_cast<size_t>(-1));
    __mock_max_concurrent.store(0);
    __mock_requests.store(0);
    __mock_max_events_per_wait.store(0);
    __mock_append_finished_sentinel.store(false);
    runai_s3_mock_set_backend_shutdown_policy(common::backend_api::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT);

    const auto guard = std::unique_lock<std::mutex>(__mutex);
    __mock_failing_path.clear();
    __mock_failing_requests.clear();
    __mock_read_failures_remaining = 0;
    __mock_read_failure_code = common::ResponseCode::RetryableFileAccessError;
    __mock_total_read_requests = 0;
    __mock_completion_overrides.clear();
    __mock_files.clear();
    __mock_last_client_config.clear();
    __mock_list_files_response = common::ResponseCode::Success;
    __mock_last_list_files_is_recursive = -1;
}

bool runai_mock_s3_is_shutdown()
{
    return !__opened;
}

}; //namespace runai::llm::streamer::common::s3
