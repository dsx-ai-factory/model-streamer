#include "streamer/streamer.h"

#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/backend_api/object_storage/object_storage.h"
#include "common/response_code/response_code.h"
#include "common/submission/submission_id.h"
#include "common/s3_credentials/s3_credentials.h"

#include "utils/logging/logging.h"
#include "utils/random/random.h"
#include "utils/env/env.h"
#include "utils/dylib/dylib.h"
#include "utils/temp/env/env.h"
#include "utils/temp/file/file.h"
#include "utils/fdlimit/fdlimit.h"

namespace runai::llm::streamer
{

namespace
{

// Test adapters over the multi-request C API. These single-submission tests drain by a known count, so
// submit() discards the returned submission id and next_response() waits for the next sub-range. A generous
// finite timeout (not 0) is used so that, if a cancellation/timing test ever produces fewer responses than
// expected, the test fails visibly instead of blocking forever (the multi-request API has no finish-on-drain).
constexpr unsigned RESPONSE_TIMEOUT_MS = 60000;

// Keeps the classic per-file argument shape so the tests below are unchanged, and adapts it to the range
// API: each file's sub ranges tile [file_offsets[i], file_offsets[i] + bytesizes[i]) in order, and every
// file is written consecutively into the single buffer at dsts[0] - the layout the previous API implied.
inline int submit(void * streamer, unsigned num_files, const char ** paths, size_t * file_offsets,
                  size_t * bytesizes, void ** dsts, unsigned * num_sizes, size_t ** internal_sizes)
{
    (void)bytesizes;   // implied by the sub range sizes

    std::vector<unsigned> num_ranges(num_files);
    std::vector<size_t> range_offsets;
    std::vector<size_t> range_sizes;
    std::vector<void *> range_dsts;

    char * dst = static_cast<char *>(dsts[0]);
    for (unsigned i = 0; i < num_files; ++i)
    {
        num_ranges[i] = num_sizes[i];

        size_t offset = file_offsets[i];
        for (unsigned j = 0; j < num_sizes[i]; ++j)
        {
            const size_t size = internal_sizes[i][j];
            range_offsets.push_back(offset);
            range_sizes.push_back(size);
            range_dsts.push_back(dst);
            offset += size;
            dst += size;
        }
    }

    SubmissionId submission_id = 0;
    return runai_file_streamer_request(streamer, &submission_id, num_files, paths, num_ranges.data(),
                         range_offsets.data(), range_sizes.data(), range_dsts.data(), RunaiFileStreamerDevice{});
}

inline int next_response(void * streamer, unsigned * file_index, unsigned * index)
{
    SubmissionId submission_id = 0;
    int submission_done = 0;
    return runai_file_streamer_response(streamer, &submission_id, file_index, index, &submission_done, RESPONSE_TIMEOUT_MS);
}

struct StreamerTest : ::testing::Test
{
    StreamerTest() :
        _concurrency(utils::random::number<int>(1, 16)),
        _size("RUNAI_STREAMER_CONCURRENCY", _concurrency),
        _chunk_bytesize("RUNAI_STREAMER_CHUNK_BYTESIZE", utils::random::number<int>(1, 1024)),
        s3_path("s3://" + utils::random::string() + "/" + utils::random::string()),
        credentials(
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr)),
        num_files(utils::random::number(1, 10)),
        s3_paths(num_files),
        file_names(num_files),
        file_offsets(num_files),
        sizes(num_files),
        dsts(num_files),
        num_ranges(num_files),
        range_sizes(num_files),
        internal_sizes(num_files),
        expected_response(num_files)
    {
        LOG(DEBUG) << "num_files: " << num_files;
        size_t dst_size = 0;
        num_expected_responses = 0;
        std::string bucket_name = utils::random::string();
        for (unsigned i = 0; i < num_files; ++i)
        {
            s3_paths[i] = "s3://" + bucket_name + "/" + utils::random::string();
            file_names[i] = s3_paths[i].c_str();
            sizes[i] = utils::random::number(10000000, 200000000);
            LOG(DEBUG) << "sizes[i]: " << sizes[i];
            dst_size += sizes[i];

            file_offsets[i] = utils::random::number<size_t>(0, sizes[i] - 1);

            num_ranges[i] = utils::random::number(1, 100);
            range_sizes[i] =  utils::random::chunks(sizes[i], num_ranges[i]);
            internal_sizes[i] = range_sizes[i].data();

            num_expected_responses += num_ranges[i];

            for (unsigned request_index = 0; request_index < num_ranges[i]; ++request_index)
            {
                expected_response[i].insert(request_index);
            }
        }
        LOG(DEBUG) << "num_expected_responses: " << num_expected_responses;
        dst.resize(dst_size);
        dsts[0] = static_cast<void *>(dst.data());
    }

    ~StreamerTest()
    {
        utils::Dylib dylib("libstreamers3.so");
        auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
        mock_cleanup();
    }

 protected:
    int _concurrency;
    utils::temp::Env _size;
    utils::temp::Env _chunk_bytesize;
    utils::temp::Env _block_bytesize;
    std::string s3_path;
    common::s3::Credentials credentials;

    // Apply `credentials` to the streamer (credentials are streamer-scoped, set once via runai_file_streamer_set_credentials).
    static void apply_credentials(void * streamer, const common::s3::Credentials & credentials)
    {
        std::vector<const char *> keys, values;
        for (const auto & entry : credentials.params())
        {
            keys.push_back(entry.first.c_str());
            values.push_back(entry.second.c_str());
        }
        runai_file_streamer_set_credentials(streamer, keys.data(), values.data(), static_cast<unsigned>(keys.size()));
    }

    unsigned num_files;
    std::vector<std::string> s3_paths;
    std::vector<const char *> file_names;
    std::vector<size_t> file_offsets;
    std::vector<size_t> sizes;
    std::vector<void *> dsts;
    std::vector<unsigned> num_ranges;
    std::vector<std::vector<size_t>> range_sizes;
    std::vector<size_t *> internal_sizes;
    unsigned num_expected_responses;
    std::vector<std::set<unsigned>> expected_response;
    std::vector<unsigned char> dst;
};

} // namespace

TEST_F(StreamerTest, Async_Read)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto set_backend_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_backend_shutdown_policy(utils::random::boolean() ? common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN : common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT);

    bool use_credentials = utils::random::boolean();
    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    if (use_credentials)
    {
        apply_credentials(streamer, credentials);
        res = submit(streamer,
                        num_files,
                        file_names.data(),
                        file_offsets.data(),
                        sizes.data(),
                        dsts.data(),
                        num_ranges.data(),
                        internal_sizes.data());
    }
    else
    {
        res = submit(streamer,
                        num_files,
                        file_names.data(),
                        file_offsets.data(),
                        sizes.data(),
                        dsts.data(),
                        num_ranges.data(),
                        internal_sizes.data());
    }

    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    // wait for all the responses to arrive
    unsigned r;
    unsigned file_index;
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        r = utils::random::number();
        file_index = utils::random::number();
        auto response_code = next_response(streamer, &file_index, &r);
        EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
        if (response_code != static_cast<int>(common::ResponseCode::Success))
        {
            break;
        }
        EXPECT_LT(file_index, num_files);
        EXPECT_EQ(expected_response[file_index].count(r), 1);
        expected_response[file_index].erase(r);
    }
    runai_file_streamer_end(streamer);
    mock_cleanup();
    EXPECT_EQ(verify_mock(), 0);
}

// End-to-end: credentials set via runai_file_streamer_set_credentials must reach the plugin's obj_create_client as the
// client config (credentials -> initial_params, endpoint -> endpoint_url). The s3 mock records the config
// of the last created client so we can assert the exact values arrived.
TEST_F(StreamerTest, Credentials_Reach_Plugin)
{
    utils::Dylib dylib("libstreamers3.so");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto last_config = dylib.dlsym<const char*(*)(const char*)>("runai_mock_s3_last_client_config_value");

    // known, fully-populated credentials so we can assert each value arrives at the plugin
    const common::s3::Credentials known("AKIAEXAMPLE", "secret-123", "token-456", "us-west-2", "https://s3.example.com");

    void * streamer;
    EXPECT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    apply_credentials(streamer, known);

    auto res = submit(streamer,
                    num_files,
                    file_names.data(),
                    file_offsets.data(),
                    sizes.data(),
                    dsts.data(),
                    num_ranges.data(),
                    internal_sizes.data());
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    // drain all responses so the client is built (obj_create_client called) before we assert
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        unsigned r = utils::random::number();
        unsigned file_index = utils::random::number();
        auto response_code = next_response(streamer, &file_index, &r);
        EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
        if (response_code != static_cast<int>(common::ResponseCode::Success))
        {
            break;
        }
    }

    // every credential applied via runai_file_streamer_set_credentials reached obj_create_client's client config; endpoint
    // is carried as endpoint_url, the other fields as initial_params under their canonical keys
    auto expect_config = [&](const char * key, const char * value)
    {
        const char * got = last_config(key);
        EXPECT_NE(got, nullptr) << "missing client config parameter " << key;
        if (got != nullptr)
        {
            EXPECT_STREQ(got, value);
        }
    };
    expect_config("access_key_id", "AKIAEXAMPLE");
    expect_config("secret_access_key", "secret-123");
    expect_config("session_token", "token-456");
    expect_config("region", "us-west-2");
    expect_config("endpoint_url", "https://s3.example.com");

    runai_file_streamer_end(streamer);
    mock_cleanup();
    EXPECT_EQ(verify_mock(), 0);
}

// An unservable FILESYSTEM strategy must not reject an OBJECT-STORAGE read.
//
// The strategy names a filesystem engine and has nothing to say about S3, but resolution used to run
// on every submission - so setting RUNAI_STREAMER_FS_STRATEGY to something this host cannot serve
// failed every object-storage request, for a reason that cannot apply to it.
TEST_F(StreamerTest, Object_Storage_Ignores_The_Filesystem_Strategy)
{
    // libaio_direct has no engine, so this list can never resolve. A filesystem submission would be
    // rejected by it; an object-storage one must not even consult it.
    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("libaio_direct"));

    utils::Dylib dylib("libstreamers3.so");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");

    void * streamer;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    const auto res = submit(streamer,
                            num_files,
                            file_names.data(),
                            file_offsets.data(),
                            sizes.data(),
                            dsts.data(),
                            num_ranges.data(),
                            internal_sizes.data());

    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success))
        << "an unservable filesystem strategy rejected an object-storage submission";

    unsigned r = 0;
    unsigned file_index = 0;
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        EXPECT_EQ(next_response(streamer, &file_index, &r), static_cast<int>(common::ResponseCode::Success));
    }

    runai_file_streamer_end(streamer);
    mock_cleanup();
}

TEST_F(StreamerTest, Async_Read_Bounded_By_Window)
{
    utils::Dylib dylib("libstreamers3.so");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto set_window = dylib.dlsym<void(*)(size_t)>("runai_mock_s3_set_inflight_window");
    auto max_concurrent = dylib.dlsym<size_t(*)()>("runai_mock_s3_max_concurrent");

    // Bound the in-flight window. The effective chunk size is clamped to the 5 MiB S3
    // minimum, so the per-client window is at most window_bytes / 5 MiB chunks.
    const size_t min_chunk = 5 * 1024 * 1024;
    const size_t window_chunks = 5;
    set_window(window_chunks * min_chunk);

    void * streamer;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    auto res = submit(streamer,
                    num_files,
                    file_names.data(),
                    file_offsets.data(),
                    sizes.data(),
                    dsts.data(),
                    num_ranges.data(),
                    internal_sizes.data());
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    // every sub-request still completes successfully under the bounded window
    unsigned r;
    unsigned file_index;
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        r = utils::random::number();
        file_index = utils::random::number();
        auto response_code = next_response(streamer, &file_index, &r);
        EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
        if (response_code != static_cast<int>(common::ResponseCode::Success))
        {
            break;
        }
        EXPECT_LT(file_index, num_files);
        EXPECT_EQ(expected_response[file_index].count(r), 1);
        expected_response[file_index].erase(r);
    }

    // the per-client in-flight never exceeded the configured window (windowing is enforced),
    // and the window was actually exercised (something was in flight)
    const size_t peak = max_concurrent();
    EXPECT_LE(peak, window_chunks);
    EXPECT_GT(peak, 0u);

    runai_file_streamer_end(streamer);
    mock_cleanup();
    EXPECT_EQ(verify_mock(), 0);
}

TEST_F(StreamerTest, Async_Read_Per_File_Error_Isolation)
{
    if (num_files < 2)
    {
        GTEST_SKIP() << "needs at least two files to observe per-file isolation";
    }

    utils::Dylib dylib("libstreamers3.so");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto set_failing_path = dylib.dlsym<void(*)(const char*)>("runai_mock_s3_set_failing_path");

    // fail every read of file 0 (its object key is unique); all other files must still succeed
    const unsigned failing_file = 0;
    set_failing_path(s3_paths[failing_file].c_str());

    void * streamer;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    auto res = submit(streamer,
                    num_files,
                    file_names.data(),
                    file_offsets.data(),
                    sizes.data(),
                    dsts.data(),
                    num_ranges.data(),
                    internal_sizes.data());
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    unsigned failed = 0;
    unsigned succeeded = 0;
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        unsigned r = utils::random::number();
        unsigned file_index = utils::random::number();
        auto response_code = next_response(streamer, &file_index, &r);
        ASSERT_LT(file_index, num_files);

        if (file_index == failing_file)
        {
            // the failing file's ranges report the error - isolated to this file
            EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::FileAccessError));
            ++failed;
        }
        else
        {
            // every other file completes successfully with its expected range indices
            EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
            EXPECT_EQ(expected_response[file_index].count(r), 1);
            expected_response[file_index].erase(r);
            ++succeeded;
        }
    }

    // exactly the failing file's ranges failed; all other files' ranges succeeded
    EXPECT_EQ(failed, num_ranges[failing_file]);
    EXPECT_EQ(succeeded, num_expected_responses - num_ranges[failing_file]);
    for (unsigned i = 0; i < num_files; ++i)
    {
        if (i != failing_file)
        {
            EXPECT_TRUE(expected_response[i].empty());
        }
    }

    runai_file_streamer_end(streamer);
    mock_cleanup();
    EXPECT_EQ(verify_mock(), 0);
}

TEST_F(StreamerTest, Async_Read_Batched_Completions)
{
    utils::Dylib dylib("libstreamers3.so");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto max_events_per_wait = dylib.dlsym<size_t(*)()>("runai_mock_s3_max_events_per_wait");

    // raise the internal batch size so the worker requests many completions per wait (the
    // production default is 1); the mock's window is unbounded, so every chunk is in flight
    // before the first wait and a single obj_wait_for_completions returns many completions
    utils::temp::Env max_responses("RUNAI_STREAMER_INTERNAL_MAX_RESPONSES", 64UL);

    void * streamer;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    auto res = submit(streamer,
                    num_files,
                    file_names.data(),
                    file_offsets.data(),
                    sizes.data(),
                    dsts.data(),
                    num_ranges.data(),
                    internal_sizes.data());
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    // every sub-request still completes successfully when completions are drained in batches
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        unsigned r = utils::random::number();
        unsigned file_index = utils::random::number();
        auto response_code = next_response(streamer, &file_index, &r);
        EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
        if (response_code != static_cast<int>(common::ResponseCode::Success))
        {
            break;
        }
        EXPECT_LT(file_index, num_files);
        EXPECT_EQ(expected_response[file_index].count(r), 1);
        expected_response[file_index].erase(r);
    }

    // at least one wait returned more than one completion (batch drain was actually exercised)
    EXPECT_GT(max_events_per_wait(), 1u);

    runai_file_streamer_end(streamer);
    mock_cleanup();
    EXPECT_EQ(verify_mock(), 0);
}

TEST_F(StreamerTest, Async_Read_Tolerates_Finished_Sentinel)
{
    utils::Dylib dylib("libstreamers3.so");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto set_append_sentinel = dylib.dlsym<void(*)(bool)>("runai_mock_s3_set_append_finished_sentinel");

    // raise the internal batch size (>1) so obj_wait_for_completions has room to append the
    // sentinel after the ready completions - at the production default of 1 the buffer is full
    // after one real event and the sentinel path would never be exercised
    utils::temp::Env max_responses("RUNAI_STREAMER_INTERNAL_MAX_RESPONSES", 64UL);

    // mimic azure/gcs: obj_wait_for_completions appends a FinishedError sentinel (handle 0)
    // once the ready completions drain. The worker must skip it, not fail on the 0 handle.
    set_append_sentinel(true);

    void * streamer;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    auto res = submit(streamer,
                    num_files,
                    file_names.data(),
                    file_offsets.data(),
                    sizes.data(),
                    dsts.data(),
                    num_ranges.data(),
                    internal_sizes.data());
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    // every sub-request still completes successfully despite the interleaved sentinel events
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        unsigned r = utils::random::number();
        unsigned file_index = utils::random::number();
        auto response_code = next_response(streamer, &file_index, &r);
        EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
        if (response_code != static_cast<int>(common::ResponseCode::Success))
        {
            break;
        }
        EXPECT_LT(file_index, num_files);
        EXPECT_EQ(expected_response[file_index].count(r), 1);
        expected_response[file_index].erase(r);
    }
    for (unsigned i = 0; i < num_files; ++i)
    {
        EXPECT_TRUE(expected_response[i].empty());
    }

    runai_file_streamer_end(streamer);
    mock_cleanup();
    EXPECT_EQ(verify_mock(), 0);
}

TEST_F(StreamerTest, Increase_Insufficient_Fd_Limit)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto set_backend_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_backend_shutdown_policy(common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT);

    for (bool use_credentials : { true, false })
    {
        auto concurrency = utils::getenv<int>("RUNAI_STREAMER_CONCURRENCY");

        const auto insufficient_fd_limit = utils::random::number<rlim_t>(50, concurrency * 64 -1);
        utils::FdLimitSetter fd_limit(insufficient_fd_limit);
        void * streamer;
        auto res = runai_file_streamer_start(&streamer);
        EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

        if (use_credentials)
        {
            apply_credentials(streamer, credentials);
            res = submit(streamer,
                            num_files,
                            file_names.data(),
                            file_offsets.data(),
                            sizes.data(),
                            dsts.data(),
                            num_ranges.data(),
                            internal_sizes.data());
        }
        else
        {
            res = submit(streamer,
                            num_files,
                            file_names.data(),
                            file_offsets.data(),
                            sizes.data(),
                            dsts.data(),
                            num_ranges.data(),
                            internal_sizes.data());
        }
        EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

        runai_file_streamer_end(streamer);
        EXPECT_EQ(verify_mock(), 0);

        // verify that fd limit was restored
        EXPECT_EQ(utils::get_cur_file_descriptors(), insufficient_fd_limit);
        mock_cleanup();
    }
}

TEST_F(StreamerTest, Stop_Before_Async_Read)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto stop_mock = dylib.dlsym<common::backend_api::ResponseCode_t(*)()>("obj_cancel_all_reads");
    auto mock_cleanup = dylib.dlsym<void(*)()>("runai_mock_s3_cleanup");
    auto set_backend_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_backend_shutdown_policy(utils::random::boolean() ? common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN : common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT);

    for (bool use_credentials : { true, false })
    {
        void * streamer;
        auto res = runai_file_streamer_start(&streamer);
        EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

        stop_mock();

        if (use_credentials)
        {
            apply_credentials(streamer, credentials);
            res = submit(streamer,
                            num_files,
                            file_names.data(),
                            file_offsets.data(),
                            sizes.data(),
                            dsts.data(),
                            num_ranges.data(),
                            internal_sizes.data());
        }
        else
        {
            res = submit(streamer,
                            num_files,
                            file_names.data(),
                            file_offsets.data(),
                            sizes.data(),
                            dsts.data(),
                            num_ranges.data(),
                            internal_sizes.data());
        }

        // request was not sent to the S3 server
        unsigned r;
        unsigned file_index;
        EXPECT_EQ(next_response(streamer, &file_index, &r), static_cast<int>(common::ResponseCode::FinishedError));

        LOG(INFO) << "******************************* Ending streamer";
        runai_file_streamer_end(streamer);
        EXPECT_EQ(verify_mock(), 0);

        mock_cleanup();
    }
}

TEST_F(StreamerTest, End_During_Async_Read)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto mock_cleanup = dylib.dlsym<common::backend_api::ResponseCode_t(*)()>("obj_remove_all_clients");
    auto set_backend_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_backend_shutdown_policy(utils::random::boolean() ? common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN : common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT);

    for (bool use_credentials : { true, false })
    {
        auto mock_response_time = dylib.dlsym<void(*)(unsigned)>("runai_mock_s3_set_response_time_ms");
        unsigned delay_ms = 1000;
        mock_response_time(delay_ms);

        void * streamer;
        auto res = runai_file_streamer_start(&streamer);
        EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

        if (use_credentials)
        {
            apply_credentials(streamer, credentials);
            res = submit(streamer,
                            num_files,
                            file_names.data(),
                            file_offsets.data(),
                            sizes.data(),
                            dsts.data(),
                            num_ranges.data(),
                            internal_sizes.data());
        }
        else
        {
            res = submit(streamer,
                            num_files,
                            file_names.data(),
                            file_offsets.data(),
                            sizes.data(),
                            dsts.data(),
                            num_ranges.data(),
                            internal_sizes.data());
        }

        ::usleep(utils::random::number(300));

        runai_file_streamer_end(streamer);

        EXPECT_EQ(verify_mock(), 0);

        mock_cleanup();
    }
}

TEST_F(StreamerTest, Multiple_Files)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto set_backend_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_backend_shutdown_policy(utils::random::boolean() ? common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN : common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT);

    void * streamer;
    EXPECT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    auto res = submit(streamer,
                             num_files,
                             file_names.data(),
                             file_offsets.data(),
                             sizes.data(),
                             dsts.data(),
                             num_ranges.data(),
                             internal_sizes.data());

    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    // wait for all the responses to arrive
    unsigned r;
    unsigned file_index;
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        r = utils::random::number();
        file_index = utils::random::number();
        auto response_code = next_response(streamer, &file_index, &r);
        EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
        if (response_code != static_cast<int>(common::ResponseCode::Success))
        {
            break;
        }
        EXPECT_LT(file_index, num_files);
        EXPECT_EQ(expected_response[file_index].count(r), 1);
        expected_response[file_index].erase(r);
    }

    runai_file_streamer_end(streamer);
    EXPECT_EQ(verify_mock(), 0);
}

// A submission must pick one backend kind (Streamer::lock_object_plugin rejects a mix), but a STREAMER
// serves both: BackendPools holds one pool per kind, created lazily, so filesystem and object-storage
// submissions coexist on the same handle. Both orders are exercised, because the pools are created
// lazily and by different paths - the filesystem pool on first push, the object-storage pool by the
// plugin lock - so an ordering bug would show up in only one of them.
TEST_F(StreamerTest, Filesystem_And_Object_Storage_Submissions_Coexist)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");

    const auto fs_data = utils::random::buffer(utils::random::number(100, 1000));
    utils::temp::File fs_file(fs_data);

    // Read the filesystem file in one range, and drain it, checking the bytes actually arrived.
    auto read_filesystem = [&](void * streamer)
    {
        std::vector<unsigned char> dst(fs_data.size());
        void * dst_ptr = dst.data();
        const char * path = fs_file.path.c_str();
        unsigned num_ranges = 1;
        size_t offset = 0;
        size_t size = fs_data.size();

        SubmissionId submission_id = 0;
        EXPECT_EQ(runai_file_streamer_request(streamer, &submission_id, 1, &path, &num_ranges, &offset, &size, &dst_ptr, RunaiFileStreamerDevice{}),
                  static_cast<int>(common::ResponseCode::Success));

        unsigned file_index = 0;
        unsigned range_index = 0;
        EXPECT_EQ(next_response(streamer, &file_index, &range_index),
                  static_cast<int>(common::ResponseCode::Success));
        EXPECT_EQ(dst, std::vector<unsigned char>(fs_data.begin(), fs_data.end()));
    };

    // Read the fixture's object-storage files, and drain every expected response.
    auto read_object_storage = [&](void * streamer)
    {
        EXPECT_EQ(submit(streamer, num_files, file_names.data(), file_offsets.data(), sizes.data(),
                         dsts.data(), num_ranges.data(), internal_sizes.data()),
                  static_cast<int>(common::ResponseCode::Success));

        for (unsigned i = 0; i < num_expected_responses; ++i)
        {
            unsigned file_index = 0;
            unsigned range_index = 0;
            const auto response_code = next_response(streamer, &file_index, &range_index);
            EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
            if (response_code != static_cast<int>(common::ResponseCode::Success))
            {
                break;
            }
        }
    };

    void * streamer;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    read_filesystem(streamer);
    read_object_storage(streamer);

    runai_file_streamer_end(streamer);
    EXPECT_EQ(verify_mock(), 0);
}

// The reverse order, as its own test rather than a second streamer in the one above: the object-storage
// mock's backend handle is process-global, so two streamers in a single test exercise the plugin
// teardown/reopen lifecycle instead of the thing under test. Every test in this file uses one streamer.
TEST_F(StreamerTest, Object_Storage_Then_Filesystem_Submission)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");

    const auto fs_data = utils::random::buffer(utils::random::number(100, 1000));
    utils::temp::File fs_file(fs_data);

    void * streamer;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    // object storage first - this is what creates the object-storage pool (via the plugin lock)
    EXPECT_EQ(submit(streamer, num_files, file_names.data(), file_offsets.data(), sizes.data(),
                     dsts.data(), num_ranges.data(), internal_sizes.data()),
              static_cast<int>(common::ResponseCode::Success));

    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        unsigned file_index = 0;
        unsigned range_index = 0;
        const auto response_code = next_response(streamer, &file_index, &range_index);
        EXPECT_EQ(response_code, static_cast<int>(common::ResponseCode::Success));
        if (response_code != static_cast<int>(common::ResponseCode::Success))
        {
            break;
        }
    }

    // then a filesystem submission on the same streamer, which creates the filesystem pool
    {
        std::vector<unsigned char> dst(fs_data.size());
        void * dst_ptr = dst.data();
        const char * path = fs_file.path.c_str();
        unsigned n_ranges = 1;
        size_t offset = 0;
        size_t size = fs_data.size();

        SubmissionId submission_id = 0;
        EXPECT_EQ(runai_file_streamer_request(streamer, &submission_id, 1, &path, &n_ranges, &offset, &size, &dst_ptr, RunaiFileStreamerDevice{}),
                  static_cast<int>(common::ResponseCode::Success));

        unsigned file_index = 0;
        unsigned range_index = 0;
        EXPECT_EQ(next_response(streamer, &file_index, &range_index),
                  static_cast<int>(common::ResponseCode::Success));
        EXPECT_EQ(dst, std::vector<unsigned char>(fs_data.begin(), fs_data.end()));
    }

    runai_file_streamer_end(streamer);
    EXPECT_EQ(verify_mock(), 0);
}

TEST_F(StreamerTest, Multiple_Files_Error)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto set_backend_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_backend_shutdown_policy(utils::random::boolean() ? common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN : common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT);

    const auto error_code = common::ResponseCode::FileAccessError;
    // Scoped to the reads only, released before runai_file_streamer_end below: the injected code reaches every mock entry
    // point, so leaving it set across teardown injects failures into the shutdown path too.
    auto env_rc = std::make_unique<utils::temp::Env>("RUNAI_STREAMER_S3_MOCK_RESPONSE_CODE", static_cast<int>(error_code));

    void * streamer;
    EXPECT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));


    auto res = submit(streamer,
                             num_files,
                             file_names.data(),
                             file_offsets.data(),
                             sizes.data(),
                             dsts.data(),
                             num_ranges.data(),
                             internal_sizes.data());

    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    // wait for all the responses to arrive
    unsigned r;
    unsigned file_index;
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        r = utils::random::number();
        file_index = utils::random::number();
        const auto response_code = next_response(streamer, &file_index, &r);
        EXPECT_EQ(response_code, static_cast<int>(error_code));
        EXPECT_LT(file_index, num_files);
        EXPECT_EQ(expected_response[file_index].count(r), 1);
        if (response_code != static_cast<int>(error_code))
        {
            break;
        }
        expected_response[file_index].erase(r);
    }

    env_rc.reset();   // stop injecting failures before the streamer tears the backend down
    runai_file_streamer_end(streamer);
    EXPECT_EQ(verify_mock(), 0);
}

namespace
{

// Collects the (path, size) pairs delivered to the runai_file_streamer_list_files callback
struct ListFilesResult
{
    std::vector<std::pair<std::string, size_t>> files;
};

void list_files_collect(const char* path, size_t size, void* user_data)
{
    static_cast<ListFilesResult*>(user_data)->files.emplace_back(path, size);
}

} // namespace

TEST_F(StreamerTest, ListFiles_S3_ReturnsEntriesAndCleansUp)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto is_shutdown = dylib.dlsym<bool(*)()>("runai_mock_s3_is_shutdown");
    auto set_files = dylib.dlsym<void(*)(const char**, const size_t*, unsigned)>("runai_mock_s3_set_files");
    auto set_backend_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_backend_shutdown_policy(common::backend_api::ObjectShutdownPolicy_t::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN);

    std::vector<const char*> paths = {"s3://bucket/models/a.safetensors", "s3://bucket/models/b.bin"};
    const std::vector<size_t> sizes = {111, 222};
    set_files(paths.data(), sizes.data(), paths.size());

    ListFilesResult result;
    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));
    auto res = runai_file_streamer_list_files(streamer, "s3://bucket/models/", 1, nullptr, 0, nullptr, 0, list_files_collect, &result);

    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));
    ASSERT_EQ(result.files.size(), 2u);

    std::map<std::string, size_t> by_path;
    for (const auto& f : result.files)
    {
        by_path[f.first] = f.second;
    }
    EXPECT_EQ(by_path["s3://bucket/models/a.safetensors"], 111u);
    EXPECT_EQ(by_path["s3://bucket/models/b.bin"], 222u);

    // S3Cleanup ran on Streamer destruction: clients released and backend closed
    runai_file_streamer_end(streamer);
    EXPECT_EQ(verify_mock(), 0);
    EXPECT_TRUE(is_shutdown());
}

TEST_F(StreamerTest, ListFiles_S3_AppliesPatternFilters)
{
    utils::Dylib dylib("libstreamers3.so");
    auto set_files = dylib.dlsym<void(*)(const char**, const size_t*, unsigned)>("runai_mock_s3_set_files");

    std::vector<const char*> paths = {"s3://bucket/m/model.safetensors", "s3://bucket/m/config.json"};
    const std::vector<size_t> sizes = {10, 20};
    set_files(paths.data(), sizes.data(), paths.size());

    std::vector<const char*> allow = {"*.safetensors"};

    ListFilesResult result;
    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));
    auto res = runai_file_streamer_list_files(streamer, "s3://bucket/m/", 1, allow.data(), allow.size(), nullptr, 0, list_files_collect, &result);
    runai_file_streamer_end(streamer);

    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));
    ASSERT_EQ(result.files.size(), 1u);
    EXPECT_EQ(result.files[0].first, "s3://bucket/m/model.safetensors");
}

TEST_F(StreamerTest, ListFiles_S3_ForwardsIsRecursive)
{
    utils::Dylib dylib("libstreamers3.so");
    auto set_files = dylib.dlsym<void(*)(const char**, const size_t*, unsigned)>("runai_mock_s3_set_files");
    auto last_is_recursive = dylib.dlsym<int(*)()>("runai_mock_s3_last_list_files_is_recursive");

    std::vector<const char*> paths = {"s3://bucket/x/f.bin"};
    const std::vector<size_t> sizes = {1};
    set_files(paths.data(), sizes.data(), paths.size());

    ListFilesResult result;
    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    runai_file_streamer_list_files(streamer, "s3://bucket/x/", 0, nullptr, 0, nullptr, 0, list_files_collect, &result);
    EXPECT_EQ(last_is_recursive(), 0);

    runai_file_streamer_list_files(streamer, "s3://bucket/x/", 1, nullptr, 0, nullptr, 0, list_files_collect, &result);
    EXPECT_EQ(last_is_recursive(), 1);

    runai_file_streamer_end(streamer);
}

TEST_F(StreamerTest, ListFiles_S3_ErrorPropagates)
{
    utils::Dylib dylib("libstreamers3.so");
    auto set_files = dylib.dlsym<void(*)(const char**, const size_t*, unsigned)>("runai_mock_s3_set_files");
    auto set_response = dylib.dlsym<void(*)(common::backend_api::ResponseCode_t)>("runai_mock_s3_set_list_files_response");

    std::vector<const char*> paths = {"s3://bucket/x/f.bin"};
    const std::vector<size_t> sizes = {1};
    set_files(paths.data(), sizes.data(), paths.size());
    set_response(common::ResponseCode::FileAccessError);

    ListFilesResult result;
    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));
    auto res = runai_file_streamer_list_files(streamer, "s3://bucket/x/", 1, nullptr, 0, nullptr, 0, list_files_collect, &result);
    runai_file_streamer_end(streamer);

    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::FileAccessError));
    EXPECT_TRUE(result.files.empty());
}

}; // namespace runai::llm::streamer
