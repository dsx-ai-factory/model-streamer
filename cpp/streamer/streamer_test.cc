#include "streamer/streamer.h"

#include <unistd.h>

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <set>

#include "common/response_code/response_code.h"
#include "common/submission/submission_id.h"

#include "utils/logging/logging.h"
#include "utils/random/random.h"
#include "utils/thread/thread.h"
#include "utils/fd/fd.h"
#include "utils/temp/env/env.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer
{

namespace
{

// Test adapters over the multi-request C API. These single-submission tests don't need the submission id or
// streams and drain by a known count, so submit() discards the id and next_response() blocks (timeout 0) for
// the next sub-range. There is no finish-on-drain: a submission's last response carries submission_done, so
// the tests never call next_response() past the expected count (that would block).
// submit() keeps the classic per-file argument shape so the tests below are unchanged, and adapts it to
// the range API: each file's sub ranges tile [file_offsets[i], file_offsets[i] + bytesizes[i]) in order,
// and every file is written consecutively into the single buffer at dsts[0] - the layout the previous
// API implied.
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

// Single-file variant that reports the submission id, for the concurrent-submission tests. The sub ranges
// tile [offset, offset + sum(sizes)) and are written consecutively from dst.
inline int submit_one(void * streamer, SubmissionId * id, const char * path, size_t offset,
                      void * dst, std::vector<size_t> & sizes)
{
    unsigned num_ranges = sizes.size();
    std::vector<size_t> range_offsets;
    std::vector<void *> range_dsts;

    char * d = static_cast<char *>(dst);
    size_t o = offset;
    for (const auto size : sizes)
    {
        range_offsets.push_back(o);
        range_dsts.push_back(d);
        o += size;
        d += size;
    }

    return runai_file_streamer_request(streamer, id, 1, &path, &num_ranges,
                         range_offsets.data(), sizes.data(), range_dsts.data(), RunaiFileStreamerDevice{});
}

inline int next_response(void * streamer, unsigned * file_index, unsigned * index)
{
    SubmissionId submission_id = 0;
    int submission_done = 0;
    return runai_file_streamer_response(streamer, &submission_id, file_index, index, &submission_done, 0);
}

struct StreamerTest : ::testing::Test
{
    StreamerTest() :
        _size("RUNAI_STREAMER_CONCURRENCY", utils::random::number<int>(1, 16)),
        _chunk_bytesize("RUNAI_STREAMER_CHUNK_BYTESIZE", utils::random::number<int>(1, 1024))
    {}

    int runai_request_file(void * streamer, const char * path, size_t offset, size_t size, void * dst)
    {
        std::vector<size_t> sizes;
        sizes.push_back(size);
        std::vector<size_t *> internal_sizes;
        internal_sizes.push_back(sizes.data());
        std::vector<unsigned> num_sizes;
        num_sizes.push_back(1);
        return submit(streamer, 1, &path, &offset, &size, &dst, num_sizes.data(), internal_sizes.data());
    }

    int runai_read_file(void * streamer, const char * path, size_t offset, size_t size, void * dst)
    {
        std::vector<size_t> sizes;
        sizes.push_back(size);
        std::vector<size_t *> internal_sizes;
        internal_sizes.push_back(sizes.data());
        std::vector<unsigned> num_sizes;
        num_sizes.push_back(1);
        auto res = submit(streamer, 1, &path, &offset, &size, &dst, num_sizes.data(), internal_sizes.data());
        if (res != static_cast<int>(runai::llm::streamer::common::ResponseCode::Success))
        {
            return res;
        }

        unsigned file_index;
        unsigned index;
        return next_response(streamer, &file_index, &index);
    }

 protected:
    utils::temp::Env _size;
    utils::temp::Env _chunk_bytesize;
    utils::temp::Env _block_bytesize;
};

} // namespace

TEST_F(StreamerTest, Creation)
{
    void * streamer = nullptr;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));
    EXPECT_NE(streamer, nullptr);

    EXPECT_NO_THROW(runai_file_streamer_end(streamer));
}

// Replaces AssignerTest.Mismatched_Input_Sizes, which asserted the old per-file vector length check.
// Lengths cannot disagree now - num_ranges describes the flat arrays - so what remains checkable at the C
// boundary is null-ness, validated in submit_request before anything is dereferenced.
TEST(Request, Null_Parameters)
{
    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    SubmissionId id = 0;
    // a real file, so the one submission that IS accepted below reads successfully and the test is about
    // argument validation only
    const auto data = utils::random::buffer(utils::random::number(10, 100));
    utils::temp::File file(data);
    const char * path = file.path.c_str();
    unsigned num_ranges = 1;
    size_t offset = 0;
    size_t size = 10;
    std::vector<char> buffer(size);
    void * dst = buffer.data();

    const auto invalid = static_cast<int>(common::ResponseCode::InvalidParameterError);

    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, nullptr, &num_ranges, &offset, &size, &dst, RunaiFileStreamerDevice{}), invalid);
    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, &path, nullptr, &offset, &size, &dst, RunaiFileStreamerDevice{}), invalid);
    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, &path, &num_ranges, nullptr, &size, &dst, RunaiFileStreamerDevice{}), invalid);
    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, &path, &num_ranges, &offset, nullptr, &dst, RunaiFileStreamerDevice{}), invalid);
    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, &path, &num_ranges, &offset, &size, nullptr, RunaiFileStreamerDevice{}), invalid);

    // a null path entry is caught before it is used to construct a std::string - the previous code built
    // the vector straight from the array, which was undefined behaviour on a null element
    const char * null_path = nullptr;
    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, &null_path, &num_ranges, &offset, &size, &dst, RunaiFileStreamerDevice{}), invalid);

    // A null destination ELEMENT (the array itself is fine) is caught deeper, by verify_requests, which
    // throws rather than returning. The specific code has to survive the C boundary: UnknownError is what
    // tells a caller to abort everything, while an argument error is attributable and recoverable, so
    // collapsing this to UnknownError would turn a bad argument into a dead stream.
    void * null_dst = nullptr;
    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, &path, &num_ranges, &offset, &size, &null_dst, RunaiFileStreamerDevice{}), invalid);

    // a zero-sized range writes nothing, so a null destination there is accepted
    size_t zero = 0;
    EXPECT_EQ(runai_file_streamer_request(streamer, &id, 1, &path, &num_ranges, &offset, &zero, &null_dst, RunaiFileStreamerDevice{}),
              static_cast<int>(common::ResponseCode::Success));

    // The only submission accepted above, so it owes one response and must be drained before the test
    // returns - an in-flight submission writes into buffers the caller has released.
    unsigned file_index = 0;
    unsigned range_index = 0;
    int submission_done = 0;
    SubmissionId response_id = 0;
    EXPECT_EQ(runai_file_streamer_response(streamer, &response_id, &file_index, &range_index, &submission_done, 0),
              static_cast<int>(common::ResponseCode::Success));
    EXPECT_EQ(response_id, id);
    EXPECT_EQ(file_index, 0u);
    EXPECT_EQ(range_index, 0u);
    EXPECT_EQ(submission_done, 1);

    EXPECT_NO_THROW(runai_file_streamer_end(streamer));
}

TEST(Creation, Invalid_Parameter)
{
    void * streamer = nullptr;

    {
        utils::temp::Env size("RUNAI_STREAMER_CONCURRENCY", 0);
        utils::temp::Env chunk_bytesize("RUNAI_STREAMER_CHUNK_BYTESIZE", utils::random::number<int>(1, 2000));
        auto res = runai_file_streamer_start(&streamer);
        EXPECT_EQ(res, static_cast<int>(common::ResponseCode::InvalidParameterError));
        EXPECT_EQ(streamer, nullptr);
    }

    {
        utils::temp::Env size("RUNAI_STREAMER_CONCURRENCY", utils::random::number<int>(1, 10));
        utils::temp::Env chunk_bytesize("RUNAI_STREAMER_CHUNK_BYTESIZE", 0);
        auto res = runai_file_streamer_start(&streamer);
        EXPECT_EQ(res, static_cast<int>(common::ResponseCode::InvalidParameterError));
        EXPECT_EQ(streamer, nullptr);
    }

    {
        utils::temp::Env size("RUNAI_STREAMER_CONCURRENCY", utils::random::number<int>(1, 10));
        utils::temp::Env chunk_bytesize("RUNAI_STREAMER_CHUNK_BYTESIZE", 0);
        auto res = runai_file_streamer_start(&streamer);
        EXPECT_EQ(res, static_cast<int>(common::ResponseCode::InvalidParameterError));
        EXPECT_EQ(streamer, nullptr);
    }
}

TEST_F(StreamerTest, Read)
{
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    const auto expected = utils::Fd::read(file.path);
    EXPECT_EQ(expected.size(), size);

    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    std::vector<unsigned char> v(size);
    res = runai_read_file(streamer, file.path.c_str(), 0, size, v.data());
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(v[i], expected[i]);
        if (v[i] != expected[i])
        {
            break;
        }
    }

    runai_file_streamer_end(streamer);
}

TEST_F(StreamerTest, Async)
{
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    const auto expected = utils::Fd::read(file.path);
    EXPECT_EQ(expected.size(), size);

    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    std::vector<unsigned char> dst(size);
    std::vector<size_t> sizes;
    sizes.push_back(size);
    EXPECT_EQ(runai_request_file(streamer, file.path.c_str(), 0, size, dst.data()), static_cast<int>(common::ResponseCode::Success));
    unsigned r = utils::random::number();
    unsigned rfile = utils::random::number();
    EXPECT_EQ(next_response(streamer, &rfile, &r), static_cast<int>(common::ResponseCode::Success));
    EXPECT_EQ(r, 0);
    EXPECT_EQ(rfile, 0);

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(dst[i], expected[i]);
        if (dst[i] != expected[i])
        {
            break;
        }
    }

    runai_file_streamer_end(streamer);
}

TEST_F(StreamerTest, Error)
{
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(utils::random::number(1, size-1));
    utils::temp::File file(data);

    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    std::vector<char> dst(size);
    std::vector<size_t> sizes;
    sizes.push_back(size);
    const auto request_ret = runai_request_file(streamer, file.path.c_str(), 0, size, dst.data());
    if (request_ret == static_cast<int>(common::ResponseCode::EofError))
    {
        return;
    }

    EXPECT_EQ(request_ret, static_cast<int>(common::ResponseCode::Success));

    unsigned value = utils::random::number();
    unsigned r = value;
    unsigned file_index = utils::random::number();
    EXPECT_EQ(next_response(streamer, &file_index, &r), static_cast<int>(common::ResponseCode::EofError));
    EXPECT_EQ(r, 0);
    EXPECT_EQ(file_index, 0);

    runai_file_streamer_end(streamer);
}

TEST(Response, Description)
{
    auto __strings = std::map<int, std::string>
    {
        { static_cast<int>(common::ResponseCode::FileAccessError),                "File access error"                       },
        { static_cast<int>(common::ResponseCode::EofError),                       "End of file reached"                     },
        { static_cast<int>(common::ResponseCode::InvalidParameterError),          "Invalid request parameters"              },
        { static_cast<int>(common::ResponseCode::EmptyRequestError),              "Empty request parameters"                },
        { static_cast<int>(common::ResponseCode::BusyError),                      "Streamer is handling previous request"   },
        { static_cast<int>(common::ResponseCode::UnknownError),                   "Unknown Error"                           },
        { static_cast<int>(common::ResponseCode::FinishedError),                  "Finished all responses"                  },
        { static_cast<int>(common::ResponseCode::Success),                        "Request sent successfuly"                },
    };

    // errors

    for (auto response_code : {common::ResponseCode::FileAccessError, common::ResponseCode::EofError, common::ResponseCode::InvalidParameterError, common::ResponseCode::EmptyRequestError, common::ResponseCode::BusyError, common::ResponseCode::UnknownError, common::ResponseCode::FinishedError} )
    {
        std::string str = runai_file_streamer_response_str(static_cast<int>(response_code));

        const auto it = __strings.find(static_cast<int>(response_code));
        EXPECT_NE(it, __strings.end());

        EXPECT_EQ(str, it->second);
    }
}

TEST_F(StreamerTest, S3_Library_Not_Found)
{
    auto size = utils::random::number(100, 1000);

    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    const auto s3_path = "s3://" + utils::random::string() + "/" + utils::random::string();

    std::vector<char> dst(size);
    std::vector<size_t> sizes;
    sizes.push_back(size);
    EXPECT_EQ(runai_request_file(streamer, s3_path.c_str(), 0, size, dst.data()), static_cast<int>(common::ResponseCode::Success));
    unsigned r = utils::random::number();
    EXPECT_EQ(next_response(streamer, &r, &r), static_cast<int>(common::ResponseCode::S3NotSupported));

    runai_file_streamer_end(streamer);
}

TEST_F(StreamerTest, GCS_Library_Not_Found)
{
    auto size = utils::random::number(100, 1000);

    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    const auto s3_path = "gs://" + utils::random::string() + "/" + utils::random::string();

    std::vector<char> dst(size);
    std::vector<size_t> sizes;
    sizes.push_back(size);
    EXPECT_EQ(runai_request_file(streamer, s3_path.c_str(), 0, size, dst.data()), static_cast<int>(common::ResponseCode::Success));
    unsigned r = utils::random::number();
    EXPECT_EQ(next_response(streamer, &r, &r), static_cast<int>(common::ResponseCode::GCSNotSupported));

    runai_file_streamer_end(streamer);
}

TEST_F(StreamerTest, End_Before_Read)
{
    auto size = utils::random::number(10000000, 100000000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    std::vector<unsigned char> dst(size);
    std::vector<size_t> sizes;
    sizes.push_back(size);

    EXPECT_EQ(runai_request_file(streamer, file.path.c_str(), 0, size, dst.data()), static_cast<int>(common::ResponseCode::Success));

    ::usleep(utils::random::number(400));

    const auto start_time = std::chrono::steady_clock::now();
    runai_file_streamer_end(streamer);
    const auto time_ = std::chrono::steady_clock::now();
    const auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(time_ - start_time);
    EXPECT_LT(duration.count(), 1000);
}

TEST_F(StreamerTest, Multiple_Files)
{
    auto num_files = utils::random::number(1, 50);
    std::vector<utils::temp::File> files(num_files);
    std::vector<const char *> file_paths(num_files);
    std::vector<size_t> file_offsets(num_files);

    std::vector<std::vector<uint8_t>> buffers(num_files);
    std::vector<std::vector<uint8_t>> expected(num_files);
    std::vector<size_t> sizes(num_files);

    // We assume that all files are written to a single continous cpu buffer
    std::vector<void *> dsts(num_files);
    std::vector<unsigned> num_ranges(num_files);
    std::vector<std::vector<size_t>> range_sizes(num_files);
    std::vector<size_t *> internal_sizes(num_files);

    std::vector<std::set<unsigned>> expected_response(num_files);

    size_t num_expected_responses = 0;
    size_t dst_size = 0;
    for (unsigned i = 0; i < num_files; ++i)
    {
        sizes[i] = utils::random::number(1000000, 10000000);
        dst_size += sizes[i];

        buffers[i] = utils::random::buffer(sizes[i]);
        files[i] = utils::temp::File(buffers[i]);
        file_paths[i] = files[i].path.c_str();
        file_offsets[i] = 0;
        expected[i] = utils::Fd::read(files[i].path);
        EXPECT_EQ(expected[i].size(), sizes[i]);

        auto num_non_zero_chunks = utils::random::number(1, 100);
        auto num_zero_chunks = utils::random::number(0, 2);
        num_ranges[i] = num_non_zero_chunks + num_zero_chunks;
        auto chunks = utils::random::chunks(sizes[i], num_non_zero_chunks);

        // add zero size chunks
        for (unsigned k = 0; k < num_non_zero_chunks || num_zero_chunks > 0;)
        {
            bool add_zero = utils::random::boolean();
            if (num_zero_chunks > 0 && add_zero)
            {
                range_sizes[i].push_back(0);
                --num_zero_chunks;
            }
            else if (k < num_non_zero_chunks)
            {
                range_sizes[i].push_back(chunks[k]);
                ++k;
            }
        }

        ASSERT_EQ(range_sizes[i].size(), num_ranges[i]);

        internal_sizes[i] = range_sizes[i].data();

        num_expected_responses += num_ranges[i];

        for (unsigned request_index = 0; request_index < num_ranges[i]; ++request_index)
        {
            expected_response[i].insert(request_index);
        }
    }
    std::vector<unsigned char> dst(dst_size);
    dsts[0] = static_cast<void *>(dst.data());

    void * streamer;
    auto res = runai_file_streamer_start(&streamer);
    EXPECT_EQ(res, static_cast<int>(common::ResponseCode::Success));

    EXPECT_EQ(submit(streamer, num_files, file_paths.data(), file_offsets.data(), sizes.data(), dsts.data(), num_ranges.data(), internal_sizes.data()), static_cast<int>(common::ResponseCode::Success));

    // wait for all the responses to arrive
    unsigned r;
    unsigned file_index;
    for (unsigned i = 0; i < num_expected_responses; ++i)
    {
        r = utils::random::number();
        file_index = utils::random::number();
        EXPECT_EQ(next_response(streamer, &file_index, &r), static_cast<int>(common::ResponseCode::Success));
        EXPECT_LT(file_index, num_files);
        EXPECT_EQ(expected_response[file_index].count(r), 1);
        expected_response[file_index].erase(r);
    }

    // verify
    size_t offset = 0;
    for (unsigned file_index = 0; file_index < num_files; ++file_index)
    {
        EXPECT_EQ(expected_response[file_index].size(), 0);
        EXPECT_EQ(expected[file_index].size(), sizes[file_index]);
        for (size_t j = 0; j < sizes[file_index]; ++j)
        {
            EXPECT_LT(offset + j, dst.size());
            EXPECT_EQ(dst[offset + j], expected[file_index][j]);
            if (dst[offset + j] != expected[file_index][j])
            {
                LOG(ERROR) << "offset = " << offset + j << " expected = " << expected[file_index][j] << " dst = " << dst[offset + j];
                break;
            }
        }
        offset += sizes[file_index];
    }

    runai_file_streamer_end(streamer);
}

TEST(AsyncEx, ConcurrentSubmissionsDemux)
{
    // Two concurrent submissions on the persistent responder, demuxed by submission_id.
    // Submission A has 2 sub-ranges, B has 1; submission_done fires exactly once per submission
    // (on its last sub-range). After draining, a timed response times out (persistent, not
    // FinishedError).
    const auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);
    const auto expected = utils::Fd::read(file.path);
    ASSERT_EQ(expected.size(), size);

    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    const char * path = file.path.c_str();
    size_t offset = 0;

    // Submission A: 2 sub-ranges into dstA
    std::vector<unsigned char> dstA(size);
    const size_t s1 = size / 2;
    std::vector<size_t> subA = { s1, size - s1 };
    SubmissionId idA = 0;
    void * dstA_ptr = dstA.data();
    ASSERT_EQ(submit_one(streamer, &idA, path, offset, dstA_ptr, subA),
              static_cast<int>(common::ResponseCode::Success));

    // Submission B: 1 sub-range into dstB
    std::vector<unsigned char> dstB(size);
    std::vector<size_t> subB = { size };
    SubmissionId idB = 0;
    void * dstB_ptr = dstB.data();
    ASSERT_EQ(submit_one(streamer, &idB, path, offset, dstB_ptr, subB),
              static_cast<int>(common::ResponseCode::Success));

    EXPECT_NE(idA, 0u);
    EXPECT_NE(idB, 0u);
    EXPECT_NE(idA, idB);

    // Drain 3 responses (2 for A, 1 for B), in any order, demuxed by submission_id
    std::map<SubmissionId, unsigned> got;         // submission_id -> responses seen
    std::map<SubmissionId, unsigned> done_count;   // submission_id -> submission_done seen
    for (int i = 0; i < 3; ++i)
    {
        SubmissionId sid = 0;
        unsigned fi = 0, idx = 0;
        int done = 0;
        int ret = runai_file_streamer_response(streamer, &sid, &fi, &idx, &done, 5000);
        ASSERT_EQ(ret, static_cast<int>(common::ResponseCode::Success));
        got[sid]++;
        if (done) done_count[sid]++;
    }

    EXPECT_EQ(got[idA], 2u);
    EXPECT_EQ(got[idB], 1u);
    EXPECT_EQ(done_count[idA], 1u);   // submission_done exactly once, on the last sub-range
    EXPECT_EQ(done_count[idB], 1u);

    // Persistent: after draining, a timed response times out (does NOT return FinishedError)
    SubmissionId sid = 0;
    unsigned fi = 0, idx = 0;
    int done = 0;
    EXPECT_EQ(runai_file_streamer_response(streamer, &sid, &fi, &idx, &done, 50),
              static_cast<int>(common::ResponseCode::TimedOut));

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(dstA[i], expected[i]);
        EXPECT_EQ(dstB[i], expected[i]);
    }

    runai_file_streamer_end(streamer);
}

TEST(AsyncEx, PerSubmissionErrorIsolation)
{
    // Two concurrent submissions: one valid, one reading past EOF. The failure is tagged to its
    // own submission_id and does not affect the other, which completes successfully.
    const auto data_size = utils::random::number(100, 500);
    const auto data = utils::random::buffer(data_size);
    utils::temp::File file(data);
    const auto expected = utils::Fd::read(file.path);
    ASSERT_EQ(expected.size(), data_size);

    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    const char * path = file.path.c_str();
    size_t offset = 0;

    // Submission A: valid full read
    std::vector<unsigned char> dstA(data_size);
    std::vector<size_t> subA = { data_size };
    SubmissionId idA = 0;
    void * dstA_ptr = dstA.data();
    ASSERT_EQ(submit_one(streamer, &idA, path, offset, dstA_ptr, subA),
              static_cast<int>(common::ResponseCode::Success));

    // Submission B: read past EOF -> EofError
    const size_t over = data_size + utils::random::number(1, 100);
    std::vector<unsigned char> dstB(over);
    std::vector<size_t> subB = { over };
    SubmissionId idB = 0;
    void * dstB_ptr = dstB.data();
    ASSERT_EQ(submit_one(streamer, &idB, path, offset, dstB_ptr, subB),
              static_cast<int>(common::ResponseCode::Success));

    EXPECT_NE(idA, idB);

    // drain both responses; demux by submission_id
    std::map<SubmissionId, int> ret_by_sub;
    std::map<SubmissionId, int> done_by_sub;
    for (int i = 0; i < 2; ++i)
    {
        SubmissionId sid = 0;
        unsigned fi = 0, idx = 0;
        int done = 0;
        int ret = runai_file_streamer_response(streamer, &sid, &fi, &idx, &done, 5000);
        ret_by_sub[sid] = ret;
        done_by_sub[sid] += done;
    }

    EXPECT_EQ(ret_by_sub[idA], static_cast<int>(common::ResponseCode::Success));
    EXPECT_EQ(ret_by_sub[idB], static_cast<int>(common::ResponseCode::EofError));
    EXPECT_EQ(done_by_sub[idA], 1);   // each submission is completed exactly once
    EXPECT_EQ(done_by_sub[idB], 1);

    // A's data is intact despite B failing
    for (size_t i = 0; i < data_size; ++i)
    {
        EXPECT_EQ(dstA[i], expected[i]);
    }

    runai_file_streamer_end(streamer);
}

TEST(AsyncEx, MultipleSubmitterThreads)
{
    // Concurrent submitters exercise the submission-id allocator + registry mutex; a single
    // consumer drains all, verifying distinct ids, per-submission completion, and data.
    const auto size = utils::random::number(100, 500);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);
    const auto expected = utils::Fd::read(file.path);
    ASSERT_EQ(expected.size(), size);

    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    const unsigned N = utils::random::number(4, 12);
    std::vector<std::vector<unsigned char>> dsts(N, std::vector<unsigned char>(size));
    std::vector<SubmissionId> ids(N, 0);
    std::vector<int> submit_ret(N, -1);

    // N concurrent submitters (joined on scope exit)
    {
        std::vector<utils::Thread> submitters;
        for (unsigned t = 0; t < N; ++t)
        {
            submitters.emplace_back([&, t]()
            {
                const char * path = file.path.c_str();
                size_t offset = 0;
                std::vector<size_t> sub = { size };
                void * dst = dsts[t].data();
                submit_ret[t] = submit_one(streamer, &ids[t], path, offset, dst, sub);
            });
        }
    }

    // all accepted, ids distinct and non-zero
    std::set<SubmissionId> unique_ids;
    for (unsigned t = 0; t < N; ++t)
    {
        EXPECT_EQ(submit_ret[t], static_cast<int>(common::ResponseCode::Success));
        EXPECT_NE(ids[t], 0u);
        unique_ids.insert(ids[t]);
    }
    EXPECT_EQ(unique_ids.size(), N);

    // single consumer drains all N responses (one sub-range each)
    std::map<SubmissionId, int> done_by_sub;
    for (unsigned i = 0; i < N; ++i)
    {
        SubmissionId sid = 0;
        unsigned fi = 0, idx = 0;
        int done = 0;
        int ret = runai_file_streamer_response(streamer, &sid, &fi, &idx, &done, 5000);
        EXPECT_EQ(ret, static_cast<int>(common::ResponseCode::Success));
        done_by_sub[sid] += done;
    }
    for (auto id : unique_ids)
    {
        EXPECT_EQ(done_by_sub[id], 1);
    }

    // all buffers received the data
    for (unsigned t = 0; t < N; ++t)
    {
        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_EQ(dsts[t][i], expected[i]);
        }
    }

    runai_file_streamer_end(streamer);
}

// Every exit of the probe must leave a USABLE block behind, including the ones that report an error.
//
// The caller lays its ring out from this number and divides by it, so the Python binding treats a zero
// as a broken promise and refuses to stream. That makes an unwritten out-parameter on an error path a
// failed model load, not a degraded one - and the API exists to do the opposite: report a code AND a
// block, so an inconclusive probe costs a page-aligned layout now and is asked again next submission.
//
// None of these paths can be reached from Python, which always passes a real streamer and real paths.
// That is exactly why they are tested here: nothing else runs them.
TEST(ProbeDirectBlockSize, Every_Exit_Leaves_A_Usable_Block)
{
    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    const auto invalid = static_cast<int>(common::ResponseCode::InvalidParameterError);

    const auto data = utils::random::buffer(utils::random::number(10, 100));
    utils::temp::File file(data);
    const char * path = file.path.c_str();

    // The success case first, so the failures below are compared against something real.
    size_t block = 0;
    EXPECT_EQ(runai_file_streamer_probe_direct_block_size(streamer, &path, 1, &block),
              static_cast<int>(common::ResponseCode::Success));
    EXPECT_GT(block, 0u);

    // A null streamer. Rejected, but the block must still be usable.
    block = 0;
    EXPECT_EQ(runai_file_streamer_probe_direct_block_size(nullptr, &path, 1, &block), invalid);
    EXPECT_GT(block, 0u) << "an error exit left the out-parameter untouched, so the caller reads 0";

    // A null path array with a non-zero count. Same requirement.
    block = 0;
    EXPECT_EQ(runai_file_streamer_probe_direct_block_size(streamer, nullptr, 1, &block), invalid);
    EXPECT_GT(block, 0u) << "an error exit left the out-parameter untouched, so the caller reads 0";

    // No paths at all. Nothing can be measured, so this reports UnknownError - and still answers with
    // a layout value, which is the whole contract.
    block = 0;
    EXPECT_EQ(runai_file_streamer_probe_direct_block_size(streamer, nullptr, 0, &block),
              static_cast<int>(common::ResponseCode::UnknownError));
    EXPECT_GT(block, 0u);

    // A null out-parameter is the one case with nowhere to write, so it only reports.
    EXPECT_EQ(runai_file_streamer_probe_direct_block_size(streamer, &path, 1, nullptr), invalid);

    EXPECT_NO_THROW(runai_file_streamer_end(streamer));
}

}; // namespace runai::llm::streamer
