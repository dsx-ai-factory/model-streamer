#include "streamer/impl/async_io/async_io_worker/async_io_worker.h"
#include "common/device/device.h"

#include <gtest/gtest.h>

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <atomic>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "posix_io/alignment/alignment.h"
#include "posix_io/mock/mock_io_engine.h"
#include "streamer/impl/assigner/assigner.h"
#include "streamer/impl/batches/batches.h"
#include "common/exception/exception.h"

#include "utils/random/random.h"
#include "utils/temp/env/env.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::impl
{

namespace
{

using posix_io::MockIoEngine;
using posix_io::Strategy;
using posix_io::WaitMode;

constexpr size_t ChunkSize = 4096;

// One file's ranges, tiled contiguously into one buffer - so they coalesce to a single transfer, which
// is the layout a model read produces.
struct Fixture
{
    // `start` is the file offset the first range reads from. Non-zero puts the region out of step
    // with the block boundary, which is what makes a direct read need its edges bounced.
    explicit Fixture(const std::vector<size_t> & range_sizes, unsigned concurrency = 1, size_t start = 0,
                     unsigned long queue_depth = 1024UL) :
        concurrency(std::string("RUNAI_STREAMER_CONCURRENCY"), static_cast<unsigned long>(concurrency)),
        chunk_bytesize(std::string("RUNAI_STREAMER_FS_CHUNK_BYTESIZE"), static_cast<unsigned long>(ChunkSize)),
        depth(std::string("RUNAI_STREAMER_FS_QUEUE_DEPTH"), queue_depth),
        group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 1UL),
        sizes(range_sizes),
        total(std::accumulate(range_sizes.begin(), range_sizes.end(), static_cast<size_t>(0))),
        config(std::make_shared<Config>(false /* do not force minimum */)),
        responder(std::make_shared<common::Responder>(0)),
        data(utils::random::buffer(start + (total == 0 ? 1 : total))),
        file(data),
        buffer(total == 0 ? 1 : total)
    {
        responder->increment(sizes.size());

        FileRanges ranges;
        ranges.path = file.path;
        size_t offset = start;
        char * dst = buffer.data();
        for (const auto size : sizes)
        {
            ranges.ranges.push_back(ReadRange{ offset, size, dst });
            offset += size;
            dst += size;
        }
        request.push_back(ranges);
    }

    // One workload holding every batch, which is what a single async worker gets.
    //
    // `path_override` gives the batches a path that is not the temp file. The worker opens
    // batch.path, so this is how a test makes the open fail while the ranges stay valid.
    Workload workload(const std::string & path_override = "")
    {
        const std::string & batch_path = path_override.empty() ? file.path : path_override;

        Assigner assigner(request, config);
        Workload out;
        for (const auto & transfer : assigner.transfers())
        {
            Batches batches(1 /* submission */, transfer.file_index, transfer.tasks, config, responder,
                            batch_path, params, transfer.range_sizes, transfer.first_range_index, common::Device::host());
            for (size_t i = 0; i < batches.size(); ++i)
            {
                out.add_batch(std::move(batches[i]));
            }
        }
        return out;
    }

    // What a fully-read range should contain: the mock fills from the absolute file offset.
    std::vector<char> expected_at(size_t offset, size_t bytesize) const
    {
        std::vector<char> out(bytesize);
        MockIoEngine::fill_expected(out.data(), offset, bytesize);
        return out;
    }

    std::vector<char> got_at(size_t offset, size_t bytesize) const
    {
        return std::vector<char>(buffer.begin() + offset, buffer.begin() + offset + bytesize);
    }

    utils::temp::Env concurrency;
    utils::temp::Env chunk_bytesize;
    utils::temp::Env depth;
    utils::temp::Env group;

    std::vector<size_t> sizes;
    size_t total;
    std::shared_ptr<Config> config;
    std::shared_ptr<common::Responder> responder;
    std::vector<uint8_t> data;
    utils::temp::File file;
    std::vector<char> buffer;
    std::vector<FileRanges> request;
    common::s3::S3ClientWrapper::Params params;   // invalid -> file system
};

// Drives a worker with a MockIoEngine the test controls.
struct Driver
{
    // `block` is what the engine advertises as its direct-read alignment. It matters because the
    // worker tests congruence against it: at 1, everything is congruent and the decision is vacuous.
    // The real engine reports 4096.
    //
    // `mount_block` is what the WORKER is told the mount measured. They are the same number in production and by default here, because the engine is
    // built from the mount's block.
    //
    // They are separated only for the adoption tests, which need the worker to start with no
    // measurement (mount_block 0 -> provisional MaxProbeBlock) while the engine still advertises a
    // usable alignment. Passing 0 for BOTH would make the engine advertise an alignment of 0, and
    // every congruence test divides by it.
    // `depth` is the node-wide queue depth the streamer resolved for this mount. It reaches the worker
    // directly, so setting RUNAI_STREAMER_FS_QUEUE_DEPTH in the fixture no longer changes it.
    explicit Driver(Strategy strategy = Strategy::IoUringBuffered, size_t block = 4096,
                    std::optional<size_t> mount_block = std::nullopt,
                    unsigned depth = Config::default_fs_async_queue_depth) :
        worker(strategy,
               mount_block.value_or(block),
               depth,
               [this, block](Strategy, const posix_io::AsyncIoConfig & config)
               {
                   posix_io::Limits limits;
                   limits.max_read_bytesize = posix_io::max_read_bytesize();
                   limits.offset_alignment = block;
                   limits.buffer_alignment = block;

                   auto owned = std::make_unique<MockIoEngine>(config.depth, limits);
                   engine = owned.get();
                   return owned;
               })
    {}

    // execute() stages everything, since the window is larger than the chunk count. Staged is not
    // issued: the worker flushes in drain(), and it counts what IT issued - so a test must never call
    // flush() itself, or the worker sees completions for reads it does not believe are outstanding.
    void execute(Workload && w) { worker.execute(std::move(w), stopped); }

    // Issues the staged reads. Nothing is ready yet, so this harvests nothing.
    void issue() { worker.drain(stopped); }

    // Harvests and routes whatever the test completed.
    void route() { worker.drain(stopped); }

    std::vector<posix_io::RequestId> in_flight() { return engine->in_flight(); }

    size_t worker_bounced_bytes() const { return worker.bounced_bytes(); }
    size_t worker_bytes_read() const { return worker.bytes_read(); }

    std::atomic<bool> stopped{ false };
    AsyncIoWorker worker;
    MockIoEngine * engine = nullptr;   // owned by the worker
};

// Every response, in arrival order.
//
// Bounded, never pop(): the failure this suite is most likely to catch is a range that is never
// answered, and an unbounded pop turns that into a hung test rather than a failed one.
std::vector<common::Response> drain_responses(common::Responder & responder, unsigned count)
{
    std::vector<common::Response> out;
    for (unsigned i = 0; i < count; ++i)
    {
        auto response = responder.pop(5000 /* ms */);
        EXPECT_NE(response.ret, common::ResponseCode::TimedOut)
            << "only " << i << " of " << count << " ranges were answered";
        out.push_back(response);
    }
    return out;
}

// How many descriptors this process holds open on `path`. The fd lifetime is not observable from the
// worker's interface, and it is exactly what a leak would break, so it is read from /proc.
unsigned open_fds_on(const std::string & path)
{
    // The temp file's path is relative ("./name"); /proc reports the absolute one.
    char resolved[PATH_MAX];
    EXPECT_NE(::realpath(path.c_str(), resolved), nullptr);
    const std::string absolute = resolved;

    unsigned count = 0;
    DIR * dir = ::opendir("/proc/self/fd");
    EXPECT_NE(dir, nullptr);
    if (dir == nullptr)
    {
        return 0;
    }

    while (const struct dirent * entry = ::readdir(dir))
    {
        char link[PATH_MAX];
        const std::string full = std::string("/proc/self/fd/") + entry->d_name;
        const auto len = ::readlink(full.c_str(), link, sizeof(link) - 1);
        if (len > 0)
        {
            link[len] = '\0';
            if (absolute == link)
            {
                ++count;
            }
        }
    }

    ::closedir(dir);
    return count;
}

} // namespace

// A file smaller than one chunk: one read, one response per range, and the bytes land where they
// belong.
TEST(AsyncIoWorker, Reads_And_Answers_Every_Range)
{
    Fixture fixture({ 100, 200, 300 });
    Driver driver;

    driver.execute(fixture.workload());

    ASSERT_NE(driver.engine, nullptr);
    ASSERT_EQ(driver.engine->staged_count(), 1u) << "three small ranges pack into one chunk";

    driver.issue();
    driver.engine->complete_all();
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 3);

    std::set<unsigned> seen;
    for (const auto & r : responses)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::Success);
        EXPECT_TRUE(seen.insert(r.index).second) << "range " << r.index << " answered twice";
    }
    EXPECT_EQ(seen, (std::set<unsigned>{ 0, 1, 2 }));

    // The data, not just the codes: a chunk staged against the wrong destination completes just as
    // cleanly. Offsets are checked where there is more than one chunk to get wrong.
    EXPECT_EQ(fixture.got_at(0, 600), fixture.expected_at(0, 600));
}

// The reason MockIoEngine exists: a later chunk landing first must not let a range be answered before
// its own bytes have arrived.
TEST(AsyncIoWorker, Out_Of_Order_Completion)
{
    Fixture fixture({ ChunkSize, ChunkSize, ChunkSize });   // one chunk each
    Driver driver;

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 3u);

    driver.issue();

    const auto in_flight = driver.in_flight();
    ASSERT_EQ(in_flight.size(), 3u);

    // Last, first, middle.
    driver.engine->complete(in_flight[2]);
    driver.route();
    driver.engine->complete(in_flight[0]);
    driver.engine->complete(in_flight[1]);
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 3);

    std::set<unsigned> seen;
    for (const auto & r : responses)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::Success);
        seen.insert(r.index);
    }
    EXPECT_EQ(seen, (std::set<unsigned>{ 0, 1, 2 }));

    for (unsigned i = 0; i < 3; ++i)
    {
        EXPECT_EQ(fixture.got_at(i * ChunkSize, ChunkSize), fixture.expected_at(i * ChunkSize, ChunkSize))
            << "chunk " << i << " landed in the wrong place";
    }
}

// A short read is re-staged on the SAME request, resuming where it stopped - not reported as success,
// and not re-queued.
TEST(AsyncIoWorker, Short_Read_Is_Restaged)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());

    driver.issue();
    const auto id = driver.in_flight().front();

    driver.engine->complete_short(id, 1000);
    driver.route();

    // Nothing answered yet - the range is not complete. A short read must never be reported as
    // success, which is the whole reason bytes_transferred is checked.
    EXPECT_EQ(fixture.responder->pop(10 /* ms */).ret, common::ResponseCode::TimedOut);

    // The remainder was staged again, on the same id, resumed at the right offset.
    ASSERT_EQ(driver.engine->staged_count(), 1u);
    const auto & restaged = driver.engine->request(id);
    EXPECT_EQ(restaged.offset, 1000u);
    EXPECT_EQ(restaged.bytesize, ChunkSize - 1000);

    driver.issue();
    driver.engine->complete(id);
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 1);
    EXPECT_EQ(responses.front().ret, common::ResponseCode::Success);
    EXPECT_EQ(fixture.got_at(0, ChunkSize), fixture.expected_at(0, ChunkSize));
}

// Zero further bytes while bytes are still owed: the file is shorter than the caller asked for.
TEST(AsyncIoWorker, Eof_Is_Reported)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());

    driver.issue();
    const auto id = driver.in_flight().front();

    driver.engine->complete_short(id, 0);
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 1);
    EXPECT_EQ(responses.front().ret, common::ResponseCode::EofError);
}

// The worker turns the kernel's errno into a ResponseCode, and the engine no longer does.
//
// The engine passes the raw result through because mapping needs the file: EINVAL means our alignment
// rule broke on a direct fd, and something else on a buffered one. Only the worker knows how the file
// was opened.
//
// EIO is used here because it maps to a code of its own. It shows the mapping really happens, which a
// value that maps to UnknownError could not - almost everything maps to UnknownError.
TEST(AsyncIoWorker, Kernel_Errno_Is_Mapped_By_The_Worker)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());

    driver.issue();
    const auto id = driver.in_flight().front();

    driver.engine->fail(id, EIO);
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 1);

    // FileAccessError, so an operator is sent to the storage. Reporting this as UnknownError would
    // hide a real disk problem.
    EXPECT_EQ(responses.front().ret, common::ResponseCode::FileAccessError);
}

// A failure that is OUR fault must never be reported as a storage fault. EFAULT means we handed the
// kernel a bad address, and the storage had no part in it.
TEST(AsyncIoWorker, Our_Own_Errors_Are_Not_Reported_As_Storage_Faults)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());

    driver.issue();
    const auto id = driver.in_flight().front();

    driver.engine->fail(id, EFAULT);
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 1);

    EXPECT_EQ(responses.front().ret, common::ResponseCode::UnknownError);
    EXPECT_NE(responses.front().ret, common::ResponseCode::FileAccessError)
        << "this would send an operator to investigate storage that is working correctly";
}

// A zero-sized range in a file that cannot be opened fails WITH the file.
//
// It reads nothing, so no chunk is ever made for it, and the file used to be opened only when a chunk
// was staged. A workload whose ranges are all zero-sized therefore never opened the file, never
// noticed it was missing, and answered Success.
//
// The synchronous reader opens the file for every batch before it looks at any range size, so it has
// always reported FileAccessError here. Which reader served a request is meant to be invisible to the
// caller, so they must agree.
//
// Found by running the Python suite against the REAL library. The mock library used by `make test`
// stubs this path, so nothing in the normal suite covered it - which is why this test is here.
TEST(AsyncIoWorker, Zero_Sized_Ranges_On_An_Unopenable_File_Fail_With_The_File)
{
    Fixture fixture({ 0, 0 });
    Driver driver;

    driver.execute(fixture.workload("/nonexistent-directory/definitely-missing.bin"));

    const auto responses = drain_responses(*fixture.responder, 2);
    ASSERT_EQ(responses.size(), 2u) << "every range owes a response, even one of zero bytes";

    for (const auto & r : responses)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::FileAccessError)
            << "the file does not exist, so the caller must be told";
    }
}

// One file, one answer. A batch that mixes zero-sized and real ranges must not report success for the
// empty ones and failure for the rest - they all failed for the same reason, that the file is missing.
TEST(AsyncIoWorker, Mixed_Ranges_On_An_Unopenable_File_All_Fail)
{
    Fixture fixture({ 0, ChunkSize });
    Driver driver;

    driver.execute(fixture.workload("/nonexistent-directory/definitely-missing.bin"));
    driver.issue();
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 2);
    ASSERT_EQ(responses.size(), 2u);

    for (const auto & r : responses)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::FileAccessError)
            << "a zero-sized range must not report success while its neighbours fail";
    }
}

// open(O_RDONLY) SUCCEEDS on a directory, so nothing before the read notices. Each reader then finds
// out somewhere different - measured: libaio refuses the whole io_submit with EINVAL, io_uring
// completes the read with -EINVAL, and pread returns EISDIR. The worker answers it at open instead, so
// all three say the same thing.
//
// FileAccessError, not UnknownError: this is one bad path, and the other files of the submission are
// fine. It also removes a dependency on an accident - a directory cannot be opened with O_DIRECT, so
// its fd is always buffered, and EINVAL on a DIRECT fd maps to UnknownError and aborts everything.
TEST(AsyncIoWorker, A_Directory_Is_Not_A_File)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload("/tmp"));
    driver.issue();
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 1);
    ASSERT_EQ(responses.size(), 1u);
    EXPECT_EQ(responses[0].ret, common::ResponseCode::FileAccessError);
}

// The same answer whether or not the ranges have bytes in them. A zero-sized range never reaches the
// read, so without the check at open it would report Success for a path that cannot be read at all.
TEST(AsyncIoWorker, A_Directory_Fails_Its_Zero_Sized_Ranges_Too)
{
    Fixture fixture({ 0, 0 });
    Driver driver;

    driver.execute(fixture.workload("/tmp"));

    const auto responses = drain_responses(*fixture.responder, 2);
    ASSERT_EQ(responses.size(), 2u);

    for (const auto & r : responses)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::FileAccessError)
            << "a directory must not answer Success just because the range was empty";
    }
}

// A zero-sized range reads nothing but owes a response like any other - answered at enqueue, since no
// completion will ever arrive for it.
TEST(AsyncIoWorker, Zero_Sized_Range_Is_Answered)
{
    Fixture fixture({ 100, 0, 100 });
    Driver driver;

    driver.execute(fixture.workload());

    // Answered before any I/O completes.
    const auto responses = drain_responses(*fixture.responder, 1);
    EXPECT_EQ(responses.front().index, 1);
    EXPECT_EQ(responses.front().ret, common::ResponseCode::Success);

    driver.issue();
    driver.engine->complete_all();
    driver.route();

    const auto rest = drain_responses(*fixture.responder, 2);
    for (const auto & r : rest)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::Success);
    }
}

// Stopping mid-flight still owes a response for every range. Staged-but-unissued chunks are the trap:
// they hold window credit and no completion will ever arrive for them, so a teardown that walks only
// the in-flight set leaves the caller waiting forever.
TEST(AsyncIoWorker, Stop_Answers_Every_Range)
{
    Fixture fixture({ ChunkSize, ChunkSize, ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 3u) << "staged, and deliberately never issued";

    driver.stopped = true;
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 3);
    for (const auto & r : responses)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::FinishedError);
    }
}

// Whether a file is opened with O_DIRECT is decided per file, from congruence - the address and the
// file offset having the same remainder. Aligning the buffer alone is not enough, and this is the
// case that proves it: the same aligned buffer, shifted by one byte, must fall back to buffered.
//
// The mock records what each request was staged with, which is the only way to see the decision: both
// modes return the same bytes.
// EINVAL on a direct read is re-staged buffered, not failed.
//
// It used to become UnknownError - the one code that tells a caller to abort the WHOLE submission -
// so a single mount needing a larger block than DirectBlockSize ended a model load. Nothing was read
// when the kernel refuses for alignment, so the same bytes can simply be re-issued without O_DIRECT.
TEST(AsyncIoWorker, Direct_Refused_For_Alignment_Is_Retried_Buffered)
{
    std::vector<char> raw(2 * ChunkSize + 4096);
    char * const aligned = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % 4096);

    Fixture fixture({ ChunkSize });
    Driver driver(Strategy::IoUringDirect);

    fixture.request[0].ranges[0].dst = aligned;

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 1u);

    const auto first = driver.engine->staged().front();
    ASSERT_TRUE(driver.engine->request(first).file.direct) << "the setup must produce a DIRECT read";

    // Issued before it can be completed - the mock refuses to complete what was never given to it.
    driver.issue();

    // The kernel refusing our alignment. Positive errno, as a real engine reports it.
    driver.engine->fail(first, EINVAL);
    driver.route();

    // Re-staged rather than answered. staged() empties on issue, so a request sitting there now is
    // the retry; history() holds both, which is what proves there were two.
    ASSERT_EQ(driver.engine->staged_count(), 1u) << "the chunk must be re-issued, not failed";
    ASSERT_EQ(driver.engine->history().size(), 2u) << "exactly one retry, not a loop";

    const auto second = driver.engine->staged().back();
    EXPECT_FALSE(driver.engine->request(second).file.direct)
        << "the retry must drop O_DIRECT, or it would be refused again";
    EXPECT_EQ(driver.engine->request(second).offset, driver.engine->request(first).offset)
        << "nothing was read, so the retry must start where the direct read did";

    // And it completes normally.
    driver.issue();
    driver.engine->complete(second);
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 1);
    ASSERT_EQ(responses.size(), 1u) << "a range refused for alignment must still be answered";
    EXPECT_EQ(responses[0].ret, common::ResponseCode::Success)
        << "the buffered retry succeeded, so the caller asked for bytes and got them";
}

// A demotion must not break requests of the same file that are already staged against the old
// descriptor.
//
// stage() writes the fd into the submission entry, and the kernel only reads it at submit time. So a
// request staged earlier in the SAME completion loop - by a short read, or by an earlier demotion -
// carries the old descriptor and is issued on the next flush. Two ways to get that wrong:
//
//   - closing the descriptor: the read fails with EBADF, and on libaio io_submit refuses the whole
//     batch, which aborts the submission;
//   - keeping it open but unchanged: the read is issued DIRECT again on a mount that just refused a
//     direct read, and its EINVAL arrives after batch_fd.direct is false - so the re-stage branch is
//     skipped and it maps to FileAccessError, failing the file.
//
// Both are checked here, on the descriptor the staged request actually holds.
TEST(AsyncIoWorker, Demotion_Keeps_Already_Staged_Requests_Working)
{
    std::vector<char> raw(3 * ChunkSize + 4096);
    char * const aligned = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % 4096);

    // Two chunks of one file, so they share a batch and therefore a descriptor.
    Fixture fixture({ ChunkSize, ChunkSize });
    Driver driver(Strategy::IoUringDirect);

    fixture.request[0].ranges[0].dst = aligned;
    fixture.request[0].ranges[1].dst = aligned + ChunkSize;

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 2u) << "two chunks, one per range";

    const auto staged = driver.engine->staged();
    const auto first = staged.front();
    const auto second = staged.back();

    ASSERT_TRUE(driver.engine->request(first).file.direct) << "the setup must produce DIRECT reads";
    ASSERT_TRUE(driver.engine->request(second).file.direct);
    ASSERT_EQ(driver.engine->request(first).file.fd, driver.engine->request(second).file.fd)
        << "both chunks are the same file, so they must share one descriptor";

    driver.issue();

    // Ordered on purpose. The short read re-stages `first` while the file is still direct, and only
    // then does `second` demote it - which is exactly the case where a staged request is left holding
    // a descriptor someone else is about to replace.
    driver.engine->complete_short(first, ChunkSize / 2);
    driver.engine->fail(second, EINVAL);
    driver.route();

    ASSERT_EQ(driver.engine->staged_count(), 2u) << "the short read and the refused read both re-stage";

    // The descriptor the re-staged remainder of `first` will be issued against.
    const int fd = driver.engine->request(driver.engine->staged().front()).file.fd;

    const int flags = ::fcntl(fd, F_GETFL);
    ASSERT_NE(flags, -1) << "the descriptor a staged request holds was closed by the demotion,"
                         << " so its next flush would fail with EBADF";
    EXPECT_EQ(flags & O_DIRECT, 0)
        << "the descriptor is still direct, so this read would be refused for alignment again -"
        << " and by then the file is marked buffered, so it would fail rather than re-stage";

    // And the whole thing still answers.
    driver.issue();
    driver.engine->complete_all();
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 2);
    for (const auto & response : responses)
    {
        EXPECT_EQ(response.ret, common::ResponseCode::Success);
    }

    // Read from `aligned`, not fixture.buffer: the destinations were redirected there above.
    const std::vector<char> got(aligned, aligned + 2 * ChunkSize);
    EXPECT_EQ(got, fixture.expected_at(0, 2 * ChunkSize))
        << "every byte must still land, whichever descriptor carried it";
}

TEST(AsyncIoWorker, Direct_Only_When_Congruent)
{
    // The destination must be page aligned AND in step with the file offset. Reading from offset 0
    // into an aligned address gives both.
    std::vector<char> raw(2 * ChunkSize + 4096);
    char * const aligned = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % 4096);

    {
        Fixture fixture({ ChunkSize });
        Driver driver(Strategy::IoUringDirect);

        // Point the range at the aligned address, reading from file offset 0.
        fixture.request[0].ranges[0].dst = aligned;

        driver.execute(fixture.workload());
        ASSERT_EQ(driver.engine->staged_count(), 1u);

        const auto id = driver.engine->staged().front();
        EXPECT_TRUE(driver.engine->request(id).file.direct)
            << "offset 0 into an aligned address is congruent, so this must be direct";
    }

    {
        Fixture fixture({ ChunkSize });
        Driver driver(Strategy::IoUringDirect);

        // One byte along. Still a perfectly good address, and now out of step with the file.
        fixture.request[0].ranges[0].dst = aligned + 1;

        driver.execute(fixture.workload());
        ASSERT_EQ(driver.engine->staged_count(), 1u);

        const auto id = driver.engine->staged().front();
        EXPECT_FALSE(driver.engine->request(id).file.direct)
            << "not congruent, so no part of the file could be read directly - buffered instead";
    }
}

// A buffered strategy never opens direct, whatever the addresses happen to be.
TEST(AsyncIoWorker, Buffered_Strategy_Never_Opens_Direct)
{
    std::vector<char> raw(2 * ChunkSize + 4096);
    char * const aligned = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % 4096);

    Fixture fixture({ ChunkSize });
    Driver driver(Strategy::IoUringBuffered);

    fixture.request[0].ranges[0].dst = aligned;   // congruent, and still must be buffered

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 1u);

    const auto id = driver.engine->staged().front();
    EXPECT_FALSE(driver.engine->request(id).file.direct);
}

// The case edge bouncing exists for: a region that does not start on a block boundary.
//
// The first pass cannot be issued as asked - its offset is not a block multiple - so it reads the
// whole block into scratch and copies out the part that was wanted. The mock fills from the ABSOLUTE
// file offset, so the bytes prove the copy came from the right place.
TEST(AsyncIoWorker, Direct_Bounces_A_Partial_Head)
{
    constexpr size_t Block = 4096;
    constexpr size_t Start = 1000;   // 1000 bytes into a block

    // A destination whose remainder matches the file offset - congruent, so direct is possible.
    std::vector<char> raw(4 * ChunkSize + 2 * Block);
    char * const base = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);
    char * const dst = base + Start;

    Fixture fixture({ ChunkSize }, 1, Start);
    fixture.request[0].ranges[0].dst = dst;

    Driver driver(Strategy::IoUringDirect, Block);
    driver.execute(fixture.workload());

    // Two chunks, because this fixture's chunk size equals the block: the range crosses a chunk
    // boundary at 4096. The FIRST is the one with the partial head.
    ASSERT_EQ(driver.engine->staged_count(), 2u);
    const auto id = driver.engine->staged().front();

    // What the kernel was actually asked for: rounded down, one whole block, into scratch.
    const auto & staged = driver.engine->request(id);
    EXPECT_TRUE(staged.file.direct);
    EXPECT_EQ(staged.offset, 0u) << "rounded down to the block below 1000";
    EXPECT_EQ(staged.bytesize, Block);
    EXPECT_NE(staged.buffer, dst) << "a bounced pass must not read into the destination";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(staged.buffer) % Block, 0u) << "scratch must be aligned";

    // Walk it to completion. A partial region takes at most three passes - head, middle, tail - and
    // extra rounds are harmless because nothing is left staged.
    for (int round = 0; round < 4; ++round)
    {
        driver.issue();
        driver.engine->complete_all();
        driver.route();
    }

    const auto responses = drain_responses(*fixture.responder, 1);
    EXPECT_EQ(responses.front().ret, common::ResponseCode::Success);

    // The bytes the caller asked for, from the offset it asked for.
    const std::vector<char> got(dst, dst + ChunkSize);
    EXPECT_EQ(got, fixture.expected_at(Start, ChunkSize));
}

// A head pass that does NOT finish the chunk. This is what separates recording the WANTED bytes from
// recording what the kernel returned.
//
// With a 512-byte block, a chunk starting 1000 bytes in has only 24 useful bytes in its first block.
// The kernel returns 512. Recording 512 would move the cursor 488 bytes too far, and those bytes would
// never be read - no error anywhere, just wrong data.
TEST(AsyncIoWorker, Bounced_Head_Advances_By_The_Wanted_Bytes)
{
    constexpr size_t Block = 512;
    constexpr size_t Start = 1000;

    std::vector<char> raw(4 * ChunkSize + 2 * Block);
    char * const base = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);
    char * const dst = base + (Start % Block);   // congruent for THIS block size

    Fixture fixture({ ChunkSize }, 1, Start);
    fixture.request[0].ranges[0].dst = dst;

    Driver driver(Strategy::IoUringDirect, Block);
    driver.execute(fixture.workload());

    const auto first = driver.engine->staged().front();
    const auto & staged = driver.engine->request(first);
    EXPECT_EQ(staged.bytesize, Block) << "one block, however much of it is wanted";
    EXPECT_EQ(staged.offset, Start - (Start % Block));

    // Several passes: the head yields 24 bytes, then whole blocks, then a tail.
    for (int round = 0; round < 12; ++round)
    {
        driver.issue();
        driver.engine->complete_all();
        driver.route();
    }

    EXPECT_EQ(drain_responses(*fixture.responder, 1).front().ret, common::ResponseCode::Success);

    // Every byte, from the right place. A cursor that ran ahead would leave a hole here.
    EXPECT_EQ(std::vector<char>(dst, dst + ChunkSize), fixture.expected_at(Start, ChunkSize));
}

// A pass that starts on a block boundary with at least a block to read goes STRAIGHT into the
// destination. This is every chunk in the middle of a region, and it is why the cost is 0.1% rather
// than everything.
TEST(AsyncIoWorker, Direct_Middle_Is_Not_Bounced)
{
    constexpr size_t Block = 4096;

    std::vector<char> raw(4 * ChunkSize + Block);
    char * const dst = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);

    Fixture fixture({ ChunkSize }, 1, 0);   // starts on a boundary, and ChunkSize is a block multiple
    fixture.request[0].ranges[0].dst = dst;

    Driver driver(Strategy::IoUringDirect, Block);
    driver.execute(fixture.workload());

    ASSERT_EQ(driver.engine->staged_count(), 1u);
    const auto id = driver.engine->staged().front();
    const auto & staged = driver.engine->request(id);

    EXPECT_TRUE(staged.file.direct);
    EXPECT_EQ(staged.buffer, dst) << "nothing to bounce, so read into the destination itself";
    EXPECT_EQ(staged.offset, 0u);
    EXPECT_EQ(staged.bytesize, ChunkSize);

    driver.issue();
    driver.engine->complete_all();
    driver.route();

    EXPECT_EQ(drain_responses(*fixture.responder, 1).front().ret, common::ResponseCode::Success);
    EXPECT_EQ(std::vector<char>(dst, dst + ChunkSize), fixture.expected_at(0, ChunkSize));
}

// Scratch buffers must come back. A leak drains the pool, and every later read silently loses the
// direct path - a slowdown with no error anywhere.
TEST(AsyncIoWorker, Scratch_Is_Returned)
{
    constexpr size_t Block = 4096;
    constexpr size_t Start = 500;

    std::vector<char> raw(4 * ChunkSize + 2 * Block);
    char * const base = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);

    // The pool holds one buffer per in-flight read, so it is sized to the queue depth. A DEPTH OF 2
    // makes a leak show up almost at once: lose one buffer per region and the third region has none,
    // and a read with no scratch is failed rather than silently made buffered. The depth reaches the
    // worker through the Driver - the environment variable the fixture sets does not.
    Driver driver(Strategy::IoUringDirect, Block, std::nullopt, 2 /* depth, so 2 scratch buffers */);

    for (int i = 0; i < 8; ++i)
    {
        Fixture fixture({ ChunkSize }, 1, Start);
        fixture.request[0].ranges[0].dst = base + Start;

        driver.execute(fixture.workload());

        for (int round = 0; round < 8; ++round)
        {
            driver.issue();
            driver.engine->complete_all();
            driver.route();
        }

        EXPECT_EQ(drain_responses(*fixture.responder, 1).front().ret, common::ResponseCode::Success)
            << "region " << i << " failed - the pool has probably been drained by a leak";
    }
}

// A region of consecutive ranges bounces AT MOST TWICE, however long it is: once for the partial
// block at the start, once for the partial block at the end. Everything between starts and ends on a
// block boundary, so it is read straight into the destination.
//
// This is the whole cost argument. If it were one bounce per chunk, a 40 GB model would copy several
// percent of itself instead of about 0.1%.
TEST(AsyncIoWorker, A_Region_Bounces_At_Most_Twice)
{
    constexpr size_t Block = 4096;
    constexpr size_t Start = 1000;          // partial block at the start
    constexpr size_t Size = 3 * ChunkSize + 1096;   // and a partial block at the end

    std::vector<char> raw(Size + 4 * Block);
    char * const base = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);
    char * const dst = base + Start;        // congruent with the file offset

    Fixture fixture({ Size }, 1, Start);
    fixture.request[0].ranges[0].dst = dst;

    Driver driver(Strategy::IoUringDirect, Block);
    driver.execute(fixture.workload());

    for (int round = 0; round < 10; ++round)
    {
        driver.issue();
        driver.engine->complete_all();
        driver.route();
    }

    EXPECT_EQ(drain_responses(*fixture.responder, 1).front().ret, common::ResponseCode::Success);

    // BYTES COPIED, not passes. A pass that copies 24 bytes and one that copies 4096 are not the same
    // cost, and the cost is what the design's "about 0.1%" claims.
    //
    // The bound is per REGION, not per chunk: at most one partial block at each end, so at most two
    // blocks whatever the region's length. A region twice as long copies exactly the same amount.
    EXPECT_LE(driver.worker_bounced_bytes(), 2 * Block)
        << "a region must copy at most one partial block at each end";
    EXPECT_GT(driver.worker_bounced_bytes(), 0u) << "this region has partial blocks, so it must copy some";

    // And the middle really did go straight to the destination, rather than being copied cheaply.
    unsigned direct_into_destination = 0;
    for (const auto & request : driver.engine->history())
    {
        if (request.buffer >= dst && request.buffer < dst + Size)
        {
            ++direct_into_destination;
        }
    }
    EXPECT_GT(direct_into_destination, 0u) << "the middle must go straight into the destination";

    // And the data is still right, which is what makes the count worth having.
    EXPECT_EQ(std::vector<char>(dst, dst + Size), fixture.expected_at(Start, Size));
}

// The bound is per REGION, so a longer region copies the SAME amount, not more. That is the whole
// claim behind "about 0.1%": the cost is fixed per region while the bytes read grow.
TEST(AsyncIoWorker, Copied_Bytes_Do_Not_Grow_With_The_Region)
{
    constexpr size_t Block = 4096;
    constexpr size_t Start = 1000;

    const auto copied_for = [](size_t size, size_t start) -> size_t
    {
        std::vector<char> raw(size + 4 * Block);
        char * const base = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);
        char * const dst = base + start;

        Fixture fixture({ size }, 1, start);
        fixture.request[0].ranges[0].dst = dst;

        Driver driver(Strategy::IoUringDirect, Block);
        driver.execute(fixture.workload());

        for (int round = 0; round < 40; ++round)
        {
            driver.issue();
            driver.engine->complete_all();
            driver.route();
        }

        EXPECT_EQ(drain_responses(*fixture.responder, 1).front().ret, common::ResponseCode::Success);
        EXPECT_EQ(std::vector<char>(dst, dst + size), fixture.expected_at(start, size));

        return driver.worker_bounced_bytes();
    };

    const size_t small = copied_for(2 * ChunkSize + 1096, Start);
    const size_t large = copied_for(16 * ChunkSize + 1096, Start);

    EXPECT_EQ(small, large) << "eight times the bytes read, and the same bytes copied";
    EXPECT_LE(large, 2 * Block);
}

// The warning fires when direct reads have stopped paying. There is no error for that case - the
// reads all succeed - so this line is the only signal, and a test should prove it appears.
//
// A non-congruent destination is refused direct and read buffered, so it copies nothing. To reach the
// warning the reads must be direct AND copying a lot, which is what a tiny block plus a region full of
// partial edges produces.
TEST(AsyncIoWorker, Bouncing_Is_Counted_Against_Bytes_Read)
{
    constexpr size_t Block = 4096;
    constexpr size_t Start = 1000;
    constexpr size_t Size = 2 * ChunkSize + 1096;

    std::vector<char> raw(Size + 4 * Block);
    char * const base = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);
    char * const dst = base + Start;

    Fixture fixture({ Size }, 1, Start);
    fixture.request[0].ranges[0].dst = dst;

    Driver driver(Strategy::IoUringDirect, Block);
    driver.execute(fixture.workload());

    for (int round = 0; round < 12; ++round)
    {
        driver.issue();
        driver.engine->complete_all();
        driver.route();
    }

    EXPECT_EQ(drain_responses(*fixture.responder, 1).front().ret, common::ResponseCode::Success);

    // Both numbers are needed to judge anything. Copied bytes alone cannot say whether that is a lot.
    EXPECT_EQ(driver.worker_bytes_read(), Size) << "every byte the caller asked for was delivered";
    EXPECT_LE(driver.worker_bounced_bytes(), 2 * Block);
    EXPECT_LT(driver.worker_bounced_bytes(), driver.worker_bytes_read())
        << "copying more than was read would mean something is counted twice";
}

// A region that starts AND ends on a block boundary bounces not at all.
TEST(AsyncIoWorker, An_Aligned_Region_Never_Bounces)
{
    constexpr size_t Block = 4096;
    constexpr size_t Size = 3 * ChunkSize;

    std::vector<char> raw(Size + Block);
    char * const dst = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Block);

    Fixture fixture({ Size }, 1, 0);
    fixture.request[0].ranges[0].dst = dst;

    Driver driver(Strategy::IoUringDirect, Block);
    driver.execute(fixture.workload());

    for (int round = 0; round < 10; ++round)
    {
        driver.issue();
        driver.engine->complete_all();
        driver.route();
    }

    EXPECT_EQ(drain_responses(*fixture.responder, 1).front().ret, common::ResponseCode::Success);

    EXPECT_EQ(driver.worker_bounced_bytes(), 0u)
        << "an aligned region has no partial block, so not one byte should be copied";

    EXPECT_EQ(std::vector<char>(dst, dst + Size), fixture.expected_at(0, Size));
}

// Teardown must not report a range while the kernel still holds its destination: a response promises
// nothing will write there again, and under the Python ring that buffer goes straight to the next
// submission. So stopping with reads ISSUED has to wait for them, not abandon them.
TEST(AsyncIoWorker, Stop_Waits_For_Reads_In_Flight)
{
    Fixture fixture({ ChunkSize, ChunkSize, ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());
    driver.issue();
    ASSERT_EQ(driver.engine->in_flight_count(), 3u) << "three reads the kernel now owns";

    // One completion per wait, as a kernel would hand them over - so the drain has to go round more
    // than once and cannot succeed by accident.
    driver.engine->set_auto_complete_on_wait(1);

    driver.stopped = true;
    driver.route();

    // The assertion that matters. Reporting with reads outstanding would leave these non-zero.
    EXPECT_EQ(driver.engine->in_flight_count(), 0u)
        << "teardown returned while the kernel still held destinations";

    const auto responses = drain_responses(*fixture.responder, 3);
    for (const auto & r : responses)
    {
        EXPECT_EQ(r.ret, common::ResponseCode::FinishedError);
    }
}

// One descriptor per batch however many chunks it has, released when the workload ends.
TEST(AsyncIoWorker, Fd_Opened_Once_And_Closed_At_Finalize)
{
    Fixture fixture({ ChunkSize, ChunkSize, ChunkSize });   // three chunks, one file
    Driver driver;

    const auto before = open_fds_on(fixture.file.path);

    driver.execute(fixture.workload());
    EXPECT_EQ(open_fds_on(fixture.file.path), before + 1) << "three chunks must share one descriptor";

    driver.issue();
    driver.engine->complete_all();
    driver.route();

    drain_responses(*fixture.responder, 3);
    EXPECT_EQ(open_fds_on(fixture.file.path), before) << "descriptor leaked past the last response";
}

// Waiting with nothing issued waits on an empty ring for a completion that will never arrive. Staged
// is not issued, so the count that decides this is the issued one.
TEST(AsyncIoWorker, Does_Not_Block_With_Nothing_Issued)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());

    // Staged but never flushed - so nothing is in flight.
    driver.engine->set_flush_stalled(true);
    driver.issue();

    ASSERT_GT(driver.engine->waits(), 0u);
    EXPECT_EQ(driver.engine->last_wait_mode(), WaitMode::NonBlocking)
        << "blocking here waits for a completion that cannot come";
}

// Having reads in flight is not a reason to stop. A window with room still has work to give, so the
// thread should go back and submit rather than sleep on io_getevents.
//
// Blocking whenever anything was outstanding cost exactly that: with a depth of 64 and five reads
// issued, the thread parked instead of filling the other fifty-nine. It happens on every workload's
// ramp-up. fio, on the same engine and one thread, waits only once in-flight reaches the target
// depth - and reaches 17 GB/s on a mount where we reached 11.7.
//
// Pinned because nothing else would notice it regressing: the same reads are issued and the same
// bytes arrive either way, and the cost is only a slower run.
TEST(AsyncIoWorker, Does_Not_Block_While_The_Window_Has_Room)
{
    // One chunk against a depth of 3 (the floor), so something is in flight and the window is not
    // full - the case that used to block.
    Fixture fixture({ ChunkSize });
    Driver driver(Strategy::IoUringBuffered, 4096, std::nullopt, 1 /* depth -> floors at 3 */);

    driver.execute(fixture.workload());
    driver.issue();

    ASSERT_EQ(driver.engine->depth(), 3u) << "this test needs room left in the window";
    EXPECT_EQ(driver.engine->last_wait_mode(), WaitMode::NonBlocking)
        << "one read outstanding in a window of three is not a reason to stop submitting";
}

// And once the window IS full there is nothing else to do, so blocking is right - and bounded, or a
// stopped worker would never notice.
TEST(AsyncIoWorker, Blocks_Once_The_Window_Is_Full)
{
    Fixture fixture({ ChunkSize, ChunkSize, ChunkSize });
    Driver driver(Strategy::IoUringBuffered, 4096, std::nullopt, 1 /* depth -> floors at 3 */);

    driver.execute(fixture.workload());
    driver.issue();

    ASSERT_EQ(driver.engine->depth(), 3u);
    EXPECT_EQ(driver.engine->last_wait_mode(), WaitMode::Block)
        << "with the window full there is no more to submit, so waiting is the only move";
    EXPECT_GT(driver.engine->last_wait_timeout_ms(), 0u) << "an unbounded wait has no way to notice stopped";
}

// A worker that started with no measurement adopts one a later submission carries.
//
// The case is a long-lived streamer - checkpoint restore - whose first submission could probe nothing,
// because every path was missing or unreadable. Without adoption it would keep the provisional block
// for the life of the process.
//
// The destination is congruent at 4096 and NOT at the provisional MaxProbeBlock, so the two answers
// differ: refused direct before adoption, read direct after it.
TEST(AsyncIoWorker, A_Measured_Block_Is_Adopted_When_The_Engine_Can_Serve_It)
{
    std::vector<char> raw(3 * posix_io::MaxProbeBlock);
    char * const at_block = raw.data()
                          + ((-reinterpret_cast<uintptr_t>(raw.data())) % posix_io::MaxProbeBlock)
                          + ChunkSize;   // 4096-aligned, deliberately NOT MaxProbeBlock-aligned

    Fixture fixture({ ChunkSize });
    // No mount measurement, so the worker starts provisional; the engine still advertises 4096.
    Driver driver(Strategy::IoUringDirect, ChunkSize, 0);

    fixture.request[0].ranges[0].dst = at_block;

    auto workload = fixture.workload();
    workload.direct_block = ChunkSize;   // what a later submission measured

    driver.execute(std::move(workload));
    ASSERT_EQ(driver.engine->staged_count(), 1u);

    const auto staged = driver.engine->staged().front();
    EXPECT_TRUE(driver.engine->request(staged).file.direct)
        << "after adopting 4096 this destination is congruent, so the read must be direct";
}

// A measured block LARGER than the engine was built for is refused.
//
// The scratch pool is sized and aligned once, in capacity(), to the block in force then. A bounced
// pass reads a whole block into one of its buffers, so adopting a larger block would have the kernel
// write past the end of one.
//
// The old code adopted it unconditionally, arguing that the provisional value is the ladder's top rung
// so a measurement could only shrink. That covers the ladder and not statx, which reports whatever the
// filesystem says.
TEST(AsyncIoWorker, A_Block_Larger_Than_The_Scratch_Buffers_Is_Refused)
{
    constexpr size_t Reported = 2 * posix_io::MaxProbeBlock;

    // Aligned to the REPORTED block, not just the provisional one. Reading from offset 0, that makes
    // the destination congruent either way - so if the larger block were adopted, the pass really
    // would be planned at it and really would read into a scratch buffer half its size. Aligning only
    // to MaxProbeBlock would break congruence at the larger block and send the file to a buffered
    // read, which passes this test without ever demonstrating the overflow.
    std::vector<char> raw(3 * Reported);
    char * const aligned = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % Reported);

    Fixture fixture({ ChunkSize });
    // Provisional at MaxProbeBlock, which is also what the engine advertises - so the setup agrees
    // with itself and the only variable is the block the workload reports.
    Driver driver(Strategy::IoUringDirect, posix_io::MaxProbeBlock, 0);

    fixture.request[0].ranges[0].dst = aligned;

    auto workload = fixture.workload();
    workload.direct_block = Reported;   // more than the scratch buffers hold

    driver.execute(std::move(workload));
    ASSERT_EQ(driver.engine->staged_count(), 1u);

    // The chunk is smaller than a block, so this is the tail case: one whole block into scratch. Its
    // size is the block actually in force, which is what makes the refusal observable - and the size
    // the kernel would have written into a buffer of MaxProbeBlock bytes.
    const auto staged = driver.engine->staged().front();
    EXPECT_TRUE(driver.engine->request(staged).file.direct)
        << "the setup must produce a DIRECT read, or the bounce never happens and this proves nothing";
    EXPECT_EQ(driver.engine->request(staged).bytesize, posix_io::MaxProbeBlock)
        << "the pass was planned at the reported block, which is larger than the scratch buffer it"
        << " reads into";
}

// An engine that failed mid-run is never used again, and later submissions are answered rather than
// staged into it.
//
// io_uring_submit and io_getevents failing with something that is not backpressure are the only ways
// here, and neither recovers. Retrying would also be unsafe: requests staged before the failure are
// neither issued nor dropped, and they hold descriptor numbers finalize() has since closed - which the
// next open takes straight back.
TEST(AsyncIoWorker, A_Dead_Engine_Is_Not_Reused)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 1u);

    // The engine's ring is gone. Nothing staged in it will ever be issued.
    driver.engine->set_flush_result(common::ResponseCode::FsAsyncEngineError);
    driver.issue();

    const auto first = drain_responses(*fixture.responder, 1);
    ASSERT_EQ(first.size(), 1u);
    EXPECT_EQ(first[0].ret, common::ResponseCode::FsAsyncEngineError)
        << "the in-flight submission is failed with the engine's own code, not UnknownError";

    // A second submission arrives after the failure. It must be answered, and it must not reach the
    // engine - staging into a ring that is never flushed again would leave it holding the caller's
    // buffers for good.
    const auto staged_before = driver.engine->history().size();

    Fixture second({ ChunkSize });
    driver.execute(second.workload());

    EXPECT_EQ(driver.engine->history().size(), staged_before)
        << "a workload was staged into an engine that has already failed";

    const auto answers = drain_responses(*second.responder, 1);
    ASSERT_EQ(answers.size(), 1u) << "a submission after the failure must still be answered";
    EXPECT_EQ(answers[0].ret, common::ResponseCode::FsAsyncEngineError);
}

// Shutdown is not a failure, so it must not mark the engine dead - there is nothing left to protect,
// and the code the caller sees has to stay FinishedError.
TEST(AsyncIoWorker, Stopping_Reports_Finished_Rather_Than_An_Engine_Failure)
{
    Fixture fixture({ ChunkSize });
    Driver driver;

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 1u);

    driver.stopped = true;
    driver.issue();

    const auto responses = drain_responses(*fixture.responder, 1);
    ASSERT_EQ(responses.size(), 1u);
    EXPECT_EQ(responses[0].ret, common::ResponseCode::FinishedError);
}

// A failed reopen fails the FILE, not the whole submission.
//
// The kernel refuses a direct read, so the worker reopens the file buffered - and that open can fail
// for reasons that are entirely the environment's: the file was unlinked since, descriptors ran out,
// NFS went stale, permissions changed.
//
// It used to fall through and map the original EINVAL against a still-direct file. is_internal_error
// calls that ours, so it became UnknownError - the one code that tells a caller to abort everything
// and treat it as a bug in the streamer. For a file somebody deleted mid-read.
TEST(AsyncIoWorker, A_Failed_Reopen_Fails_Only_That_File)
{
    std::vector<char> raw(2 * ChunkSize + 4096);
    char * const aligned = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % 4096);

    Fixture fixture({ ChunkSize });
    Driver driver(Strategy::IoUringDirect);

    fixture.request[0].ranges[0].dst = aligned;

    driver.execute(fixture.workload());
    ASSERT_EQ(driver.engine->staged_count(), 1u);

    const auto first = driver.engine->staged().front();
    ASSERT_TRUE(driver.engine->request(first).file.direct) << "the setup must produce a DIRECT read";

    driver.issue();

    // The file disappears between the direct open and the completion, so the buffered reopen cannot
    // succeed. This is the case the old code called an internal error.
    ASSERT_EQ(::unlink(fixture.file.path.c_str()), 0);

    driver.engine->fail(first, EINVAL);
    driver.route();

    const auto responses = drain_responses(*fixture.responder, 1);
    ASSERT_EQ(responses.size(), 1u);
    EXPECT_EQ(responses[0].ret, common::ResponseCode::FileAccessError)
        << "a file that vanished is this file's failure; UnknownError would end the whole submission";
}

// An engine that could never be built is reported like one that died later, not as an internal error.
//
// The two are the same thing to a caller: this mount is not read asynchronously, and the synchronous
// pool serves it. UnknownError would say something else entirely - abort every submission and restart
// the streamer - for a condition with a working fallback.
TEST(AsyncIoWorker, An_Engine_That_Cannot_Be_Built_Is_Not_An_Internal_Error)
{
    Fixture fixture({ ChunkSize });

    bool told = false;
    AsyncIoWorker worker(Strategy::IoUringBuffered, 4096, Config::default_fs_async_queue_depth,
                         [](Strategy, const posix_io::AsyncIoConfig &) -> std::unique_ptr<posix_io::IoEngine>
                         {
                             return nullptr;   // this host cannot serve the strategy it was given
                         },
                         [&told]() { told = true; });

    std::atomic<bool> stopped{ false };
    worker.execute(fixture.workload(), stopped);

    const auto responses = drain_responses(*fixture.responder, 1);
    ASSERT_EQ(responses.size(), 1u) << "a workload that never got a window must still be answered";
    EXPECT_EQ(responses[0].ret, common::ResponseCode::FsAsyncEngineError);

    EXPECT_TRUE(told) << "the streamer must learn the mount is dead, or it keeps routing here";
}

}; // namespace runai::llm::streamer::impl
