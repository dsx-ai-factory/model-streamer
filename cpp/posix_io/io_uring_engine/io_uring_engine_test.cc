#include "posix_io/io_uring_engine/io_uring_engine.h"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "posix_io/alignment/alignment.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif

#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif

// Whether this kernel will register a buffer, asked of the kernel for the same reason as ring_works()
// below: an expectation must never be computed by the thing it is checking. Reading IoUringProbe or
// the engine's own answer would make the assertion vacuous.
bool ring_registers()
{
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));

    const int fd = ::syscall(__NR_io_uring_setup, 8, &params);
    if (fd < 0)
    {
        return false;
    }

    alignas(4096) static unsigned char probe[4096];
    struct iovec iov;
    iov.iov_base = probe;
    iov.iov_len = sizeof(probe);

    const long ret = ::syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS, &iov, 1u);
    ::close(fd);
    return ret == 0;
}

// Ask the kernel directly - not IoUringProbe, and not IoUringEngine.
//
// The gate must not be computed by anything it gates. If it read the probe, a probe that wrongly
// reported "unavailable" would skip every test that would have caught it, and the suite would stay
// green while the engine went entirely untested.
bool ring_works()
{
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));

    const int fd = ::syscall(__NR_io_uring_setup, 8, &params);
    if (fd < 0)
    {
        return false;
    }

    ::close(fd);
    return true;
}

// Skipping is silent, and on a host that is supposed to have io_uring a silent skip is
// indistinguishable from a pass. RUNAI_STREAMER_REQUIRE_IO_URING says "io_uring works here", turning
// the skip into a failure. CI sets it, through BAZEL_TEST_FLAGS (.github/workflows/on-pr.yaml).
bool require_io_uring()
{
    const char * value = std::getenv("RUNAI_STREAMER_REQUIRE_IO_URING");
    return value != nullptr && std::string(value) == "1";
}

#define SKIP_WITHOUT_RING()                                                                   \
    do {                                                                                      \
        if (!ring_works())                                                                    \
        {                                                                                     \
            if (require_io_uring())                                                           \
            {                                                                                 \
                FAIL() << "io_uring_setup failed (" << std::strerror(errno) << ") but "        \
                       << "RUNAI_STREAMER_REQUIRE_IO_URING=1 says this host has io_uring";    \
            }                                                                                 \
            GTEST_SKIP() << "io_uring unavailable (" << std::strerror(errno)                  \
                         << "); set RUNAI_STREAMER_REQUIRE_IO_URING=1 where it should work";  \
        }                                                                                     \
    } while (0)

AsyncIoConfig config_with(unsigned depth)
{
    AsyncIoConfig config;
    config.depth = depth;
    config.chunk_bytesize = 1 << 20;

    // On, because the registration tests below are the reason this file exists in its current form.
    // The mount policy is tested where it is decided, in the router - not here.
    config.register_buffers = true;
    return config;
}

// A file of known bytes, plus the fd to read it through.
struct Fixture
{
    explicit Fixture(size_t bytesize) :
        data(utils::random::buffer(bytesize)),
        file(data),
        fd(::open(file.path.c_str(), O_RDONLY))
    {
        EXPECT_GE(fd, 0) << "open " << file.path << ": " << std::strerror(errno);
    }

    ~Fixture()
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }

    FileRef ref() const { return FileRef{ fd, false }; }

    // What the file holds at [offset, offset + bytesize).
    std::vector<char> expected_at(size_t offset, size_t bytesize) const
    {
        return std::vector<char>(data.begin() + offset, data.begin() + offset + bytesize);
    }

    std::vector<uint8_t> data;
    utils::temp::File file;
    int fd = -1;
};

// Reap until `count` completions have arrived, or the engine reports one too many rounds of nothing.
std::vector<Completion> reap(IoUringEngine & engine, unsigned count)
{
    std::vector<Completion> out;
    std::vector<Completion> batch(count);

    for (unsigned round = 0; round < count + 8 && out.size() < count; ++round)
    {
        unsigned got = 0;
        EXPECT_EQ(engine.wait_for_completions(batch.data(), batch.size(), got, WaitMode::Block, 2000),
                  common::ResponseCode::Success);
        out.insert(out.end(), batch.begin(), batch.begin() + got);
    }

    EXPECT_EQ(out.size(), count) << "only " << out.size() << " of " << count << " completions arrived";
    return out;
}

} // namespace

// The ring's real size, not the requested one. io_uring rounds entries up to a power of two, and the
// caller's in-flight window is sized from depth() - so reporting the request instead would leave slots
// unused, and reporting anything LARGER than the ring would let the window exceed the queue.
TEST(IoUringEngine, Depth_Is_The_Rings_Real_Size)
{
    SKIP_WITHOUT_RING();

    const IoUringEngine exact(config_with(512));
    EXPECT_EQ(exact.depth(), 512u);

    const IoUringEngine rounded(config_with(700));
    EXPECT_EQ(rounded.depth(), 1024u) << "700 entries rounds up to 1024";
    EXPECT_GE(rounded.depth(), 700u) << "the ring must never be smaller than the window";
}

// limits() had no test at all until this one, so the three values it reports were never checked.
//
// The alignments matter most. The caller tests congruence against them, and everything is congruent
// modulo 1, so reporting 1 would open every file with O_DIRECT and then fail every unaligned read with
// EINVAL. They also have to match what routing assumes before any engine exists, which is why both
// read one constant (see DirectBlockSize).
TEST(IoUringEngine, Limits_Describe_A_Direct_Read)
{
    SKIP_WITHOUT_RING();

    const IoUringEngine engine(config_with(8));
    const auto limits = engine.limits();

    EXPECT_EQ(limits.offset_alignment, direct_block_size());
    EXPECT_EQ(limits.buffer_alignment, direct_block_size());
    EXPECT_EQ(limits.max_read_bytesize, max_read_bytesize());
}

// One read, end to end: the bytes must be the file's bytes, at the right offset.
TEST(IoUringEngine, Reads_A_File)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(64 << 10);
    IoUringEngine engine(config_with(8));

    std::vector<char> buffer(4096);
    ASSERT_EQ(engine.stage(7, fixture.ref(), 8192, buffer.size(), buffer.data()), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 1u);

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(completions[0].id, 7u) << "user_data must come back as the id we staged under";
    EXPECT_FALSE(completions[0].failed());
    EXPECT_EQ(completions[0].bytes_transferred(), buffer.size());
    EXPECT_EQ(buffer, fixture.expected_at(8192, buffer.size()));
}

// The same read, through a REGISTERED buffer. Same bytes, or the optimisation is worse than useless.
//
// The registration is a plain page-aligned allocation rather than a staging pool's slab: what the
// engine needs is a region and an id, and it has no opinion about where they came from.
//
// Skipped where the kernel will not register, asked of the kernel rather than of IoUringProbe - a probe
// that wrongly said no would otherwise skip the only test that exercises this path.
TEST(IoUringEngine, Reads_Through_A_Registered_Buffer)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(64 << 10);
    IoUringEngine engine(config_with(8));

    // One region, several reads out of it at different offsets - the shape a slab of buffers has.
    constexpr size_t RegionBytes = 64 << 10;
    void * region = nullptr;
    ASSERT_EQ(::posix_memalign(&region, 4096, RegionBytes), 0);
    std::memset(region, 0, RegionBytes);

    Registration registration;
    registration.base = region;
    registration.bytesize = RegionBytes;
    registration.id = 0;

    constexpr size_t Read = 4096;
    for (unsigned i = 0; i < 3; ++i)
    {
        char * const into = static_cast<char *>(region) + static_cast<size_t>(i) * Read;

        ASSERT_EQ(engine.stage(100 + i, fixture.ref(), i * Read, Read, into, registration),
                  common::ResponseCode::Success);
    }

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    ASSERT_EQ(issued, 3u);

    // THE PATH UNDER TEST ACTUALLY RAN. A fixed read and an ordinary one return identical bytes, so
    // without this the test would pass on a host that registered nothing.
    EXPECT_EQ(engine.registered_regions(), ring_registers() ? 1u : 0u)
        << "the engine disagreed with the kernel about whether this region could be registered";

    const auto completions = reap(engine, 3);
    ASSERT_EQ(completions.size(), 3u);

    for (const auto & completion : completions)
    {
        ASSERT_FALSE(completion.failed())
            << "a registered read failed with " << completion.res
            << " - the engine must fall back to an ordinary read rather than fail one";
        EXPECT_EQ(completion.bytes_transferred(), Read);
    }

    // The bytes, which is the whole point: a wrong buffer index reads into memory that is not ours.
    for (unsigned i = 0; i < 3; ++i)
    {
        const char * const into = static_cast<const char *>(region) + static_cast<size_t>(i) * Read;
        const auto expected = fixture.expected_at(i * Read, Read);

        EXPECT_EQ(std::vector<char>(into, into + Read), expected) << "read " << i << " holds wrong bytes";
    }

    ::free(region);
}

// A pool grows WHILE READS ARE IN FLIGHT - that is the normal case, since it grows on demand under
// load. So a region offered then must still register: if the kernel refuses a live ring, the engine
// would mark that slab refused forever and never register it again.
TEST(IoUringEngine, Registers_A_Region_While_Reads_Are_In_Flight)
{
    SKIP_WITHOUT_RING();
    if (!ring_registers())
    {
        GTEST_SKIP() << "this kernel registers nothing, so there is no registration to time";
    }

    Fixture fixture(64 << 10);
    IoUringEngine engine(config_with(8));

    // One ordinary read, issued and deliberately NOT reaped: the ring is live from here.
    std::vector<char> plain(4096);
    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, plain.size(), plain.data()), common::ResponseCode::Success);
    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    ASSERT_EQ(issued, 1u);

    void * region = nullptr;
    ASSERT_EQ(::posix_memalign(&region, 4096, 8192), 0);
    std::memset(region, 0, 8192);

    Registration registration;
    registration.base = region;
    registration.bytesize = 8192;
    registration.id = 0;

    ASSERT_EQ(engine.stage(2, fixture.ref(), 0, 4096, static_cast<char *>(region), registration),
              common::ResponseCode::Success);

    EXPECT_EQ(engine.registered_regions(), 1u)
        << "a region offered while the ring was live was not registered";

    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, 2);
    ASSERT_EQ(completions.size(), 2u);
    for (const auto & completion : completions)
    {
        EXPECT_FALSE(completion.failed()) << "res " << completion.res;
    }

    ::free(region);
}

// The same, with the ring FULL rather than holding one read. Registration happens exactly when the
// staging pool grows, and a pool grows under load - so if a busy ring refuses registration, it refuses
// it in the only conditions that ever ask.
TEST(IoUringEngine, Registers_A_Region_With_A_Full_Ring)
{
    SKIP_WITHOUT_RING();
    if (!ring_registers())
    {
        GTEST_SKIP() << "this kernel registers nothing";
    }

    constexpr unsigned Depth = 32;
    Fixture fixture(1 << 20);
    IoUringEngine engine(config_with(Depth));

    std::vector<std::vector<char>> plain(Depth, std::vector<char>(4096));
    for (unsigned i = 0; i < Depth; ++i)
    {
        ASSERT_EQ(engine.stage(i, fixture.ref(), i * 4096, 4096, plain[i].data()),
                  common::ResponseCode::Success);
    }

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    ASSERT_EQ(issued, Depth) << "the ring must be full for this test to mean anything";

    // Not reaped: every one of those is still in flight.
    void * region = nullptr;
    ASSERT_EQ(::posix_memalign(&region, 4096, 8192), 0);
    std::memset(region, 0, 8192);

    Registration registration;
    registration.base = region;
    registration.bytesize = 8192;
    registration.id = 0;

    ASSERT_EQ(engine.stage(1000, fixture.ref(), 0, 4096, static_cast<char *>(region), registration),
              common::ResponseCode::Success);

    EXPECT_EQ(engine.registered_regions(), 1u)
        << "a full ring refused registration - the lazy-register-on-growth design would then pay a"
           " failed syscall on every read of that slab";

    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, Depth + 1);
    EXPECT_EQ(completions.size(), Depth + 1);

    ::free(region);
}

// BEST EFFORT, against a real kernel refusal rather than an argument.
//
// RLIMIT_MEMLOCK is the refusal production will actually meet - io_uring charges registration against
// it and CUDA pinning does not, so a host with a small limit registers nothing while the staging pool
// still allocates happily. The read must still be served, with the right bytes.
//
// The limit is lowered and restored inside this test: gtest runs every test in one process, so leaving
// it lowered would silently disable registration for whatever ran next.
TEST(IoUringEngine, A_Kernel_Refusal_Still_Reads)
{
    SKIP_WITHOUT_RING();

    struct rlimit original;
    ASSERT_EQ(::getrlimit(RLIMIT_MEMLOCK, &original), 0);

    struct rlimit tiny = original;
    tiny.rlim_cur = 4096;   // far under the region below
    if (::setrlimit(RLIMIT_MEMLOCK, &tiny) != 0)
    {
        GTEST_SKIP() << "cannot lower RLIMIT_MEMLOCK here, so a refusal cannot be provoked";
    }

    {
        Fixture fixture(64 << 10);
        IoUringEngine engine(config_with(8));

        constexpr size_t RegionBytes = 16 << 20;   // way past the limit just set
        void * region = nullptr;
        ASSERT_EQ(::posix_memalign(&region, 4096, RegionBytes), 0);
        std::memset(region, 0, 4096);

        Registration registration;
        registration.base = region;
        registration.bytesize = RegionBytes;
        registration.id = 0;

        ASSERT_EQ(engine.stage(31, fixture.ref(), 0, 4096, static_cast<char *>(region), registration),
                  common::ResponseCode::Success)
            << "a refused registration must not fail the staging";

        unsigned issued = 0;
        ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

        const auto completions = reap(engine, 1);
        ASSERT_EQ(completions.size(), 1u);
        EXPECT_FALSE(completions[0].failed())
            << "res " << completions[0].res << " - a refused registration must not fail the read";

        const auto expected = fixture.expected_at(0, 4096);
        EXPECT_EQ(std::vector<char>(static_cast<char *>(region), static_cast<char *>(region) + 4096),
                  expected)
            << "the fallback read delivered the wrong bytes";

        EXPECT_EQ(engine.registered_regions(), 0u) << "the kernel refused, so nothing is registered";

        ::free(region);
    }

    ASSERT_EQ(::setrlimit(RLIMIT_MEMLOCK, &original), 0) << "the limit must go back for later tests";
}

// A buffer OUTSIDE the region it was offered with is read the ordinary way, not failed.
//
// The kernel answers EFAULT for a fixed read whose buffer is not inside its registered region, so
// passing one through would turn a caller's bookkeeping slip into a failed read - which io_engine.h
// promises registration can never do.
TEST(IoUringEngine, A_Buffer_Outside_Its_Region_Still_Reads)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(64 << 10);
    IoUringEngine engine(config_with(8));

    void * region = nullptr;
    ASSERT_EQ(::posix_memalign(&region, 4096, 8192), 0);

    std::vector<char> elsewhere(4096);   // not in the region at all

    Registration registration;
    registration.base = region;
    registration.bytesize = 8192;
    registration.id = 0;

    ASSERT_EQ(engine.stage(21, fixture.ref(), 0, elsewhere.size(), elsewhere.data(), registration),
              common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_FALSE(completions[0].failed())
        << "res " << completions[0].res << " - a mismatched region must fall back, not fail";
    EXPECT_EQ(elsewhere, fixture.expected_at(0, elsewhere.size()));

    ::free(region);
}

// An id past the table is served the ordinary way. Registration is an optimisation, so running out of
// slots must cost speed and never a read.
TEST(IoUringEngine, An_Unregisterable_Region_Still_Reads)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(64 << 10);
    IoUringEngine engine(config_with(8));

    std::vector<char> buffer(4096);

    Registration registration;
    registration.base = buffer.data();
    registration.bytesize = buffer.size();
    registration.id = 1u << 20;   // far past any table this engine builds

    ASSERT_EQ(engine.stage(11, fixture.ref(), 0, buffer.size(), buffer.data(), registration),
              common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_FALSE(completions[0].failed());
    EXPECT_EQ(buffer, fixture.expected_at(0, buffer.size()));
    EXPECT_EQ(engine.registered_regions(), 0u) << "an id past the table must register nothing";
    EXPECT_EQ(engine.refused_regions(), 0u)
        << "an id the table cannot hold is not a refusal by the kernel, and must not be counted as one";
}

// Ids are echoed, not positional. Completions arrive in whatever order the kernel finishes them, so
// routing depends entirely on user_data surviving the round trip.
TEST(IoUringEngine, Ids_Survive_The_Round_Trip)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(256 << 10);
    IoUringEngine engine(config_with(16));

    // Ids well outside [0, depth) - they are opaque tokens, not slot indices, and the worker's are
    // monotonic and never reused.
    const std::vector<RequestId> ids{ 1000, 999999, 4294967296ULL };
    std::vector<std::vector<char>> buffers(ids.size(), std::vector<char>(4096));

    for (size_t i = 0; i < ids.size(); ++i)
    {
        ASSERT_EQ(engine.stage(ids[i], fixture.ref(), i * 4096, 4096, buffers[i].data()),
                  common::ResponseCode::Success);
    }

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, ids.size());

    const auto completions = reap(engine, ids.size());

    for (const auto & completion : completions)
    {
        const auto it = std::find(ids.begin(), ids.end(), completion.id);
        ASSERT_NE(it, ids.end()) << "completion for an id never staged: " << completion.id;

        const auto index = std::distance(ids.begin(), it);
        EXPECT_FALSE(completion.failed());
        EXPECT_EQ(buffers[index], fixture.expected_at(index * 4096, 4096))
            << "id " << completion.id << " wrote the wrong region";
    }
}

// Reading past the end returns FEWER bytes with no error. Treating "no error" as "all bytes arrived"
// is how a tensor gets silently truncated, so bytes_transferred is the value that matters.
TEST(IoUringEngine, Short_Read_At_Eof_Is_Not_An_Error)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(4096);
    IoUringEngine engine(config_with(8));

    std::vector<char> buffer(8192);   // asking for twice the file
    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, buffer.size(), buffer.data()), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_FALSE(completions[0].failed()) << "a short read is not a failure";
    EXPECT_EQ(completions[0].bytes_transferred(), 4096u) << "and it must say how much actually arrived";
}

// A bad fd fails that one request and nothing else. The error arrives as a completion, not as a
// staging or submission failure - which is why the caller must check every cqe->res.
TEST(IoUringEngine, Bad_Fd_Fails_Only_Its_Own_Request)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(16 << 10);
    IoUringEngine engine(config_with(8));

    std::vector<char> good(4096);
    std::vector<char> bad(4096);

    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, good.size(), good.data()), common::ResponseCode::Success);
    ASSERT_EQ(engine.stage(2, FileRef{ -1, false }, 0, bad.size(), bad.data()), common::ResponseCode::Success)
        << "staging cannot know the fd is bad; only the completion can";

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, 2);
    ASSERT_EQ(completions.size(), 2u);

    for (const auto & completion : completions)
    {
        if (completion.id == 1)
        {
            EXPECT_FALSE(completion.failed());
            EXPECT_EQ(good, fixture.expected_at(0, good.size()));
        }
        else
        {
            // The exact errno, now that the engine passes the kernel's result through unchanged.
            // This is what the caller maps, so it is the value worth pinning.
            EXPECT_EQ(completion.res, -EBADF) << "a closed fd must not read as success";
        }
    }
}

// Staging is not issuing. flush() is the only thing that hands work to the kernel, and a worker that
// waits without having flushed waits for a completion that cannot come.
TEST(IoUringEngine, Staging_Does_Not_Issue)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(16 << 10);
    IoUringEngine engine(config_with(8));

    std::vector<char> buffer(4096);
    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, buffer.size(), buffer.data()), common::ResponseCode::Success);

    // Nothing submitted yet, so nothing can complete.
    Completion completions[4];
    unsigned count = 0;
    ASSERT_EQ(engine.wait_for_completions(completions, 4, count, WaitMode::NonBlocking),
              common::ResponseCode::Success);
    EXPECT_EQ(count, 0u) << "a staged request completed without ever being submitted";

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 1u);
    EXPECT_EQ(reap(engine, 1).size(), 1u);
}

// An expired wait is Success with nothing harvested, never an error - it is also the teardown
// wake-up, since no other thread may touch the engine.
TEST(IoUringEngine, Expired_Wait_Is_Not_An_Error)
{
    SKIP_WITHOUT_RING();

    IoUringEngine engine(config_with(8));   // nothing staged, nothing issued

    Completion completions[4];
    unsigned count = 0;
    EXPECT_EQ(engine.wait_for_completions(completions, 4, count, WaitMode::Block, 50),
              common::ResponseCode::Success);
    EXPECT_EQ(count, 0u);
}

// flush() with nothing staged is a no-op, not a syscall and not an error.
TEST(IoUringEngine, Flush_With_Nothing_Staged)
{
    SKIP_WITHOUT_RING();

    IoUringEngine engine(config_with(8));

    unsigned issued = 7;   // must be overwritten
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 0u);
}

// The whole window at once: depth requests staged, one flush, all of them complete and land where
// they belong. This is the shape a real workload takes.
TEST(IoUringEngine, Fills_The_Window)
{
    SKIP_WITHOUT_RING();

    constexpr unsigned Depth = 32;
    constexpr size_t Bytes = 4096;

    Fixture fixture(Depth * Bytes);
    IoUringEngine engine(config_with(Depth));
    ASSERT_EQ(engine.depth(), Depth);

    std::vector<std::vector<char>> buffers(Depth, std::vector<char>(Bytes));
    for (unsigned i = 0; i < Depth; ++i)
    {
        ASSERT_EQ(engine.stage(i + 1, fixture.ref(), i * Bytes, Bytes, buffers[i].data()),
                  common::ResponseCode::Success)
            << "staged " << i << " of " << Depth << " - the ring must hold a full window";
    }

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, Depth);

    const auto completions = reap(engine, Depth);
    ASSERT_EQ(completions.size(), Depth);

    for (const auto & completion : completions)
    {
        ASSERT_FALSE(completion.failed());
        const auto index = completion.id - 1;
        EXPECT_EQ(buffers[index], fixture.expected_at(index * Bytes, Bytes));
    }
}

// ---- O_DIRECT, against the real kernel ----
//
// Everything else that tests direct reads uses MockIoEngine, which now applies the alignment rule
// itself. These tests exist to check that the rule the mock applies is the rule the kernel applies.
// Without them the mock could enforce something we invented, and every test would agree with it.

namespace
{

// Tied to the routing constant, not written out again. Routing decides congruence before any engine
// exists, so it carries its own block size; if the two drift apart, routing and the worker disagree
// about which files can be read directly and nothing fails - the reads just take a path nobody
// intended.
// Follows the engine rather than the default - see the same note in libaio_engine_test.
const size_t Block = direct_block_size();

// Like Fixture, but the fd is opened with O_DIRECT.
//
// Some filesystems have no O_DIRECT at all. `supported` says whether this one accepted the open, so a
// test can skip instead of reporting a mount limit as a failure.
struct DirectFixture
{
    explicit DirectFixture(size_t bytesize) :
        data(utils::random::buffer(bytesize)),
        file(data),
        fd(::open(file.path.c_str(), O_RDONLY | O_DIRECT))
    {
        // Kept, rather than read from errno later. errno belongs to the last failed call anywhere, so
        // by the time the test reports why it stopped, something else may have overwritten it.
        if (fd < 0)
        {
            error = errno;
        }
    }

    ~DirectFixture()
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }

    bool supported() const { return fd >= 0; }

    FileRef ref() const { return FileRef{ fd, true }; }

    std::vector<char> expected_at(size_t offset, size_t bytesize) const
    {
        return std::vector<char>(data.begin() + offset, data.begin() + offset + bytesize);
    }

    std::vector<uint8_t> data;
    utils::temp::File file;
    int fd = -1;
    int error = 0;   // the errno from the open, when it failed
};

// A buffer whose address is a multiple of Block. Tests that want an unaligned address add 1 to it.
struct AlignedBuffer
{
    explicit AlignedBuffer(size_t bytesize)
    {
        void * raw = nullptr;
        if (::posix_memalign(&raw, Block, bytesize) != 0)
        {
            raw = nullptr;
        }
        _memory = static_cast<char *>(raw);
    }

    ~AlignedBuffer() { ::free(_memory); }

    char * get() { return _memory; }

 private:
    char * _memory = nullptr;
};

// Stage one read and reap its single completion.
Completion read_once(IoUringEngine & engine, FileRef file, size_t offset, size_t bytesize, char * buffer)
{
    const auto staged = engine.stage(1, file, offset, bytesize, buffer);
    EXPECT_EQ(staged, common::ResponseCode::Success) << "the engine prepares the SQE; the kernel judges it";

    unsigned issued = 0;
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 1u);

    const auto completions = reap(engine, 1);
    return completions.empty() ? Completion{} : completions.front();
}

// The SAME gate as SKIP_WITHOUT_RING, for the second thing that can make these tests disappear.
//
// A filesystem can refuse O_DIRECT even where io_uring works, and only these two tests check our
// alignment rule against the KERNEL - everything else checks it against MockIoEngine, which enforces
// the rule we believe in. So when they skip, that rule is unverified and nothing says so.
//
// RUNAI_STREAMER_REQUIRE_IO_URING means "this host is set up for the async backend", which covers
// O_DIRECT as well as the ring. A host prepared for benchmarking has a filesystem that supports it,
// so a refusal there is a setup problem worth stopping for rather than a reason to test less.
#define SKIP_WITHOUT_O_DIRECT(fixture)                                                  \
    do {                                                                                \
        if (!(fixture).supported())                                                     \
        {                                                                               \
            if (require_io_uring())                                                     \
            {                                                                           \
                FAIL() << "this filesystem refused O_DIRECT ("                           \
                       << std::strerror((fixture).error) << ") but "                     \
                       << "RUNAI_STREAMER_REQUIRE_IO_URING=1 says this host is set up "  \
                       << "for direct reads - the alignment rule would go unchecked";    \
            }                                                                           \
            GTEST_SKIP() << "this filesystem refused O_DIRECT ("                        \
                         << std::strerror((fixture).error) << ")";                      \
        }                                                                               \
    } while (0)

} // namespace

// A direct read that follows the rule returns the file's bytes, like any other read.
TEST(IoUringEngine, Direct_Read_Returns_The_Right_Bytes)
{
    SKIP_WITHOUT_RING();

    DirectFixture fixture(Block * 8);
    SKIP_WITHOUT_O_DIRECT(fixture);

    IoUringEngine engine(config_with(8));
    AlignedBuffer buffer(Block * 2);
    ASSERT_NE(buffer.get(), nullptr);

    // Offset, length and address are all multiples of the block size.
    const auto completion = read_once(engine, fixture.ref(), Block * 3, Block * 2, buffer.get());

    ASSERT_FALSE(completion.failed());
    ASSERT_EQ(completion.bytes_transferred(), Block * 2);

    EXPECT_EQ(std::vector<char>(buffer.get(), buffer.get() + Block * 2),
              fixture.expected_at(Block * 3, Block * 2));
}

// THE TEST THAT MAKES THE ONE ABOVE MEAN SOMETHING.
//
// Some mounts accept O_DIRECT and then ignore it. On such a mount the aligned test would pass while
// testing nothing. This checks that the kernel really does refuse a read that breaks the rule, so we
// know O_DIRECT is being applied here.
//
// It also checks the rule the mock now enforces is the real one, for each of the three values.
TEST(IoUringEngine, Direct_Read_Is_Refused_When_Misaligned)
{
    SKIP_WITHOUT_RING();

    DirectFixture fixture(Block * 8);
    SKIP_WITHOUT_O_DIRECT(fixture);

    AlignedBuffer buffer(Block * 3);
    ASSERT_NE(buffer.get(), nullptr);

    struct Case
    {
        const char * what;
        size_t offset;
        size_t bytesize;
        size_t address_shift;
    };

    const Case cases[] = {
        { "file offset",    Block * 3 + 1, Block,     0 },
        { "length",         Block * 3,     Block - 8, 0 },
        { "buffer address", Block * 3,     Block,     1 },
    };

    for (const auto & test_case : cases)
    {
        IoUringEngine engine(config_with(8));

        const auto completion = read_once(engine, fixture.ref(), test_case.offset, test_case.bytesize,
                                          buffer.get() + test_case.address_shift);

        // EINVAL exactly. This is the value completion_mapper reads to tell an alignment bug from a
        // storage fault, so the test pins the number and not only "it failed".
        EXPECT_EQ(completion.res, -EINVAL)
            << "the kernel accepted a direct read with a misaligned " << test_case.what
            << " - either this mount ignores O_DIRECT, or the rule the mock enforces is wrong";
    }
}

// Both engines report what submitting cost, in the same shape, so the two can be compared.
//
// This exists because io_uring's submit is DESCRIBED as a ring append rather than work done inline,
// while libaio's io_submit is known to block. That difference decides whether batching many reads
// into one call is free or expensive - and it is a claim about the kernel, not something to take on
// faith. Measuring only libaio would mean tuning both engines on one engine's evidence.
//
// A counter that is never filled would answer that question with a silent zero, which is why the
// nanos are asserted non-zero rather than merely present.
TEST(IoUringEngine, Submit_Time_Is_Measured)
{
    SKIP_WITHOUT_RING();

    Fixture fixture(64 << 10);
    IoUringEngine engine(config_with(8));

    std::vector<char> buffer(4096);
    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, buffer.size(), buffer.data()), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    reap(engine, 1);

    const auto stats = engine.submit_stats();
    EXPECT_EQ(stats.calls, 1u);
    EXPECT_EQ(stats.requests, 1u);
    EXPECT_GT(stats.nanos, 0u);
    EXPECT_GT(stats.max_nanos, 0u);
    EXPECT_LE(stats.max_nanos, stats.nanos);
}

// An empty flush must not be counted. Otherwise the average batch size - requests / calls - is
// diluted by calls that carried nothing, and that ratio is the number the batching question turns on.
TEST(IoUringEngine, An_Empty_Flush_Is_Not_Counted)
{
    SKIP_WITHOUT_RING();

    IoUringEngine engine(config_with(8));

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 0u);

    EXPECT_EQ(engine.submit_stats().calls, 0u)
        << "an empty flush should not reach io_uring_submit at all";
}

}; // namespace runai::llm::streamer::posix_io
