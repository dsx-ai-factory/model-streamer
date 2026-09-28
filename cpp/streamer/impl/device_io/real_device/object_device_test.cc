#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda.h>

#include "device/cuda/cuda_device.h"
#include "device/cuda/cuda_lib.h"
#include "streamer/streamer.h"

namespace runai::llm::streamer
{

// REAL OBJECT STORAGE to a REAL DEVICE, through the public C API.
//
// The mock plugin can show that the copy path works - context, pinning, events, the copy itself - but
// never that the right bytes reach the right offsets, because it does not write into the destination
// at all. Only a real server can answer that, and the emulators the Python suites already use are
// real servers: minio, azurite and fake-gcs-server speak the same protocols as the services.
//
// ONE TEST FOR ALL THREE. The streamer routes on the URI scheme, so what changes between providers is
// the URI and the credentials in the environment - which is exactly what the Makefile targets set up.
//
// SKIPS unless both halves are present: a usable CUDA device, and an object to read. That keeps it out
// of the way of `bazel test //...`, which runs with neither.
//
//   RUNAI_TEST_OBJECT_URI    s3://bucket/key, gs://bucket/key or azure://container/blob
//   RUNAI_TEST_OBJECT_FILE   the same bytes on disk, which is what the result is compared against
class ObjectStorageDevice : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        const char * const uri = ::getenv("RUNAI_TEST_OBJECT_URI");
        const char * const file = ::getenv("RUNAI_TEST_OBJECT_FILE");

        if (uri == nullptr || file == nullptr)
        {
            GTEST_SKIP() << "RUNAI_TEST_OBJECT_URI and RUNAI_TEST_OBJECT_FILE are unset, so there is"
                         << " no object storage to read from - see tests/Makefile";
        }

        _uri = uri;
        _expected = read_file(file);
        ASSERT_FALSE(_expected.empty()) << "the fixture at " << file << " is empty or unreadable";

        if (device::cuda::CudaLib::get() == nullptr)
        {
            GTEST_SKIP() << "no CUDA driver on this host";
        }

        _backend = device::cuda::backend();
        ASSERT_NE(_backend, nullptr);

        unsigned count = 0;
        if (_backend->device_count(count) != common::ResponseCode::Success || count == 0)
        {
            GTEST_SKIP() << "a CUDA driver, but no device this process can use";
        }

        if (_backend->open_device(0, _device) != common::ResponseCode::Success ||
            _device->bind_thread() != common::ResponseCode::Success)
        {
            GTEST_SKIP() << "device 0 could not be opened here";
        }
    }

    static std::vector<char> read_file(const char * path)
    {
        std::ifstream in(path, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

    // Device to host, for the comparison. The Device interface has no read-back - nothing in the
    // streamer needs one - so this reaches the driver directly, as the other real-device tests do.
    std::vector<char> read_back(void * source, size_t bytesize) const
    {
        std::vector<char> out(bytesize);

        void * const handle = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        EXPECT_NE(handle, nullptr);

        const auto copy = reinterpret_cast<CUresult (*)(void *, CUdeviceptr, size_t)>(
            ::dlsym(handle, "cuMemcpyDtoH_v2"));
        EXPECT_NE(copy, nullptr);

        EXPECT_EQ(copy(out.data(), reinterpret_cast<CUdeviceptr>(source), bytesize), CUDA_SUCCESS);
        return out;
    }

    std::string _uri;
    std::vector<char> _expected;
    std::shared_ptr<device::Backend> _backend;
    std::shared_ptr<device::Device> _device;
};

// The whole point: bytes from a real bucket, landing on a real device, byte for byte.
TEST_F(ObjectStorageDevice, Reads_An_Object_Into_Device_Memory)
{
    const size_t total = _expected.size();

    // SEVERAL RANGES, not one. A single range would be read as one chunk and copied once, which is the
    // easy case; several make the reader cut, stage and copy in pieces, and a piece landing at the
    // wrong offset is exactly what a byte check is for.
    constexpr unsigned Ranges = 4;
    ASSERT_GE(total, Ranges) << "the fixture is too small to cut";

    const size_t each = total / Ranges;

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(total, &target), common::ResponseCode::Success);

    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    const char * path = _uri.c_str();
    unsigned num_ranges = Ranges;

    std::vector<size_t> offsets(Ranges);
    std::vector<size_t> sizes(Ranges, each);
    std::vector<void *> dsts(Ranges);
    for (unsigned i = 0; i < Ranges; ++i)
    {
        offsets[i] = i * each;
        dsts[i] = static_cast<char *>(target) + offsets[i];
    }
    sizes[Ranges - 1] = total - offsets[Ranges - 1];   // the remainder, so the whole object is covered

    RunaiFileStreamerDevice device;
    device.type = RUNAI_FILE_STREAMER_DEVICE_CUDA;
    device.id = 0;

    RunaiFileStreamerSubmissionId submission_id = 0;
    ASSERT_EQ(runai_file_streamer_request(streamer, &submission_id, 1, &path, &num_ranges,
                                          offsets.data(), sizes.data(), dsts.data(), device),
              static_cast<int>(common::ResponseCode::Success));

    for (unsigned i = 0; i < Ranges; ++i)
    {
        RunaiFileStreamerSubmissionId answered = 0;
        unsigned file_index = 0;
        unsigned index = 0;
        int submission_done = 0;
        EXPECT_EQ(runai_file_streamer_response(streamer, &answered, &file_index, &index,
                                               &submission_done, 120000 /* ms */),
                  static_cast<int>(common::ResponseCode::Success)) << "range " << i;
    }

    runai_file_streamer_end(streamer);

    const auto landed = read_back(target, total);
    EXPECT_EQ(std::memcmp(landed.data(), _expected.data(), total), 0)
        << "the bytes on the device are not the bytes in the object";

    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);
}

} // namespace runai::llm::streamer
