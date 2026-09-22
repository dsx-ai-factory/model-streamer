#include "common/device/device.h"

#include <gtest/gtest.h>

#include <sstream>

namespace runai::llm::streamer::common
{

// A zeroed struct asks for the host on both sides of the boundary, which is what lets a caller that
// never heard of devices leave the field alone.
TEST(Device, A_Default_Device_Is_The_Host)
{
    const Device device;

    EXPECT_TRUE(device.is_host());
    EXPECT_EQ(device, Device::host());
    EXPECT_EQ(static_cast<int>(device.type), RUNAI_FILE_STREAMER_DEVICE_CPU);
}

TEST(Device, Cuda_Carries_Its_Ordinal)
{
    const auto device = Device::cuda(3);

    EXPECT_FALSE(device.is_host());
    EXPECT_EQ(device.id, 3u);
    EXPECT_EQ(static_cast<int>(device.type), RUNAI_FILE_STREAMER_DEVICE_CUDA);
    EXPECT_NE(device, Device::cuda(4));
    EXPECT_NE(device, Device::host());
}

// The ordinal means nothing for the host, so two host devices are the same however they were built.
TEST(Device, The_Host_Ordinal_Is_Not_Compared)
{
    Device odd;
    odd.id = 7;

    EXPECT_EQ(odd, Device::host());
}

TEST(Device, Printing)
{
    auto text = [](const Device & device)
    {
        std::ostringstream os;
        os << device;
        return os.str();
    };

    EXPECT_EQ(text(Device::host()), "cpu");
    EXPECT_EQ(text(Device::cuda(2)), "cuda:2");
}

} // namespace runai::llm::streamer::common
