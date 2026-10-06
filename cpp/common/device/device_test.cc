/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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

// A Device keys the channel map, so it needs an ordering that agrees with operator==.
TEST(Device, Orders_By_Type_Then_Ordinal)
{
    EXPECT_TRUE(Device::host() < Device::cuda(0));
    EXPECT_FALSE(Device::cuda(0) < Device::host());

    EXPECT_TRUE(Device::cuda(0) < Device::cuda(1));
    EXPECT_FALSE(Device::cuda(1) < Device::cuda(0));
    EXPECT_FALSE(Device::cuda(1) < Device::cuda(1));
}

// Two hosts are ONE key. Built differently they must still not order against each other, or a map
// would hold two entries for the same device.
TEST(Device, Hosts_Never_Order_Against_Each_Other)
{
    const auto other = Device{ DeviceType::Cpu, 7 };

    EXPECT_EQ(other, Device::host());
    EXPECT_FALSE(other < Device::host());
    EXPECT_FALSE(Device::host() < other);
}

} // namespace runai::llm::streamer::common
