#include "device/cuda/cuda_lib.h"

#include <gtest/gtest.h>

namespace runai::llm::streamer::device::cuda
{

namespace
{

// A code the mapping does not name, used to check that the call site's choice is what comes back.
constexpr CUresult unnamed = CUDA_ERROR_LAUNCH_FAILED;

} // namespace

TEST(ToResponseCode, Success_Ignores_The_Fallback)
{
    EXPECT_EQ(to_response_code(CUDA_SUCCESS, common::ResponseCode::DeviceTransferError), common::ResponseCode::Success);
}

// Each of these has its own remedy, which is the reason it is a separate code.
TEST(ToResponseCode, Classified_Codes)
{
    const auto fallback = common::ResponseCode::UnknownError;

    EXPECT_EQ(to_response_code(CUDA_ERROR_NO_DEVICE, fallback), common::ResponseCode::DeviceUnavailable);
    EXPECT_EQ(to_response_code(CUDA_ERROR_INVALID_DEVICE, fallback), common::ResponseCode::InvalidDevice);
    EXPECT_EQ(to_response_code(CUDA_ERROR_OUT_OF_MEMORY, fallback), common::ResponseCode::DeviceOutOfMemory);
    EXPECT_EQ(to_response_code(CUDA_ERROR_NOT_INITIALIZED, fallback), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(to_response_code(CUDA_ERROR_DEINITIALIZED, fallback), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(to_response_code(CUDA_ERROR_INVALID_CONTEXT, fallback), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(to_response_code(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, fallback), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(to_response_code(CUDA_ERROR_INVALID_VALUE, fallback), common::ResponseCode::InvalidParameterError);
}

// A classified code must not be overridden by the call site: a device that ran out of memory says so
// whether the caller was copying or allocating.
TEST(ToResponseCode, Classified_Codes_Beat_The_Fallback)
{
    EXPECT_EQ(to_response_code(CUDA_ERROR_OUT_OF_MEMORY, common::ResponseCode::DeviceTransferError), common::ResponseCode::DeviceOutOfMemory);
    EXPECT_EQ(to_response_code(CUDA_ERROR_NO_DEVICE, common::ResponseCode::DeviceTransferError), common::ResponseCode::DeviceUnavailable);
}

// The same unnamed result means different things depending on what was attempted, which is why the
// fallback is a parameter rather than a constant.
TEST(ToResponseCode, Unclassified_Code_Takes_The_Call_Sites_Choice)
{
    EXPECT_EQ(to_response_code(unnamed, common::ResponseCode::DeviceTransferError), common::ResponseCode::DeviceTransferError);
    EXPECT_EQ(to_response_code(unnamed, common::ResponseCode::DeviceDriverError), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(to_response_code(unnamed, common::ResponseCode::DeviceOutOfMemory), common::ResponseCode::DeviceOutOfMemory);
}

// UnknownError tells a caller to abort and distrust every response it already holds. Nothing the
// device path reports should mean that, so no mapping may produce it on its own.
TEST(ToResponseCode, Never_Invents_Unknown_Error)
{
    const int first = static_cast<int>(CUDA_SUCCESS);
    const int last = static_cast<int>(CUDA_ERROR_UNKNOWN);

    for (int value = first; value <= last; ++value)
    {
        const auto code = to_response_code(static_cast<CUresult>(value), common::ResponseCode::DeviceTransferError);
        EXPECT_NE(code, common::ResponseCode::UnknownError) << "CUresult " << value << " mapped to UnknownError";
    }
}

TEST(Report, Success_Is_Not_An_Error)
{
    CudaLib lib = {};
    EXPECT_EQ(lib.report(CUDA_SUCCESS, "test", common::ResponseCode::DeviceTransferError), common::ResponseCode::Success);
}

// report() runs on a failure path where the driver may be half-resolved, so it must not depend on
// the text functions being present.
TEST(Report, Works_Without_The_Error_Text_Symbols)
{
    CudaLib lib = {};
    EXPECT_EQ(lib.report(CUDA_ERROR_OUT_OF_MEMORY, "test", common::ResponseCode::DeviceTransferError), common::ResponseCode::DeviceOutOfMemory);
    EXPECT_STREQ(lib.error_text(CUDA_ERROR_OUT_OF_MEMORY), "unknown CUDA error");
}

} // namespace runai::llm::streamer::device::cuda
