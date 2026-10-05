#include "device/backends/backends.h"

#include <gtest/gtest.h>

namespace runai::llm::streamer::device
{

// The factory, not the backend: calling it needs a driver, and this must pass on CPU CI.
TEST(Backends, Cuda_Has_A_Backend)
{
    EXPECT_TRUE(static_cast<bool>(backend_for(common::DeviceType::Cuda)));
}

// The host is not a backend. An empty factory is what DeviceWriter reports as DeviceUnavailable, so
// returning one here is the whole answer for a type with no implementation.
TEST(Backends, The_Host_Has_No_Backend)
{
    EXPECT_FALSE(static_cast<bool>(backend_for(common::DeviceType::Cpu)));
}

// A type the C header publishes and this lookup was never taught. Nothing in the build warns about
// it (no -Wall), so the only protection is that it answers empty rather than crashing - and says so.
TEST(Backends, An_Unknown_Type_Has_No_Backend)
{
    const auto unknown = static_cast<common::DeviceType>(4242);

    EXPECT_FALSE(static_cast<bool>(backend_for(unknown)));
}

} // namespace runai::llm::streamer::device
