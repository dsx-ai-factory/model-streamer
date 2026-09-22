#pragma once

#include <ostream>

#include "streamer/device.h"

namespace runai::llm::streamer::common
{

// The C++ name for each published device type. Every value comes from <streamer/device.h>, so the
// two cannot drift: a type is added there first, and named here.
enum class DeviceType : int
{
    Cpu  = RUNAI_FILE_STREAMER_DEVICE_CPU,
    Cuda = RUNAI_FILE_STREAMER_DEVICE_CUDA,
};

// Where a submission's destinations live. ONE PER SUBMISSION, so every batch of it carries the same
// value - a load spanning several GPUs submits once per GPU.
//
// The ordinal is UNSIGNED here and signed in the C struct, which types it as CUdevice does. The C
// layer rejects a negative one, so an impl-side Device cannot hold an ordinal that was never valid.
struct Device
{
    DeviceType type = DeviceType::Cpu;   // a zeroed Device is the host, as a zeroed C struct is
    unsigned   id = 0;                   // meaningless for Cpu

    static Device host() { return Device{}; }
    static Device cuda(unsigned ordinal) { return Device{ DeviceType::Cuda, ordinal }; }

    bool is_host() const { return type == DeviceType::Cpu; }
};

bool operator==(const Device & left, const Device & right);
bool operator!=(const Device & left, const Device & right);

std::ostream & operator<<(std::ostream & os, DeviceType type);
std::ostream & operator<<(std::ostream & os, const Device & device);

} // namespace runai::llm::streamer::common
