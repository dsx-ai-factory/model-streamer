#include "device/backends/backends.h"

#include "device/cuda/cuda_device.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::device
{

BackendFactory backend_for(common::DeviceType type)
{
    switch (type)
    {
    case common::DeviceType::Cuda:
        return cuda::backend;

    case common::DeviceType::Cpu:
        return BackendFactory();   // the host is not a backend
    }

    // A type published in the C header and not handled above. The build sets neither -Wall nor
    // -Werror, so the compiler says nothing about it - and the quiet answer here would be
    // DeviceUnavailable on a machine where the device is present and working.
    LOG(ERROR) << "[RunAI Streamer] no backend is built for device type " << static_cast<int>(type);
    return BackendFactory();
}

} // namespace runai::llm::streamer::device
