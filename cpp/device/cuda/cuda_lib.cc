#include "device/cuda/cuda_lib.h"

#include <dlfcn.h>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::device::cuda
{

namespace
{

// Resolves a symbol, preferring the versioned name. CUDA 4 widened CUdeviceptr to 64 bits and gave
// the affected entry points a _v2 suffix; the plain name still exists on older drivers.
template <typename T>
bool resolve(void * handle, T & fn, const char * versioned, const char * plain = nullptr)
{
    fn = reinterpret_cast<T>(dlsym(handle, versioned));
    if (fn == nullptr && plain != nullptr)
    {
        fn = reinterpret_cast<T>(dlsym(handle, plain));
    }
    if (fn == nullptr)
    {
        LOG(WARNING) << "[RunAI Streamer] CUDA symbol not found: " << versioned;
    }
    return fn != nullptr;
}

bool load_symbols(void * handle, CudaLib & lib)
{
    bool ok = true;

    ok &= resolve(handle, lib.cuInit,                    "cuInit");
    ok &= resolve(handle, lib.cuDriverGetVersion,        "cuDriverGetVersion");
    ok &= resolve(handle, lib.cuGetErrorName,            "cuGetErrorName");
    ok &= resolve(handle, lib.cuGetErrorString,          "cuGetErrorString");

    ok &= resolve(handle, lib.cuDeviceGetCount,          "cuDeviceGetCount");
    ok &= resolve(handle, lib.cuDeviceGet,               "cuDeviceGet");
    ok &= resolve(handle, lib.cuDeviceGetAttribute,      "cuDeviceGetAttribute");
    ok &= resolve(handle, lib.cuDevicePrimaryCtxRetain,  "cuDevicePrimaryCtxRetain");
    ok &= resolve(handle, lib.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
    ok &= resolve(handle, lib.cuCtxSetCurrent,           "cuCtxSetCurrent");
    ok &= resolve(handle, lib.cuCtxGetCurrent,           "cuCtxGetCurrent");
    ok &= resolve(handle, lib.cuMemGetInfo,              "cuMemGetInfo_v2", "cuMemGetInfo");

    ok &= resolve(handle, lib.cuMemHostAlloc,            "cuMemHostAlloc");
    ok &= resolve(handle, lib.cuMemFreeHost,             "cuMemFreeHost");
    ok &= resolve(handle, lib.cuMemHostRegister,         "cuMemHostRegister_v2", "cuMemHostRegister");
    ok &= resolve(handle, lib.cuMemHostUnregister,       "cuMemHostUnregister");
    ok &= resolve(handle, lib.cuMemAlloc,                "cuMemAlloc_v2", "cuMemAlloc");
    ok &= resolve(handle, lib.cuMemFree,                 "cuMemFree_v2", "cuMemFree");

    ok &= resolve(handle, lib.cuStreamCreate,            "cuStreamCreate");
    ok &= resolve(handle, lib.cuStreamDestroy,           "cuStreamDestroy_v2", "cuStreamDestroy");
    ok &= resolve(handle, lib.cuStreamSynchronize,       "cuStreamSynchronize");
    ok &= resolve(handle, lib.cuStreamQuery,             "cuStreamQuery");
    ok &= resolve(handle, lib.cuStreamWaitEvent,         "cuStreamWaitEvent");

    ok &= resolve(handle, lib.cuEventCreate,             "cuEventCreate");
    ok &= resolve(handle, lib.cuEventDestroy,            "cuEventDestroy_v2", "cuEventDestroy");
    ok &= resolve(handle, lib.cuEventRecord,             "cuEventRecord");
    ok &= resolve(handle, lib.cuEventQuery,              "cuEventQuery");
    ok &= resolve(handle, lib.cuEventSynchronize,        "cuEventSynchronize");

    // The _ptsz variants bind the per-thread default stream. Not wanted: every call here passes an
    // explicit stream, so the legacy names keep the meaning the signatures describe.
    ok &= resolve(handle, lib.cuMemcpyHtoDAsync,         "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    ok &= resolve(handle, lib.cuMemcpyDtoDAsync,         "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync");
    ok &= resolve(handle, lib.cuMemsetD8Async,           "cuMemsetD8Async");

    return ok;
}

CudaLib load()
{
    CudaLib lib = {};

    // The SONAME first. The unversioned name is a toolkit symlink, and on a machine that has only
    // the toolkit it points at a stub that resolves everything and then reports no device.
    void * handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr)
    {
        handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    }

    if (handle == nullptr)
    {
        LOG(INFO) << "[RunAI Streamer] CUDA driver library not found; CUDA streaming is unavailable";
        return {};
    }

    if (!load_symbols(handle, lib))
    {
        LOG(WARNING) << "[RunAI Streamer] CUDA driver is missing symbols the streamer needs; CUDA streaming is unavailable";
        return {};
    }

    // Nothing else in the driver API may be called first. A framework in the same process has
    // usually done this already, and a second call is harmless.
    const CUresult result = lib.cuInit(0);
    if (result != CUDA_SUCCESS)
    {
        LOG(INFO) << "[RunAI Streamer] cuInit failed (" << lib.error_text(result) << "); CUDA streaming is unavailable";
        return {};
    }

    int version = 0;
    if (lib.cuDriverGetVersion(&version) == CUDA_SUCCESS)
    {
        LOG(INFO) << "[RunAI Streamer] CUDA driver loaded, version " << version;
    }

    return lib;
}

} // namespace

const CudaLib * CudaLib::get()
{
    static const CudaLib lib = load();
    return lib.cuInit != nullptr ? &lib : nullptr;
}

const char * CudaLib::error_text(CUresult result) const
{
    const char * text = nullptr;
    if (cuGetErrorString != nullptr && cuGetErrorString(result, &text) == CUDA_SUCCESS && text != nullptr)
    {
        return text;
    }
    return "unknown CUDA error";
}

common::ResponseCode to_response_code(CUresult result, common::ResponseCode unclassified)
{
    switch (result)
    {
        case CUDA_SUCCESS:
            return common::ResponseCode::Success;

        case CUDA_ERROR_NO_DEVICE:
            return common::ResponseCode::DeviceUnavailable;

        case CUDA_ERROR_INVALID_DEVICE:
        case CUDA_ERROR_DEVICE_NOT_LICENSED:
            return common::ResponseCode::InvalidDevice;

        case CUDA_ERROR_OUT_OF_MEMORY:
            return common::ResponseCode::DeviceOutOfMemory;

        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
        case CUDA_ERROR_SYSTEM_NOT_READY:
            return common::ResponseCode::DeviceDriverError;

        // A malformed argument of ours, not a device problem.
        case CUDA_ERROR_INVALID_VALUE:
        case CUDA_ERROR_INVALID_HANDLE:
            return common::ResponseCode::InvalidParameterError;

        default:
            return unclassified;
    }
}

common::ResponseCode CudaLib::report(CUresult result, const char * operation, common::ResponseCode unclassified) const
{
    if (result == CUDA_SUCCESS)
    {
        return common::ResponseCode::Success;
    }

    const char * name = nullptr;
    if (cuGetErrorName == nullptr || cuGetErrorName(result, &name) != CUDA_SUCCESS || name == nullptr)
    {
        name = "unknown";
    }

    LOG(ERROR) << "[RunAI Streamer] " << operation << " failed: " << name
               << " (" << static_cast<int>(result) << ") - " << error_text(result);

    return to_response_code(result, unclassified);
}

} // namespace runai::llm::streamer::device::cuda
