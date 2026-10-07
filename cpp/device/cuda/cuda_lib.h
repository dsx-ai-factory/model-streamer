#pragma once

#include <cuda.h>

#include "common/response_code/response_code.h"

namespace runai::llm::streamer::device::cuda
{

// The subset of the CUDA driver API the streamer uses, resolved from libcuda.so.1 at run time.
//
// cuda.h is included for the official signatures, so the compiler checks these against the real
// ABI. It is not linked: a build with no driver on the host still runs, and reports no device.
struct CudaLib
{
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDriverGetVersion)(int *);
    CUresult (*cuGetErrorName)(CUresult, const char **);
    CUresult (*cuGetErrorString)(CUresult, const char **);

    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxGetCurrent)(CUcontext *);
    CUresult (*cuMemGetInfo)(size_t *, size_t *);

    CUresult (*cuMemHostAlloc)(void **, size_t, unsigned int);
    CUresult (*cuMemFreeHost)(void *);
    CUresult (*cuMemHostRegister)(void *, size_t, unsigned int);
    CUresult (*cuMemHostUnregister)(void *);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);

    CUresult (*cuStreamCreate)(CUstream *, unsigned int);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamQuery)(CUstream);
    CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int);

    CUresult (*cuEventCreate)(CUevent *, unsigned int);
    CUresult (*cuEventDestroy)(CUevent);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventQuery)(CUevent);
    CUresult (*cuEventSynchronize)(CUevent);

    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);

    // The process-wide instance, or nullptr when the driver could not be loaded and initialised.
    // Loading is attempted once; a failure is remembered, not retried.
    static const CudaLib * get();

    // The driver's own text for a result, for the log. Never null.
    const char * error_text(CUresult result) const;

    // Logs the driver's numeric code and its message, then maps the result.
    // `unclassified` is what to return for a result this function cannot place on its own.
    common::ResponseCode report(CUresult result, const char * operation, common::ResponseCode unclassified) const;
};

// Maps a driver result onto the streamer's codes, by what a caller would do about it.
//
// `unclassified` is the answer for everything not recognised here. The call site supplies it
// because the result alone does not say what was being attempted: the same code means one thing
// from a copy and another from opening a device.
common::ResponseCode to_response_code(CUresult result, common::ResponseCode unclassified);

} // namespace runai::llm::streamer::device::cuda
