// A fake libcuda.so.1, for testing the LOADER without a driver.
//
// Everything else about CudaDevice and CudaBackend is testable by filling CudaLib's function pointers
// with stubs - no dlopen, no library, no driver. What that cannot reach is loading itself: finding the
// library, resolving each name, falling back from the _v2 form, and cuInit. This exists for that, and
// it therefore exports EVERY symbol the loader asks for. One missing name and load_symbols() gives up
// on the whole library, which is exactly the failure this is here to catch.
//
// Built as libcuda.so.1 - the name dlopen searches for - and found through the test binary's DT_RPATH
// ahead of any real driver, so the test behaves the same on a GPU machine and on CPU-only CI.
//
// The calls do as little as possible while staying honest about ownership: pinned memory is malloc,
// device memory is malloc, and a copy is memcpy. That is enough for the loader tests and keeps this
// from drifting into a second MockDevice, which already exists for the layer above.

#include <atomic>
#include <cstdlib>
#include <cstring>

// ABI-compatible declarations, so this builds with no CUDA SDK present.
typedef int                  CUresult;
typedef unsigned long long   CUdeviceptr;
typedef int                  CUdevice;
typedef struct CUstream_st * CUstream;
typedef struct CUctx_st *    CUcontext;
typedef struct CUevent_st *  CUevent;

#define CUDA_SUCCESS                 0
#define CUDA_ERROR_INVALID_VALUE     1
#define CUDA_ERROR_OUT_OF_MEMORY     2
#define CUDA_ERROR_NOT_READY       600

namespace
{

std::atomic<unsigned> __inits{0};
std::atomic<unsigned> __devices{1};      // one device, unless a test says otherwise
std::atomic<uintptr_t> __next_handle{1};

void * handle()
{
    return reinterpret_cast<void *>(__next_handle.fetch_add(1));
}

} // namespace

extern "C"
{

// --- what a test controls and reads ---

void runai_mock_cuda_set_device_count(unsigned count) { __devices.store(count); }
unsigned runai_mock_cuda_init_calls()                 { return __inits.load(); }

// --- library, versions and error text ---

CUresult cuInit(unsigned int) { __inits.fetch_add(1); return CUDA_SUCCESS; }

CUresult cuDriverGetVersion(int * version)
{
    if (version == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *version = 12000;
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult, const char ** name)
{
    if (name == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *name = "MOCK_CUDA_ERROR";
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult, const char ** text)
{
    if (text == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *text = "mock cuda error";
    return CUDA_SUCCESS;
}

// --- devices and contexts ---

CUresult cuDeviceGetCount(int * count)
{
    if (count == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *count = static_cast<int>(__devices.load());
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice * device, int ordinal)
{
    if (device == nullptr || ordinal < 0 || static_cast<unsigned>(ordinal) >= __devices.load())
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int * value, int, CUdevice)
{
    if (value == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *value = 1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext * context, CUdevice)
{
    if (context == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *context = static_cast<CUcontext>(handle());
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice) { return CUDA_SUCCESS; }
CUresult cuCtxSetCurrent(CUcontext)             { return CUDA_SUCCESS; }

CUresult cuCtxGetCurrent(CUcontext * context)
{
    if (context == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *context = static_cast<CUcontext>(handle());
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_v2(size_t * free_bytes, size_t * total_bytes)
{
    if (free_bytes == nullptr || total_bytes == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *total_bytes = 1ull << 30;
    *free_bytes = *total_bytes / 2;
    return CUDA_SUCCESS;
}

// --- memory: pinned and device alike are ordinary allocations here ---

CUresult cuMemHostAlloc(void ** out, size_t bytesize, unsigned int)
{
    if (out == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *out = std::malloc(bytesize);
    return *out != nullptr ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuMemFreeHost(void * p)                          { std::free(p); return CUDA_SUCCESS; }
CUresult cuMemHostRegister_v2(void *, size_t, unsigned int) { return CUDA_SUCCESS; }
CUresult cuMemHostUnregister(void *)                      { return CUDA_SUCCESS; }

CUresult cuMemAlloc_v2(CUdeviceptr * out, size_t bytesize)
{
    if (out == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    void * const memory = std::malloc(bytesize);
    *out = reinterpret_cast<CUdeviceptr>(memory);
    return memory != nullptr ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuMemFree_v2(CUdeviceptr p)
{
    std::free(reinterpret_cast<void *>(p));
    return CUDA_SUCCESS;
}

// --- streams ---

CUresult cuStreamCreate(CUstream * stream, unsigned int)
{
    if (stream == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *stream = static_cast<CUstream>(handle());
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy_v2(CUstream)            { return CUDA_SUCCESS; }
CUresult cuStreamSynchronize(CUstream)           { return CUDA_SUCCESS; }
CUresult cuStreamQuery(CUstream)                 { return CUDA_SUCCESS; }
CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned int) { return CUDA_SUCCESS; }

// --- events ---

CUresult cuEventCreate(CUevent * event, unsigned int)
{
    if (event == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
    *event = static_cast<CUevent>(handle());
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy_v2(CUevent)          { return CUDA_SUCCESS; }
CUresult cuEventRecord(CUevent, CUstream)    { return CUDA_SUCCESS; }
CUresult cuEventQuery(CUevent)               { return CUDA_SUCCESS; }
CUresult cuEventSynchronize(CUevent)         { return CUDA_SUCCESS; }

// --- copies: a memcpy, so a test can check the bytes that arrived ---

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void * src, size_t bytesize, CUstream)
{
    std::memcpy(reinterpret_cast<void *>(dst), src, bytesize);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytesize, CUstream)
{
    std::memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), bytesize);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t bytesize, CUstream)
{
    std::memset(reinterpret_cast<void *>(dst), value, bytesize);
    return CUDA_SUCCESS;
}

} // extern "C"
