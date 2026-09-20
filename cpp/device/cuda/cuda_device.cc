#include "device/cuda/cuda_device.h"

#include <unistd.h>

#include <cstdlib>
#include <string>

#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::device::cuda
{

namespace
{

CUstream to_stream(StreamHandle stream)
{
    return reinterpret_cast<CUstream>(stream);
}

CUevent to_event(EventHandle event)
{
    return reinterpret_cast<CUevent>(event);
}

size_t page_size()
{
    const long size = ::sysconf(_SC_PAGESIZE);
    return size > 0 ? static_cast<size_t>(size) : 4096;
}

} // namespace

PinnedMode pinned_mode()
{
    const std::string mode = utils::getenv<std::string>("RUNAI_STREAMER_PINNED_MEMORY_MODE", std::string("allocate"));
    if (mode == "register")
    {
        return PinnedMode::Register;
    }
    if (mode != "allocate")
    {
        LOG(WARNING) << "[RunAI Streamer] unknown RUNAI_STREAMER_PINNED_MEMORY_MODE '" << mode << "'; using allocate";
    }
    return PinnedMode::Allocate;
}

CudaDevice::CudaDevice(const CudaLib & lib, CUdevice device, CUcontext context, PinnedMode mode) :
    _lib(lib),
    _device(device),
    _context(context),
    _pinned_mode(mode)
{
}

CudaDevice::~CudaDevice()
{
    _lib.cuDevicePrimaryCtxRelease(_device);
}

common::ResponseCode CudaDevice::bind_thread()
{
    return _lib.report(_lib.cuCtxSetCurrent(_context), "cuCtxSetCurrent", common::ResponseCode::DeviceDriverError);
}

common::ResponseCode CudaDevice::get_attribute(Attribute attribute, int & value) const
{
    CUdevice_attribute name = CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
    switch (attribute)
    {
        case Attribute::UnifiedAddressing:
            name = CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
            break;
        case Attribute::PageableAccessUsesHostPageTables:
            name = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES;
            break;
    }
    return _lib.report(_lib.cuDeviceGetAttribute(&value, name, _device), "cuDeviceGetAttribute", common::ResponseCode::DeviceDriverError);
}

common::ResponseCode CudaDevice::memory_info(size_t & free_bytes, size_t & total_bytes) const
{
    return _lib.report(_lib.cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo", common::ResponseCode::DeviceDriverError);
}

common::ResponseCode CudaDevice::host_alloc(size_t bytesize, void ** ptr)
{
    *ptr = nullptr;

    if (_pinned_mode == PinnedMode::Allocate)
    {
        return _lib.report(_lib.cuMemHostAlloc(ptr, bytesize, 0), "cuMemHostAlloc", common::ResponseCode::DeviceOutOfMemory);
    }

    // Page aligned because page-locking works on whole pages, and rounded up for the same reason:
    // a tail sharing its page with something else would pin that too.
    const size_t alignment = page_size();
    const size_t rounded = (bytesize + alignment - 1) / alignment * alignment;

    void * const memory = ::aligned_alloc(alignment, rounded);
    if (memory == nullptr)
    {
        LOG(ERROR) << "[RunAI Streamer] failed to allocate " << rounded << " bytes of host memory to register";
        return common::ResponseCode::DeviceOutOfMemory;
    }

    // PORTABLE always: cuda.h promises allocated pinned memory to every context under unified
    // addressing, but makes no such promise for registered memory - only this flag does, and the
    // streamer may hold more than one device.
    const CUresult result = _lib.cuMemHostRegister(memory, rounded, CU_MEMHOSTREGISTER_PORTABLE);
    if (result != CUDA_SUCCESS)
    {
        ::free(memory);
        return _lib.report(result, "cuMemHostRegister", common::ResponseCode::DeviceOutOfMemory);
    }

    *ptr = memory;
    return common::ResponseCode::Success;
}

common::ResponseCode CudaDevice::host_free(void * ptr)
{
    if (ptr == nullptr)
    {
        return common::ResponseCode::Success;
    }

    if (_pinned_mode == PinnedMode::Allocate)
    {
        return _lib.report(_lib.cuMemFreeHost(ptr), "cuMemFreeHost", common::ResponseCode::DeviceDriverError);
    }

    const common::ResponseCode code = _lib.report(_lib.cuMemHostUnregister(ptr), "cuMemHostUnregister", common::ResponseCode::DeviceDriverError);
    ::free(ptr);
    return code;
}

common::ResponseCode CudaDevice::device_alloc(size_t bytesize, void ** ptr)
{
    CUdeviceptr allocated = 0;
    const common::ResponseCode code = _lib.report(_lib.cuMemAlloc(&allocated, bytesize), "cuMemAlloc", common::ResponseCode::DeviceOutOfMemory);
    *ptr = reinterpret_cast<void *>(allocated);
    return code;
}

common::ResponseCode CudaDevice::device_free(void * ptr)
{
    return _lib.report(_lib.cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)), "cuMemFree", common::ResponseCode::DeviceDriverError);
}

common::ResponseCode CudaDevice::stream_create(StreamHandle & stream)
{
    CUstream created = nullptr;
    const common::ResponseCode code = _lib.report(_lib.cuStreamCreate(&created, CU_STREAM_NON_BLOCKING), "cuStreamCreate", common::ResponseCode::DeviceTransferError);
    stream = created;
    return code;
}

common::ResponseCode CudaDevice::stream_destroy(StreamHandle stream)
{
    return _lib.report(_lib.cuStreamDestroy(to_stream(stream)), "cuStreamDestroy", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::stream_synchronize(StreamHandle stream)
{
    return _lib.report(_lib.cuStreamSynchronize(to_stream(stream)), "cuStreamSynchronize", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::stream_query(StreamHandle stream, Status & status)
{
    const CUresult result = _lib.cuStreamQuery(to_stream(stream));
    if (result == CUDA_ERROR_NOT_READY)
    {
        status = Status::NotReady;
        return common::ResponseCode::Success;
    }

    const common::ResponseCode code = _lib.report(result, "cuStreamQuery", common::ResponseCode::DeviceTransferError);
    if (code == common::ResponseCode::Success)
    {
        status = Status::Ready;
    }
    return code;
}

common::ResponseCode CudaDevice::stream_wait_event(StreamHandle stream, EventHandle event)
{
    return _lib.report(_lib.cuStreamWaitEvent(to_stream(stream), to_event(event), 0), "cuStreamWaitEvent", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::event_create(EventHandle & event)
{
    CUevent created = nullptr;
    // BLOCKING_SYNC is what makes a waiting thread free. Without it the driver spins: measured on an
    // H200 and a B200, waiting on a 29 ms copy costs 100% of a core by default and 1% with this flag,
    // at the same wall time. InstantTensor does not set it, so its wait thread burns a core per chunk.
    //
    // DISABLE_TIMING because nothing here reads a duration, and untimed events are cheaper.
    const unsigned int flags = CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC;
    const common::ResponseCode code = _lib.report(_lib.cuEventCreate(&created, flags), "cuEventCreate", common::ResponseCode::DeviceTransferError);
    event = created;
    return code;
}

common::ResponseCode CudaDevice::event_destroy(EventHandle event)
{
    return _lib.report(_lib.cuEventDestroy(to_event(event)), "cuEventDestroy", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::event_record(EventHandle event, StreamHandle stream)
{
    return _lib.report(_lib.cuEventRecord(to_event(event), to_stream(stream)), "cuEventRecord", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::event_query(EventHandle event, Status & status)
{
    const CUresult result = _lib.cuEventQuery(to_event(event));
    if (result == CUDA_ERROR_NOT_READY)
    {
        status = Status::NotReady;
        return common::ResponseCode::Success;
    }

    const common::ResponseCode code = _lib.report(result, "cuEventQuery", common::ResponseCode::DeviceTransferError);
    if (code == common::ResponseCode::Success)
    {
        status = Status::Ready;
    }
    return code;
}

common::ResponseCode CudaDevice::event_synchronize(EventHandle event)
{
    return _lib.report(_lib.cuEventSynchronize(to_event(event)), "cuEventSynchronize", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::memcpy_h2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream)
{
    return _lib.report(_lib.cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), src, bytesize, to_stream(stream)), "cuMemcpyHtoDAsync", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::memcpy_d2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream)
{
    return _lib.report(_lib.cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), bytesize, to_stream(stream)), "cuMemcpyDtoDAsync", common::ResponseCode::DeviceTransferError);
}

common::ResponseCode CudaDevice::memset_async(void * dst, unsigned char value, size_t bytesize, StreamHandle stream)
{
    return _lib.report(_lib.cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(dst), value, bytesize, to_stream(stream)), "cuMemsetD8Async", common::ResponseCode::DeviceTransferError);
}

CudaBackend::CudaBackend(const CudaLib & lib) :
    _lib(lib)
{
}

CudaBackend::~CudaBackend() = default;

Capabilities CudaBackend::capabilities() const
{
    Capabilities capabilities;
    capabilities.pinned_host_memory = true;
    capabilities.device_to_device = true;
    return capabilities;
}

common::ResponseCode CudaBackend::device_count(unsigned & count) const
{
    count = 0;

    int devices = 0;
    const common::ResponseCode code =
        _lib.report(_lib.cuDeviceGetCount(&devices), "cuDeviceGetCount", common::ResponseCode::DeviceDriverError);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    count = devices > 0 ? static_cast<unsigned>(devices) : 0;
    return common::ResponseCode::Success;
}

common::ResponseCode CudaBackend::open_device(unsigned ordinal, std::shared_ptr<Device> & device)
{
    const std::lock_guard<std::mutex> guard(_mutex);

    const auto cached = _devices.find(ordinal);
    if (cached != _devices.end())
    {
        device = cached->second;
        return common::ResponseCode::Success;
    }

    CUdevice handle = 0;
    common::ResponseCode code =
        _lib.report(_lib.cuDeviceGet(&handle, static_cast<int>(ordinal)), "cuDeviceGet", common::ResponseCode::InvalidDevice);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // Retained, not created: the primary context is the one the framework in this process already
    // uses, so buffers and streams are shared with it rather than competing with it.
    CUcontext context = nullptr;
    code = _lib.report(_lib.cuDevicePrimaryCtxRetain(&context, handle), "cuDevicePrimaryCtxRetain", common::ResponseCode::DeviceDriverError);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    try
    {
        auto opened = std::make_shared<CudaDevice>(_lib, handle, context, pinned_mode());
        _devices.emplace(ordinal, opened);
        device = opened;
    }
    catch (...)
    {
        // The retain above has no owner yet, so nothing else would ever release it.
        _lib.cuDevicePrimaryCtxRelease(handle);
        throw;
    }

    return common::ResponseCode::Success;
}

std::shared_ptr<Backend> backend()
{
    // One per process: the device table and the contexts it retains are the point of the class,
    // and a second instance would quietly duplicate both.
    static const std::shared_ptr<Backend> instance = []() -> std::shared_ptr<Backend>
        {
            const CudaLib * const lib = CudaLib::get();
            return lib != nullptr ? std::make_shared<CudaBackend>(*lib) : nullptr;
        }();

    return instance;
}

} // namespace runai::llm::streamer::device::cuda
