/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include "device/device.h"
#include "device/cuda/cuda_lib.h"

namespace runai::llm::streamer::device::cuda
{

// How a pinned staging buffer is obtained. Selected by RUNAI_STREAMER_PINNED_MEMORY_MODE.
//
// Both exist because which one is faster is unsettled: an allocation pins in one driver call,
// while a registration must fault in every page first but leaves the buffer on ordinary,
// cacheable memory. The answer is expected to differ by host, so it is measured, not argued.
enum class PinnedMode
{
    Allocate,   // cuMemHostAlloc
    Register,   // aligned_alloc + cuMemHostRegister
};

PinnedMode pinned_mode();

// One device and its retained primary context.
class CudaDevice : public Device
{
 public:
    CudaDevice(const CudaLib & lib, CUdevice device, CUcontext context, PinnedMode mode);
    ~CudaDevice() override;

    common::ResponseCode bind_thread() override;
    common::ResponseCode get_attribute(Attribute attribute, int & value) const override;
    common::ResponseCode memory_info(size_t & free_bytes, size_t & total_bytes) const override;

    common::ResponseCode host_alloc(size_t bytesize, void ** ptr) override;
    common::ResponseCode host_free(void * ptr) override;
    common::ResponseCode device_alloc(size_t bytesize, void ** ptr) override;
    common::ResponseCode device_free(void * ptr) override;

    common::ResponseCode stream_create(StreamHandle & stream) override;
    common::ResponseCode stream_destroy(StreamHandle stream) override;
    common::ResponseCode stream_synchronize(StreamHandle stream) override;
    common::ResponseCode stream_query(StreamHandle stream, Status & status) override;
    common::ResponseCode stream_wait_event(StreamHandle stream, EventHandle event) override;

    common::ResponseCode event_create(EventHandle & event) override;
    common::ResponseCode event_destroy(EventHandle event) override;
    common::ResponseCode event_record(EventHandle event, StreamHandle stream) override;
    common::ResponseCode event_query(EventHandle event, Status & status) override;
    common::ResponseCode event_synchronize(EventHandle event) override;

    common::ResponseCode memcpy_h2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream) override;
    common::ResponseCode memcpy_d2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream) override;
    common::ResponseCode memset_async(void * dst, unsigned char value, size_t bytesize, StreamHandle stream) override;

 private:
    // Every member is const: one device is shared by all worker threads, so nothing here may be
    // written after construction.
    const CudaLib &  _lib;
    const CUdevice   _device;
    const CUcontext  _context;
    const PinnedMode _pinned_mode;
};

// The loaded driver. Devices are opened once and shared: the primary context is retained for the
// streamer's lifetime, so pinned buffers allocated against it stay valid for that long.
//
// Reached through backend(), which returns one instance for the process. A second backend would
// keep a second device table and retain the same primary contexts again, which is what the
// "opened once" above is meant to prevent.
class CudaBackend : public Backend
{
 public:
    explicit CudaBackend(const CudaLib & lib);
    ~CudaBackend() override;

    Capabilities capabilities() const override;
    common::ResponseCode device_count(unsigned & count) const override;
    common::ResponseCode open_device(unsigned ordinal, std::shared_ptr<Device> & device) override;

 private:
    const CudaLib & _lib;

    mutable std::mutex _mutex;
    std::map<unsigned, std::shared_ptr<CudaDevice>> _devices;
};

// The CUDA backend, or nullptr where no driver could be loaded. Never throws.
std::shared_ptr<Backend> backend();

} // namespace runai::llm::streamer::device::cuda
