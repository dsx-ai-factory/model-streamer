#pragma once

#include <functional>
#include <memory>

#include "common/device/device.h"
#include "device/device.h"

namespace runai::llm::streamer::device
{

// How a backend is obtained, deferred so that no driver is loaded until a device is actually asked
// for. A streamer that only ever reads to host memory loads nothing.
using BackendFactory = std::function<std::shared_ptr<Backend>()>;

// The backend for a device type, or an EMPTY factory where there is none - which is the answer for
// Cpu, and for any type this build was not given an implementation of.
//
// Its own package because the lookup must name every vendor, and the interface in device/ must name
// none: having device/ depend on device/cuda would point the dependency the wrong way.
//
// An accelerator needs a backend package and a case here. That is not the whole job: the streamer
// still builds ONE DeviceWriter, for Cuda, so serving a second type also means carrying the type
// down to it instead of only the ordinal.
BackendFactory backend_for(common::DeviceType type);

} // namespace runai::llm::streamer::device
