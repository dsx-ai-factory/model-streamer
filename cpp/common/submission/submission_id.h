#pragma once

#include "streamer/submission_id.h"

namespace runai::llm::streamer
{

// The C API's submission id under the name the C++ layers use.
//
// The alias lives here, in a dependency-free leaf under common/, so that the lower layers carrying the
// id - common::Response, SubmissionsMgr, Batch, AsyncIoStats - can name it without including the whole
// C API header, which would invert the layering.
using SubmissionId = RunaiFileStreamerSubmissionId;

} // namespace runai::llm::streamer
