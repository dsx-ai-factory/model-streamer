/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include "common/shared_queue/shared_queue.h"
#include "common/response/response.h"

namespace runai::llm::streamer::common
{

// Responder to hold and return ResponseType objects
// Initialized with the expected number of responses

// Maintains a queue of responses, each response corresponds to a Request object which represents a sub range (e.g. tensor data)
// Implements producer-consumer design:
//    push: ready responses are pushed to the queue and notify the semaphore
//    pop : waits on the semaphore and returns the first response in the queue
//          if no responses are expected returns a ResponseType indicating FinishedError

// Designed for multi producers that push responses and a single consumer that is waiting for responses

using Responder = SharedQueue<Response>;

} // namespace runai::llm::streamer::common
