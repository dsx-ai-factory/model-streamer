/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include "common/backend_api/object_storage/object_storage.h"

namespace runai::llm::streamer::common::backend_api
{

std::ostream & operator<<(std::ostream & os, const ObjectConfigParam_t & param);

std::ostream & operator<<(std::ostream & os, const ObjectClientConfig_t & config);

} //namespace runai::llm::streamer::common::backend_api
