/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/backend_api/object_storage/utils.h"

#include <cstdlib>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::common::backend_api
{

double inflight_window_margin()
{
    if (const char* env = std::getenv("RUNAI_STREAMER_INFLIGHT_WINDOW_MARGIN"))
    {
        char* end = nullptr;
        const double value = std::strtod(env, &end);
        if (end != env && *end == '\0' && value > 0.0)
        {
            return value;
        }
        LOG(ERROR) << "Ignoring invalid RUNAI_STREAMER_INFLIGHT_WINDOW_MARGIN='" << env
                   << "'; expected a positive float, using default 1.5";
    }
    return 1.5;
}

std::ostream & operator<<(std::ostream & os, const ObjectConfigParam_t & param)
{
    os << param.key << " : " << param.value;
    return os;
}

std::ostream & operator<<(std::ostream & os, const ObjectClientConfig_t & config)
{
    os << "endpoint_url: " << config.endpoint_url << ", num_initial_params: " << config.num_initial_params << ", initial_params: ";
    for (unsigned i = 0; i < config.num_initial_params; ++i)
    {
        os << " " << config.initial_params[i] << ", ";
    }
    return os;
}

} // namespace runai::llm::streamer::common::backend_api
