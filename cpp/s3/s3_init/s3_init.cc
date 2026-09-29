/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include "s3/s3_init/s3_init.h"

#include "utils/logging/logging.h"
#include "utils/env/env.h"

namespace runai::llm::streamer::impl::s3
{

S3Init::S3Init()
{
    options.httpOptions.installSigPipeHandler = true;
    auto trace_aws = utils::getenv<bool>("RUNAI_STREAMER_S3_TRACE", false);
    if (trace_aws)
    {
        // aws trace logs are written to a file
        options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Trace;
    }

    Aws::InitAPI(options);
}

S3Init::~S3Init()
{
    LOG(DEBUG) << "Shutting down s3";
    try
    {
        Aws::ShutdownAPI(options);
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while shutting down";
    }
}

}; // namespace runai::llm::streamer::impl::s3