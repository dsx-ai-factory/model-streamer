/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include "s3/s3_init/s3_init.h"

#include "s3/ca_http_client_factory/ca_http_client_factory.h"

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

    // The SDK creates the HTTP clients for STS, SSO and IMDS itself, without a CA; give them the same CA
    // the S3 client uses (AWS_CA_BUNDLE, profile ca_bundle), or the system bundle
    const bool init_curl = options.httpOptions.initAndCleanupCurl;
    const bool install_sigpipe_handler = options.httpOptions.installSigPipeHandler;
    options.httpOptions.httpClientFactory_create_fn = [init_curl, install_sigpipe_handler]()
    {
        return Aws::MakeShared<CaHttpClientFactory>("RunaiS3Init", init_curl, install_sigpipe_handler);
    };

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