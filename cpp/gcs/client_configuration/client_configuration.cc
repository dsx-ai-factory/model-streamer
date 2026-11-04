/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gcs/client_configuration/client_configuration.h"

#include "google/cloud/storage/client.h"
#include "google/cloud/storage/grpc_plugin.h"

#include "common/exception/exception.h"
#include "common/response_code/response_code.h"
#include "utils/logging/logging.h"
#include "utils/env/env.h"

#include "google/cloud/log.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <unistd.h>

namespace runai::llm::streamer::impl::gcs
{

namespace
{

constexpr char kKeyFileEnv[] = "RUNAI_STREAMER_GCS_CREDENTIAL_FILE";
constexpr char kAdcFileEnv[] = "GOOGLE_APPLICATION_CREDENTIALS";
constexpr char kSdkTracingEnv[] = "GOOGLE_CLOUD_CPP_ENABLE_TRACING";

// Tracing components only produce messages; the SDK prints them through its log sink, whose
// default backend passes FATAL and above unless GOOGLE_CLOUD_CPP_ENABLE_CLOG is set. Without this,
// RUNAI_STREAMER_S3_TRACE generated trace messages and dropped them.
//
// EnableStdClog() is a no-op while a default backend exists, so the default is replaced. A logging
// setup the user chose through the SDK's own variables is left alone.
void print_sdk_logs()
{
    static std::once_flag once;
    std::call_once(once, []
    {
        if (!utils::getenv<std::string>("GOOGLE_CLOUD_CPP_ENABLE_CLOG", "").empty() ||
            !utils::getenv<std::string>("GOOGLE_CLOUD_CPP_EXPERIMENTAL_LOG_CONFIG", "").empty())
        {
            LOG(DEBUG) << "google-cloud-cpp log output is configured by its own environment variables";
            return;
        }

        LOG(DEBUG) << "Printing google-cloud-cpp logs to stderr";
        google::cloud::LogSink::DisableStdClog();
        google::cloud::LogSink::EnableStdClog();
    });
}

} // namespace

std::string credentials_source()
{
    const auto key_file = utils::getenv<std::string>(kKeyFileEnv, "");
    if (!key_file.empty())
    {
        return std::string("service account key file ") + key_file + " (" + kKeyFileEnv + ")";
    }

    const auto adc_file = utils::getenv<std::string>(kAdcFileEnv, "");
    if (!adc_file.empty())
    {
        std::string description = std::string("application default credentials file ") + adc_file + " (" + kAdcFileEnv + ")";
        if (::access(adc_file.c_str(), R_OK) != 0)
        {
            description += " - not readable: ";
            description += std::strerror(errno);
        }
        return description;
    }

    return "application default credentials - the gcloud well-known file if present, otherwise the metadata server";
}

ClientConfiguration::ClientConfiguration(unsigned concurrent_readers)
{
    use_grpc = utils::getenv<bool>("RUNAI_STREAMER_GCS_USE_GRPC", false);

    const auto max_connections = utils::getenv<unsigned long>("RUNAI_STREAMER_S3_MAX_CONNECTIONS", 0);
    if (max_connections) {
        max_concurrency = max_connections;
    } else {
        unsigned nprocs = std::thread::hardware_concurrency();
        // Use at least 8 threads if hardware_concurrency cannot be computed.
        LOG(SPAM) << "Hardware concurrency detected: " << nprocs;
        unsigned default_max_concurrency = nprocs == 0 ? 8U : 1U;
        // A DIVISOR below, so it must not be zero: an unstated count means the caller said nothing,
        // not that it will build no clients.
        // Floored, not defaulted: a zero would divide by zero, and only a caller outside the streamer
        // can send one.
        const unsigned worker_concurrency = std::max(1U, concurrent_readers);
        LOG(SPAM) << "Streamer worker concurrency: " << worker_concurrency;
        max_concurrency = std::max(default_max_concurrency, nprocs * 2 / worker_concurrency);
    }
    LOG(DEBUG) << "GCS per-client concurrency is set to: " << max_concurrency;

    // if the transfer speed is less than the low speed limit for request_timeout_ms milliseconds the transfer is aborted and retried
    const auto request_timeout_ms = utils::getenv<unsigned long>("RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS", 600000);
    if (request_timeout_ms)
    {
        LOG(DEBUG) << "GCS request timeout is set to " << request_timeout_ms << " ms";
        std::chrono::seconds request_timeout_seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::milliseconds(request_timeout_ms));
        options.set<google::cloud::storage::DownloadStallTimeoutOption>(request_timeout_seconds);
    }

    const auto low_speed_limit = utils::getenv<unsigned long>("RUNAI_STREAMER_S3_LOW_SPEED_LIMIT", 0);
    if (low_speed_limit)
    {
        LOG(DEBUG) << "GCS minimum speed is set to " << low_speed_limit << " bytes in second";
        options.set<google::cloud::storage::DownloadStallMinimumRateOption>(low_speed_limit);
    }

    const auto trace_gcs = utils::getenv<bool>("RUNAI_STREAMER_S3_TRACE", false);
    if (trace_gcs)
    {
        // The SDK reads GOOGLE_CLOUD_CPP_ENABLE_TRACING only when the option is unset, so a
        // component list the user chose is left for it to apply.
        const auto sdk_tracing = utils::getenv<std::string>(kSdkTracingEnv, "");
        if (sdk_tracing.empty())
        {
            LOG(DEBUG) << "Enabling google-cloud-cpp tracing of the rpc/auth/http components";
            options.set<google::cloud::LoggingComponentsOption>(std::set<std::string>{"rpc", "auth", "http"});
        }
        else
        {
            LOG(DEBUG) << "Enabling google-cloud-cpp tracing of the components in " << kSdkTracingEnv << "=" << sdk_tracing;
        }
        print_sdk_logs();
    }

    credentials = credentials_source();
    LOG(DEBUG) << "GCS credentials: " << credentials;

    const auto sa_key_file_name = utils::getenv<std::string>(kKeyFileEnv, "");
    if (!sa_key_file_name.empty()) {
        try {
            // Open the file stream.
            auto is = std::ifstream(sa_key_file_name);

            // Configure the stream to throw an exception on any failure (e.g., file not found, read error).
            is.exceptions(std::ifstream::failbit | std::ifstream::badbit);

            // Read the entire file into the 'contents' string.
            auto contents = std::string(std::istreambuf_iterator<char>(is.rdbuf()), {});

            // Use the contents to set the credentials.
            options.set<google::cloud::UnifiedCredentialsOption>(
                google::cloud::MakeServiceAccountCredentials(contents));
        } catch (const std::ios_base::failure& ex) {
            const int error = errno;
            LOG(ERROR) << "Failed to read service account key file " << sa_key_file_name
                       << " (" << kKeyFileEnv << "): " << std::strerror(error);
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }
    }
}

}; // namespace runai::llm::streamer::impl::gcs
