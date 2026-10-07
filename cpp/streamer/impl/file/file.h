/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <string>
#include <vector>

#include "streamer/impl/reader/reader.h"
#include "streamer/impl/config/config/config.h"

#include "utils/fd/fd.h"

namespace runai::llm::streamer::impl
{

struct File : Reader
{
    File(const std::string & path, const Config & config);
    virtual ~File() {}

    void read(size_t bytesize, char * buffer) override;

    void seek(size_t offset) override;

    void async_read(const common::s3::S3ClientWrapper::Params & params, common::backend_api::ObjectRequestId_t request_handle, const common::Range & range, char * buffer) override;
    common::ResponseCode async_response(std::vector<common::backend_api::Response> & responses, unsigned max_responses) override;

 private:
    utils::Fd _fd;
    size_t _block_size;
};

}; // namespace runai::llm::streamer::impl
