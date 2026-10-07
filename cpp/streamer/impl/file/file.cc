/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/file/file.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <utility>

#include "common/exception/exception.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

File::File(const std::string & path, const Config & config) :
    Reader(Reader::Mode::Sync),
    // Not a bare open: a FIFO would block this thread forever, and a directory would fail later with
    // a different error than the asynchronous readers give (utils::Fd::open_for_read).
    _fd(utils::Fd::open_for_read(path)),
    _block_size(config.fs_sync_read_block_bytesize)
{
    if (_fd.fd() == -1)
    {
        LOG(ERROR) << "Failed to access file " << path;
        throw common::Exception(common::ResponseCode::FileAccessError);
    }
}

void File::seek(size_t offset)
{
    try
    {
        _fd.seek(offset);
    }
    catch(const std::exception& e)
    {
        throw common::Exception(common::ResponseCode::EofError);
    }
}

void File::read(size_t bytesize, char * buffer)
{
    size_t result = 0;
    try
    {
        result = _fd.read(bytesize, buffer, utils::Fd::Read::Eof, _block_size);
    }
    catch(const std::exception& e)
    {
        // FileAccessError, not UnknownError: a failed read is attributable to THIS file, and the caller
        // may keep going with the rest. UnknownError is reserved for unrecoverable conditions (corruption,
        // out of memory) and tells the caller to abort everything - so using it here would let one file's
        // I/O error poison every other in-flight submission. seek() and the short-read path below already
        // report specific codes for the same reason.
        LOG(ERROR) << "Failed to read " << bytesize << " bytes with fd " << _fd.fd();
        throw common::Exception(common::ResponseCode::FileAccessError);
    }

    if (result != bytesize)
    {
        LOG(ERROR) << "Read " << result << " bytes. Expected " << bytesize << " bytes with fd " << _fd.fd();
        throw common::Exception(common::ResponseCode::EofError);
    }
}

void File::async_read(const common::s3::S3ClientWrapper::Params & params, common::backend_api::ObjectRequestId_t request_handle, const common::Range & range, char * buffer)
{
    LOG(ERROR) << "Not implemented";
    throw common::Exception(common::ResponseCode::UnknownError);
}

common::ResponseCode File::async_response(std::vector<common::backend_api::Response> & responses, unsigned max_responses)
{
    LOG(ERROR) << "Not implemented";
    throw common::Exception(common::ResponseCode::UnknownError);
}

}; // namespace runai::llm::streamer::impl
