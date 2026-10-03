#include <memory>
#include <regex>
#include <vector>

#include "common/exception/exception.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "common/backend_api/response/response.h"
#include "streamer/impl/s3/s3.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

S3::S3(std::shared_ptr<common::s3::S3ClientWrapper> client, const Config & config) :
    Reader(Reader::Mode::Async),
    _client(client),
    _config(config)
{
}

void S3::seek(size_t offset)
{
    LOG(ERROR) << "Not implemented";
    throw common::Exception(common::ResponseCode::UnknownError);
}

void S3::read(size_t bytesize, char * buffer)
{
    LOG(ERROR) << "Not implemented";
    throw common::Exception(common::ResponseCode::UnknownError);
}

void S3::async_read(const common::s3::S3ClientWrapper::Params & params, common::backend_api::ObjectRequestId_t request_handle, const common::Range & range, char * buffer)
{
    auto response_code = _client->async_read(params, request_handle, range, buffer);
    if (response_code != common::ResponseCode::Success)
    {
        throw common::Exception(response_code);
    }
}

common::ResponseCode S3::async_response(std::vector<common::backend_api::Response> & responses, unsigned max_responses)
{
    std::vector<common::backend_api::ObjectCompletionEvent_t> event_buffer(max_responses);
    auto response_code = _client->async_read_response(event_buffer, max_responses);
    if (response_code != common::ResponseCode::Success)
    {
        return response_code;
    }

    responses.reserve(event_buffer.size());
    for (unsigned i = 0; i < event_buffer.size(); ++i)
    {
        responses.emplace_back(event_buffer[i]);
    }
    return common::ResponseCode::Success;
}

size_t S3::max_inflight_bytes() const
{
    return _client->max_inflight_bytes();
}

S3Cleanup::~S3Cleanup()
{
    try
    {
        common::s3::S3ClientWrapper::shutdown();
    }
    catch(...)
    {
    }
}

S3Stop::~S3Stop()
{
    try
    {
        common::s3::S3ClientWrapper::stop();
    }
    catch(...)
    {
    }
}

}; // namespace runai::llm::streamer::impl
