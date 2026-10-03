#pragma once

#include <memory>
#include <vector>

#include "common/s3_wrapper/s3_wrapper.h"
#include "streamer/impl/reader/reader.h"
#include "streamer/impl/config/config/config.h"

namespace runai::llm::streamer::impl
{

struct S3Cleanup
{
    S3Cleanup() = default;

    ~S3Cleanup();
};

struct S3Stop
{
    S3Stop() = default;

    ~S3Stop();
};

struct S3 : Reader
{
    S3(std::shared_ptr<common::s3::S3ClientWrapper> client, const Config & config);
    virtual ~S3() {}

    void read(size_t bytesize, char * buffer) override;
    void seek(size_t offset) override;

    void async_read(const common::s3::S3ClientWrapper::Params & params, common::backend_api::ObjectRequestId_t request_handle, const common::Range & range, char * buffer) override;
    common::ResponseCode async_response(std::vector<common::backend_api::Response> & responses, unsigned max_responses) override;

    size_t max_inflight_bytes() const override;

 private:
    std::shared_ptr<common::s3::S3ClientWrapper> _client;
    const Config & _config;
};

}; //namespace runai::llm::streamer::impl
