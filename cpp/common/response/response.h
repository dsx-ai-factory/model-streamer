
#pragma once

#include <ostream>

#include "common/response_code/response_code.h"
#include "common/submission/submission_id.h"

namespace runai::llm::streamer::common
{

struct Response
{
    // submission_id identifies the runai_file_streamer_request submission this response belongs to,
    // so a shared responder can be demuxed across concurrent submissions. It is NOT the
    // internal impl::Request (a sub-range, addressed by file_index + index).
    Response(SubmissionId submission_id, unsigned file_index, unsigned index, common::ResponseCode ret);
    Response(unsigned file_index, unsigned index, common::ResponseCode ret);
    Response(unsigned index, common::ResponseCode ret);
    Response(unsigned index);
    Response(common::ResponseCode ret);


    bool operator==(const common::ResponseCode other);
    bool operator!=(const common::ResponseCode other);

    // id of the owning submission (one runai_file_streamer_request call); 0 by default
    SubmissionId submission_id = 0;

    // index of file (0 by default: sentinel responses - FinishedError/TimedOut, built via the
    // 1-arg ctor - carry no file/sub-range, and the C API copies these out unconditionally)
    unsigned file_index = 0;

    // index of sub request
    unsigned index = 0;


    // response code
    common::ResponseCode ret;
};

std::ostream & operator<<(std::ostream & os, const Response & response);

}; // namespace runai::llm::streamer::common
