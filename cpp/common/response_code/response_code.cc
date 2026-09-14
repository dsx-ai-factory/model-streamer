
#include "common/response_code/response_code.h"

#include <array>
#include <string>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::common
{

const char * response_invalid = "Invalid response code";

ResponseCode response_code_from(int value)
{
    if (value < 0 || value >= static_cast<int>(ResponseCode::__Max))
    {
        LOG(ERROR) << "Value " << value << " is not a valid response code";
        throw;
    }

    return static_cast<ResponseCode>(value);
}

constexpr std::array<const char *, static_cast<size_t>(ResponseCode::__Max)> __messages = {
    "Request sent successfuly",
    "Finished all responses",
    "File access error",
    "End of file reached",
    "S3 not supported",
    "GLIBC version should be at least 2.29",
    "Increase process fd limit or decrease the concurrency level. Recommended value for the streamer alone is the concurrency multiplied by 64, in addition to your application fd usage",
    "Invalid request parameters",
    "Empty request parameters",
    "Streamer is handling previous request",
    "CA bundle file not found",
    "Unknown Error",
    "Error loading object storage plugin",
    "GCS not supported",
    "Azure Blob not supported",
    "Object storage returned an unexpected number of bytes for the requested range (truncated or over-length response)",
    "Timed out waiting for a response",
    "Streamer is locked to a single object-storage backend (S3/GCS/Azure); mixing object-storage backends in one streamer or submission is not supported",
    "Credentials were already set to a different value; create a new streamer to use different credentials",
    "Retryable object storage file access error",
    "The filesystem read strategy was already set to a different value; set RUNAI_STREAMER_FS_STRATEGY, or call runai_file_streamer_set_fs_strategy, once before the first request",
    "None of the filesystem read strategies in the list can be served on this host; add sync_buffered to the list to allow the synchronous reader",
    "The asynchronous filesystem reader for this mount failed and will not be used again; the storage itself is healthy, and re-requesting these ranges reads them through the synchronous reader",
    "The requested device type is not supported by this build of the streamer; only RUNAI_FILE_STREAMER_DEVICE_CPU can be read into",
};

const char * description(int response_code)
{
    if (response_code < 0 || response_code >= static_cast<int>(ResponseCode::__Max))
    {
        return response_invalid;
    }

    return __messages[response_code];
}

std::ostream & operator<<(std::ostream & os, const ResponseCode & ret)
{
    return os << " response code: " << description(static_cast<int>(ret));
}

}; // namespace runai::llm::streamer::common
