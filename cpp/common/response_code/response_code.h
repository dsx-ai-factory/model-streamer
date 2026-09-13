
#pragma once

#include <ostream>

#include "streamer/response_code.h"

namespace runai::llm::streamer::common
{

// The C++ name for each published code. Every value comes from <streamer/response_code.h>, so the two
// cannot drift: a code is added there first, and named here.
enum class ResponseCode : int
{
    Success                  = NV_FILE_STREAMER_RESPONSE_SUCCESS,

    FinishedError            = NV_FILE_STREAMER_RESPONSE_FINISHED_ERROR,
    FileAccessError          = NV_FILE_STREAMER_RESPONSE_FILE_ACCESS_ERROR,
    EofError                 = NV_FILE_STREAMER_RESPONSE_EOF_ERROR,
    S3NotSupported           = NV_FILE_STREAMER_RESPONSE_S3_NOT_SUPPORTED,
    GlibcPrerequisite        = NV_FILE_STREAMER_RESPONSE_GLIBC_PREREQUISITE,
    InsufficientFdLimit      = NV_FILE_STREAMER_RESPONSE_INSUFFICIENT_FD_LIMIT,
    InvalidParameterError    = NV_FILE_STREAMER_RESPONSE_INVALID_PARAMETER_ERROR,
    EmptyRequestError        = NV_FILE_STREAMER_RESPONSE_EMPTY_REQUEST_ERROR,
    BusyError                = NV_FILE_STREAMER_RESPONSE_BUSY_ERROR,
    CaFileNotFound           = NV_FILE_STREAMER_RESPONSE_CA_FILE_NOT_FOUND,
    UnknownError             = NV_FILE_STREAMER_RESPONSE_UNKNOWN_ERROR,
    ObjPluginLoadError       = NV_FILE_STREAMER_RESPONSE_OBJ_PLUGIN_LOAD_ERROR,
    GCSNotSupported          = NV_FILE_STREAMER_RESPONSE_GCS_NOT_SUPPORTED,
    AzureBlobNotSupported    = NV_FILE_STREAMER_RESPONSE_AZURE_BLOB_NOT_SUPPORTED,
    FileTruncatedError       = NV_FILE_STREAMER_RESPONSE_FILE_TRUNCATED_ERROR,
    TimedOut                 = NV_FILE_STREAMER_RESPONSE_TIMED_OUT,
    UnsupportedBackendMix    = NV_FILE_STREAMER_RESPONSE_UNSUPPORTED_BACKEND_MIX,
    CredentialsAlreadySet    = NV_FILE_STREAMER_RESPONSE_CREDENTIALS_ALREADY_SET,
    RetryableFileAccessError = NV_FILE_STREAMER_RESPONSE_RETRYABLE_FILE_ACCESS_ERROR,

    // Filesystem strategy problems. Two codes, because the operator has to do something different
    // for each one: set the value once, or add a candidate the host can serve.
    //
    // Both used to report UnsupportedBackendMix, whose message is about mixing S3, GCS and Azure.
    // That sent the reader to object storage for a problem that has nothing to do with it.
    FsStrategyConflict       = NV_FILE_STREAMER_RESPONSE_FS_STRATEGY_CONFLICT,
    FsStrategyUnavailable    = NV_FILE_STREAMER_RESPONSE_FS_STRATEGY_UNAVAILABLE,

    // One mount's asynchronous reader failed permanently, mid-run - io_uring_submit or io_getevents
    // returned an error that is not backpressure.
    //
    // NOT UnknownError. That code means the failure is ours and nothing we report can be trusted, so a
    // caller seeing it should abort everything and treat it as a bug in the streamer. None of that
    // applies here: every other mount has its own engine, object storage is a different pool, and this
    // mount is still readable by the synchronous reader.
    //
    // NOT FileAccessError either. The storage is healthy and the ring is not, and that code sends an
    // operator to look at the wrong thing - the same mistake the two codes above were added to correct.
    //
    // What a caller does with it: these ranges were not read, and asking for them again succeeds,
    // because the engine is not reused.
    //
    // ONE code for both engines. The decision it drives is the same whichever one failed; which engine
    // it was, and with what errno, is in the log.
    FsAsyncEngineError       = NV_FILE_STREAMER_RESPONSE_FS_ASYNC_ENGINE_ERROR,

    // The submission named a device this build cannot serve.
    UnsupportedDeviceType    = NV_FILE_STREAMER_RESPONSE_UNSUPPORTED_DEVICE_TYPE,

    __Max,
};

const char * description(int response_code);

ResponseCode response_code_from(int value);

std::ostream & operator<<(std::ostream &, const ResponseCode &);

}; // namespace runai::llm::streamer::common
