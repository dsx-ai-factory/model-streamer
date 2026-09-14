#ifndef RUNAI_FILE_STREAMER_RESPONSE_CODE_H
#define RUNAI_FILE_STREAMER_RESPONSE_CODE_H

// Plain C that also compiles as C++, because this header ships in the SDK tarball and a C program
// must be able to include it.

// What every entry point returns. The value is the contract, so each one is written out: a code may
// only be APPENDED, and no existing number may change. A compiled caller holds the old numbers.
//
// runai_file_streamer_response_str() turns any of these into a message, including a value this build does not know.
typedef enum RunaiFileStreamerResponseCode
{
    RUNAI_FILE_STREAMER_RESPONSE_SUCCESS                      = 0,

    RUNAI_FILE_STREAMER_RESPONSE_FINISHED_ERROR               = 1,
    RUNAI_FILE_STREAMER_RESPONSE_FILE_ACCESS_ERROR            = 2,
    RUNAI_FILE_STREAMER_RESPONSE_EOF_ERROR                    = 3,
    RUNAI_FILE_STREAMER_RESPONSE_S3_NOT_SUPPORTED             = 4,
    RUNAI_FILE_STREAMER_RESPONSE_GLIBC_PREREQUISITE           = 5,
    RUNAI_FILE_STREAMER_RESPONSE_INSUFFICIENT_FD_LIMIT        = 6,
    RUNAI_FILE_STREAMER_RESPONSE_INVALID_PARAMETER_ERROR      = 7,
    RUNAI_FILE_STREAMER_RESPONSE_EMPTY_REQUEST_ERROR          = 8,

    // Not returned by this build. The number is kept because removing it would move every code below.
    RUNAI_FILE_STREAMER_RESPONSE_BUSY_ERROR                   = 9,

    RUNAI_FILE_STREAMER_RESPONSE_CA_FILE_NOT_FOUND            = 10,
    RUNAI_FILE_STREAMER_RESPONSE_UNKNOWN_ERROR                = 11,
    RUNAI_FILE_STREAMER_RESPONSE_OBJ_PLUGIN_LOAD_ERROR        = 12,
    RUNAI_FILE_STREAMER_RESPONSE_GCS_NOT_SUPPORTED            = 13,
    RUNAI_FILE_STREAMER_RESPONSE_AZURE_BLOB_NOT_SUPPORTED     = 14,
    RUNAI_FILE_STREAMER_RESPONSE_FILE_TRUNCATED_ERROR         = 15,
    RUNAI_FILE_STREAMER_RESPONSE_TIMED_OUT                    = 16,
    RUNAI_FILE_STREAMER_RESPONSE_UNSUPPORTED_BACKEND_MIX      = 17,
    RUNAI_FILE_STREAMER_RESPONSE_CREDENTIALS_ALREADY_SET      = 18,
    RUNAI_FILE_STREAMER_RESPONSE_RETRYABLE_FILE_ACCESS_ERROR  = 19,
    RUNAI_FILE_STREAMER_RESPONSE_FS_STRATEGY_CONFLICT         = 20,
    RUNAI_FILE_STREAMER_RESPONSE_FS_STRATEGY_UNAVAILABLE      = 21,
    RUNAI_FILE_STREAMER_RESPONSE_FS_ASYNC_ENGINE_ERROR        = 22,
    RUNAI_FILE_STREAMER_RESPONSE_UNSUPPORTED_DEVICE_TYPE      = 23,
} RunaiFileStreamerResponseCode;

#endif // RUNAI_FILE_STREAMER_RESPONSE_CODE_H
