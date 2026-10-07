/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <ostream>

#include "streamer/response_code.h"

namespace runai::llm::streamer::common
{

// The C++ name for each published code. Every value comes from <streamer/response_code.h>, so the two
// cannot drift: a code is added there first, and named here.
enum class ResponseCode : int
{
    Success                  = RUNAI_FILE_STREAMER_RESPONSE_SUCCESS,

    FinishedError            = RUNAI_FILE_STREAMER_RESPONSE_FINISHED_ERROR,
    FileAccessError          = RUNAI_FILE_STREAMER_RESPONSE_FILE_ACCESS_ERROR,
    EofError                 = RUNAI_FILE_STREAMER_RESPONSE_EOF_ERROR,
    S3NotSupported           = RUNAI_FILE_STREAMER_RESPONSE_S3_NOT_SUPPORTED,
    GlibcPrerequisite        = RUNAI_FILE_STREAMER_RESPONSE_GLIBC_PREREQUISITE,
    InsufficientFdLimit      = RUNAI_FILE_STREAMER_RESPONSE_INSUFFICIENT_FD_LIMIT,
    InvalidParameterError    = RUNAI_FILE_STREAMER_RESPONSE_INVALID_PARAMETER_ERROR,
    EmptyRequestError        = RUNAI_FILE_STREAMER_RESPONSE_EMPTY_REQUEST_ERROR,
    BusyError                = RUNAI_FILE_STREAMER_RESPONSE_BUSY_ERROR,
    CaFileNotFound           = RUNAI_FILE_STREAMER_RESPONSE_CA_FILE_NOT_FOUND,
    UnknownError             = RUNAI_FILE_STREAMER_RESPONSE_UNKNOWN_ERROR,
    ObjPluginLoadError       = RUNAI_FILE_STREAMER_RESPONSE_OBJ_PLUGIN_LOAD_ERROR,
    GCSNotSupported          = RUNAI_FILE_STREAMER_RESPONSE_GCS_NOT_SUPPORTED,
    AzureBlobNotSupported    = RUNAI_FILE_STREAMER_RESPONSE_AZURE_BLOB_NOT_SUPPORTED,
    FileTruncatedError       = RUNAI_FILE_STREAMER_RESPONSE_FILE_TRUNCATED_ERROR,
    TimedOut                 = RUNAI_FILE_STREAMER_RESPONSE_TIMED_OUT,
    UnsupportedBackendMix    = RUNAI_FILE_STREAMER_RESPONSE_UNSUPPORTED_BACKEND_MIX,
    CredentialsAlreadySet    = RUNAI_FILE_STREAMER_RESPONSE_CREDENTIALS_ALREADY_SET,
    RetryableFileAccessError = RUNAI_FILE_STREAMER_RESPONSE_RETRYABLE_FILE_ACCESS_ERROR,

    // Filesystem strategy problems. Two codes, because the operator has to do something different
    // for each one: set the value once, or add a candidate the host can serve.
    //
    // Both used to report UnsupportedBackendMix, whose message is about mixing S3, GCS and Azure.
    // That sent the reader to object storage for a problem that has nothing to do with it.
    FsStrategyConflict       = RUNAI_FILE_STREAMER_RESPONSE_FS_STRATEGY_CONFLICT,
    FsStrategyUnavailable    = RUNAI_FILE_STREAMER_RESPONSE_FS_STRATEGY_UNAVAILABLE,

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
    FsAsyncEngineError       = RUNAI_FILE_STREAMER_RESPONSE_FS_ASYNC_ENGINE_ERROR,

    // The submission named a device this build cannot serve.
    UnsupportedDeviceType    = RUNAI_FILE_STREAMER_RESPONSE_UNSUPPORTED_DEVICE_TYPE,

    // Device failures, one per remedy. NOT UnknownError: that code tells a caller to abort and
    // distrust every response it already has, and none of these mean that.
    //
    // DeviceUnavailable is an error, not a signal to read into host memory instead. A caller that
    // asked for a device destination gets no bytes if there is no device; choosing a host
    // destination belongs to the layer above.
    DeviceUnavailable        = RUNAI_FILE_STREAMER_RESPONSE_DEVICE_UNAVAILABLE,
    InvalidDevice            = RUNAI_FILE_STREAMER_RESPONSE_INVALID_DEVICE,
    DeviceOutOfMemory        = RUNAI_FILE_STREAMER_RESPONSE_DEVICE_OUT_OF_MEMORY,

    // The copy failed, so these ranges must not be trusted. The driver's own code and message are in
    // the log. SAFE to free or reuse the destination: whatever was enqueued has ended by the time this
    // is reported - which is what separates it from DeviceDriverError below.
    DeviceTransferError      = RUNAI_FILE_STREAMER_RESPONSE_DEVICE_TRANSFER_ERROR,

    // The driver would not answer. Reported for anything from a failed context bind to a stream that
    // could not be drained, so it is a broad code.
    //
    // TREAT THE DESTINATION AS STILL LIVE - do not free or reuse it. That is true of one producer: a
    // copy was enqueued and the stream could not be drained, so nothing can say when it ends. It is
    // over-cautious for the others, where the destination was never touched, but a caller cannot tell
    // them apart and the driver is already broken either way.
    //
    // A dedicated code would carry this better than widening this one. See the device notes.
    DeviceDriverError        = RUNAI_FILE_STREAMER_RESPONSE_DEVICE_DRIVER_ERROR,

    __Max,
};

const char * description(int response_code);

ResponseCode response_code_from(int value);

std::ostream & operator<<(std::ostream &, const ResponseCode &);

}; // namespace runai::llm::streamer::common
