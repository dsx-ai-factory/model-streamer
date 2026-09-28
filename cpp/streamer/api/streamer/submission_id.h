#ifndef RUNAI_FILE_STREAMER_SUBMISSION_ID_H
#define RUNAI_FILE_STREAMER_SUBMISSION_ID_H

// Plain C that also compiles as C++, because this header ships in the SDK tarball and a C program
// must be able to include it.

#include <stdint.h>

// Identifier for one submission: assigned when the read is submitted, stamped on every response, and
// echoed back so a shared responder can be demuxed across concurrent submissions. An opaque token -
// 64-bit so the id space is effectively unbounded, and 0 is reserved as the "none" value.
typedef uint64_t RunaiFileStreamerSubmissionId;

#endif // RUNAI_FILE_STREAMER_SUBMISSION_ID_H
