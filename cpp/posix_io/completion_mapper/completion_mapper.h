/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>

#include "posix_io/alignment/alignment.h"
#include "posix_io/io_engine/io_engine.h"
#include "common/response_code/response_code.h"

namespace runai::llm::streamer::posix_io
{

// Turn an engine result into a common::ResponseCode.
//
// `res` is what the kernel reported for one read: bytes transferred when >= 0, a negative errno
// otherwise. io_uring and libaio both report this way, and both use this - one table, because the
// promise that "one file's error does not poison other submissions" has to hold identically for
// either engine, and two copies of it would drift apart unnoticed.
//
// Lives beside the engines, not above them: Completion::ret is a common::ResponseCode, so mapping is part of
// being an engine. This is the same layering object storage has, where each plugin maps its own SDK's
// errors behind the C ABI and the worker only ever sees a common::ResponseCode.
//
//   res >= 0                            Success - the CALLER still compares res against bytesize, and
//                                       re-stages the remainder if it is short. A short read is not an
//                                       error, and only the caller knows what it asked for.
//   -ECANCELED                          FinishedError - teardown, NOT a storage fault. Reporting it as
//                                       one sends an operator to investigate the wrong system.
//   -EFAULT -EBADF                      our bug: UnknownError
//   -EINVAL on a direct fd              our bug: the alignment contract broke. See below.
//   anything else                       FileAccessError - this file failed; others carry on. The raw
//                                       errno is logged.
//
// THE DEFAULT IS FileAccessError, and the three "our bug" rows are the exceptions. That direction is
// the point of this table.
//
// UnknownError tells the caller to abort the whole submission, so it must be earned. The three rows
// above are the results we can prove are ours. Any other errno is far more likely to describe the
// file than our own bookkeeping.
//
// An allowlist of storage errnos (EIO, ENXIO, ESTALE, ETIMEDOUT, ECONNRESET, EREMOTEIO) used to sit
// here instead, with everything else falling through to UnknownError. It could not be complete.
// EISDIR was one it missed, so reading a directory ended an entire model load.
//
// The synchronous reader made the same choice, with the same reasoning - see file.cc. Both readers
// must agree, because which one served a request is meant to be invisible to the caller.
//
// No -EINTR row on purpose. It is unreachable as a completion: regular-file reads sit in
// uninterruptible sleep, io-wq workers do not take signals the usual way, and libaio surfaces EINTR
// from io_getevents - the wait, not the result. The default branch covers it if it ever appears, which
// beats an untested branch claiming to handle it.
common::ResponseCode map_completion(long res, const FileRef & file);

// True when a result means "our bug" rather than "the storage failed".
//
// map_completion logs these at ERROR and returns UnknownError. It does not assert and does not throw:
// this repo's ASSERT is fatal in EVERY build, and throwing here would break wait_for_completions,
// which returns a code. Never FileAccessError - that sends an operator to the wrong system.
bool is_internal_error(long res, const FileRef & file);

}; // namespace runai::llm::streamer::posix_io
