/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/completion_mapper/completion_mapper.h"

#include <cerrno>
#include <sstream>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

} // namespace

bool is_internal_error(long res, const FileRef & file)
{
    if (res >= 0)
    {
        return false;
    }

    const long err = -res;

    // EFAULT means we handed the kernel a bad address; EBADF means the fd we passed was not open.
    // Neither can be the storage's doing - the engine never opens or closes anything, so an invalid fd
    // can only come from the caller's own bookkeeping.
    if (err == EFAULT || err == EBADF)
    {
        return true;
    }

    // EINVAL is diagnostic only on a DIRECT fd, where it means the alignment contract broke. On a
    // buffered fd it means something else entirely and belongs in the default branch.
    return err == EINVAL && file.direct;
}

common::ResponseCode map_completion(long res, const FileRef & file)
{
    if (res >= 0)
    {
        // Including res == 0. Whether that is EOF depends on what the caller asked for and how much
        // has already arrived, which only the caller knows - so it decides, not this.
        return common::ResponseCode::Success;
    }

    const long err = -res;

    if (err == ECANCELED)
    {
        return common::ResponseCode::FinishedError;
    }

    if (is_internal_error(res, file))
    {
        // Loud, but NOT fatal and NOT an exception.
        //
        // The design says "assert in debug, UnknownError in release". This repo's ASSERT is fatal in
        // every build - no NDEBUG guard, see logging.h - so using it here would throw out of
        // wait_for_completions, which is declared to return a code. A library inside a serving stack
        // must not take down its host either. So: log at ERROR and return.
        //
        // Never FileAccessError. That is the one outcome that sends an operator to investigate the
        // storage when the fault is ours.
        LOG(ERROR) << "Internal error: errno " << err << " on fd " << file.fd
                   << (file.direct ? " (direct)" : " (buffered)")
                   << " - an invariant of ours broke, not the storage";
        return common::ResponseCode::UnknownError;
    }

    // Everything else is this FILE's failure, and the caller may carry on with the rest.
    //
    // The default runs THIS way round on purpose. UnknownError tells the caller to abort the whole
    // submission, so it must need evidence: the three cases above are the ones we can prove are ours.
    // An errno we do not recognise is far more likely to be about the file than about our own
    // bookkeeping.
    //
    // The two directions cost very different amounts when they are wrong. Calling an internal bug a
    // file failure loses some diagnostic quality and one range. Calling a file failure UnknownError
    // ends a whole model load because one file could not be read.
    //
    // The synchronous reader already decided this, and for the same reason - see the comment in
    // file.cc. Both readers must agree, because which one served a request is meant to be invisible
    // to the caller.
    //
    // The errno is still logged, so nothing is lost by not naming it here. An allowlist of storage
    // errnos used to sit in this place; it could not be complete, and EISDIR was one it missed.
    LOG(ERROR) << "Read failed with errno " << err << " on fd " << file.fd
               << (file.direct ? " (direct)" : " (buffered)") << " - reporting it against this file";
    return common::ResponseCode::FileAccessError;
}

}; // namespace runai::llm::streamer::posix_io
