/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/completion_mapper/completion_mapper.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <string>
#include <vector>

namespace runai::llm::streamer::posix_io
{

namespace
{

FileRef buffered() { return FileRef{ 7, false }; }
FileRef direct()   { return FileRef{ 7, true  }; }

} // namespace

// A byte count, including zero. Whether zero means EOF depends on what was asked and what has already
// arrived, which only the caller knows.
TEST(CompletionMapper, Non_Negative_Is_Success)
{
    EXPECT_EQ(map_completion(4096, buffered()), common::ResponseCode::Success);
    EXPECT_EQ(map_completion(1, buffered()), common::ResponseCode::Success);
    EXPECT_EQ(map_completion(0, buffered()), common::ResponseCode::Success);
    EXPECT_EQ(map_completion(0, direct()), common::ResponseCode::Success);
}

// Attributable to this file. Other files, and other submissions, carry on.
//
// These reach FileAccessError through the DEFAULT now, not through a list of their own. They are kept
// as a test because they are the errnos a real disk or network share produces, and they must stay
// per-file however the mapping is written.
TEST(CompletionMapper, Storage_Errors_Are_Per_File)
{
    for (const long err : { EIO, ENXIO, ESTALE, ETIMEDOUT, ECONNRESET, EREMOTEIO })
    {
        EXPECT_EQ(map_completion(-err, buffered()), common::ResponseCode::FileAccessError) << "errno " << err;
        EXPECT_EQ(map_completion(-err, direct()), common::ResponseCode::FileAccessError) << "errno " << err;
    }
}

// Teardown, not a storage fault. Reporting a cancelled read as FileAccessError sends an operator to
// investigate the wrong system.
TEST(CompletionMapper, Cancelled_Is_Finished_Not_File_Access)
{
    EXPECT_EQ(map_completion(-ECANCELED, buffered()), common::ResponseCode::FinishedError);
    EXPECT_EQ(map_completion(-ECANCELED, direct()), common::ResponseCode::FinishedError);
}

// EINVAL means our alignment contract broke - but only on a direct fd. On a buffered one it means
// something else and must not be reported as an internal bug.
TEST(CompletionMapper, Einval_Is_Mode_Aware)
{
    EXPECT_TRUE(is_internal_error(-EINVAL, direct()));
    EXPECT_FALSE(is_internal_error(-EINVAL, buffered()));

    // Buffered EINVAL falls through to the default branch, which is FileAccessError. On a buffered fd
    // EINVAL is not about alignment, so there is no reason to call it our bug.
    EXPECT_EQ(map_completion(-EINVAL, buffered()), common::ResponseCode::FileAccessError);
}

// The engine never opens or closes anything, so a bad fd can only be our own bookkeeping.
TEST(CompletionMapper, Our_Bugs_Are_Not_Storage_Errors)
{
    for (const auto & file : { buffered(), direct() })
    {
        EXPECT_TRUE(is_internal_error(-EFAULT, file));
        EXPECT_TRUE(is_internal_error(-EBADF, file));

        EXPECT_FALSE(is_internal_error(-EIO, file));
        EXPECT_FALSE(is_internal_error(-ECANCELED, file));
        EXPECT_FALSE(is_internal_error(4096, file));
    }
}

// The mapper must RETURN for an internal error, not throw or abort: wait_for_completions is declared
// to return a code, and a library inside a serving stack must not take its host down. Testing the
// predicate alone missed this - the mapper asserted here, and ASSERT is fatal in every build.
TEST(CompletionMapper, Internal_Errors_Return_Unknown_Rather_Than_Throwing)
{
    EXPECT_EQ(map_completion(-EFAULT, buffered()), common::ResponseCode::UnknownError);
    EXPECT_EQ(map_completion(-EBADF, buffered()), common::ResponseCode::UnknownError);
    EXPECT_EQ(map_completion(-EFAULT, direct()), common::ResponseCode::UnknownError);
    EXPECT_EQ(map_completion(-EINVAL, direct()), common::ResponseCode::UnknownError);

    EXPECT_NO_THROW(map_completion(-EINVAL, direct()));
    EXPECT_NO_THROW(map_completion(-EBADF, direct()));
}

// An errno with no row of its own is FileAccessError, so one file's failure does not end the whole
// submission.
//
// This REVERSES an earlier decision, which sent unrecognised errnos to UnknownError on the grounds
// that guessing "the storage failed" was the wrong direction to be wrong in. That argument weighed
// only the chance of guessing wrong, and not what each wrong guess costs.
//
// UnknownError tells the caller to abort everything. So calling a file problem UnknownError ends a
// whole model load, while calling an internal bug a file problem costs one range and some diagnostic
// quality. The errno is logged either way.
//
// EISDIR is the case that proved it. Reading a directory returns errno 21, which was on no list, and
// the whole submission was abandoned. The synchronous reader has always reported it per file.
TEST(CompletionMapper, Unrecognised_Errno_Is_Attributed_To_The_File)
{
    EXPECT_EQ(map_completion(-EISDIR, buffered()), common::ResponseCode::FileAccessError);
    EXPECT_EQ(map_completion(-ENOSPC, buffered()), common::ResponseCode::FileAccessError);
    EXPECT_EQ(map_completion(-EPERM, buffered()), common::ResponseCode::FileAccessError);
}

// UnknownError must be EARNED. Only the results we can prove are ours get it, and nothing else does.
TEST(CompletionMapper, Only_Our_Own_Bugs_Are_Unknown_Errors)
{
    EXPECT_EQ(map_completion(-EFAULT, buffered()), common::ResponseCode::UnknownError);
    EXPECT_EQ(map_completion(-EBADF, buffered()), common::ResponseCode::UnknownError);
    EXPECT_EQ(map_completion(-EINVAL, direct()), common::ResponseCode::UnknownError);

    // Everything else, including errnos nobody listed anywhere.
    for (const long err : { EISDIR, ENOSPC, EPERM, EACCES, ENOTDIR, ELOOP, EOVERFLOW })
    {
        EXPECT_EQ(map_completion(-err, buffered()), common::ResponseCode::FileAccessError) << "errno " << err;
        EXPECT_EQ(map_completion(-err, direct()), common::ResponseCode::FileAccessError) << "errno " << err;
    }
}

// An interrupted read has no row here - is_transient() answers that separately. Asserted so that
// adding a row is a conscious act.
TEST(CompletionMapper, Interrupted_Reads_Have_No_Row_Here)
{
    EXPECT_EQ(map_completion(-EINTR, buffered()), common::ResponseCode::FileAccessError);
    EXPECT_EQ(map_completion(-512, buffered()), common::ResponseCode::FileAccessError);
}

// --- is_transient ------------------------------------------------------------------------------

// The kernel's restart codes, and EINTR, all mean the read did not happen.
TEST(CompletionMapper, Interrupted_Results_Are_Transient)
{
    for (const long err : { 512L, 513L, 514L, 516L, static_cast<long>(EINTR) })
    {
        EXPECT_TRUE(is_transient(-err)) << "errno " << err;
    }
}

// Storage failures and our own bugs are not retried.
//
// EAGAIN is in this list: as a completion it would need a non-blocking read, which we never ask for.
TEST(CompletionMapper, Failures_Are_Not_Transient)
{
    for (const long err : { EIO, EISDIR, ENOSPC, EFAULT, EBADF, EINVAL, ECANCELED, EAGAIN })
    {
        EXPECT_FALSE(is_transient(-err)) << "errno " << err;
    }
}

// Being in the kernel's private range is not enough - only the four that say "do it again" count.
TEST(CompletionMapper, Other_Kernel_Internal_Codes_Are_Not_Transient)
{
    EXPECT_FALSE(is_transient(-515));   // ENOIOCTLCMD, between two restart codes
    EXPECT_FALSE(is_transient(-518));   // EOPENSTALE
    EXPECT_FALSE(is_transient(-528));   // EJUKEBOX
    EXPECT_FALSE(is_transient(-529));   // EIOCBQUEUED
}

// Bytes transferred are never a retry, including a short read - the caller re-stages that from its
// cursor, and zero is the caller's question too.
TEST(CompletionMapper, Transferred_Bytes_Are_Not_Transient)
{
    EXPECT_FALSE(is_transient(0));
    EXPECT_FALSE(is_transient(4096));
}

}; // namespace runai::llm::streamer::posix_io
