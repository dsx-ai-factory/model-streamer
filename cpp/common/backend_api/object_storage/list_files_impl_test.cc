/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/backend_api/object_storage/list_files_impl.h"

#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/response_code/response_code.h"

namespace runai::llm::streamer::common::backend_api
{

namespace
{

// Minimal stand-in for a backend client: records the arguments it was called with
// and returns a caller-configured result / error / exception.
struct FakeClient
{
    std::vector<std::pair<std::string, size_t>> to_return;
    common::ResponseCode ret = common::ResponseCode::Success;
    bool should_throw = false;

    int seen_is_recursive = -1;
    std::string seen_prefix;

    common::ResponseCode list_files(const char* prefix, int is_recursive, std::vector<std::pair<std::string, size_t>>& results)
    {
        seen_prefix = prefix;
        seen_is_recursive = is_recursive;
        if (should_throw)
        {
            throw std::runtime_error("list_files failed");
        }
        results = to_return;
        return ret;
    }
};

// Sentinel pointer used to prove out_entries is left untouched on error paths
ObjectFileEntry_t* const sentinel = reinterpret_cast<ObjectFileEntry_t*>(0xdead);

} // namespace

TEST(ListFilesImpl, SuccessPacksEntries)
{
    FakeClient client;
    client.to_return = {
        {"s3://bucket/a", 10},
        {"s3://bucket/longer/path/b", 20},
        {"", 0}, // empty path exercises the path_area offset arithmetic
    };

    ObjectFileEntry_t* entries = nullptr;
    unsigned num_entries = 0;

    const auto ret = impl_obj_list_files<FakeClient>(&client, "s3://bucket/", 1, &entries, &num_entries);

    EXPECT_EQ(ret, common::ResponseCode::Success);
    ASSERT_EQ(num_entries, 3u);
    ASSERT_NE(entries, nullptr);

    EXPECT_STREQ(entries[0].path, "s3://bucket/a");
    EXPECT_EQ(entries[0].size, 10u);
    EXPECT_STREQ(entries[1].path, "s3://bucket/longer/path/b");
    EXPECT_EQ(entries[1].size, 20u);
    EXPECT_STREQ(entries[2].path, "");
    EXPECT_EQ(entries[2].size, 0u);

    // each path is independently null-terminated and lives at a distinct offset
    EXPECT_NE(entries[0].path, entries[1].path);
    EXPECT_EQ(std::strlen(entries[0].path), std::string("s3://bucket/a").size());

    impl_obj_free_file_list(entries, num_entries);
}

TEST(ListFilesImpl, ForwardsPrefixAndIsRecursive)
{
    FakeClient client;
    ObjectFileEntry_t* entries = nullptr;
    unsigned num_entries = 0;

    impl_obj_list_files<FakeClient>(&client, "s3://bucket/p", 0, &entries, &num_entries);
    EXPECT_EQ(client.seen_prefix, "s3://bucket/p");
    EXPECT_EQ(client.seen_is_recursive, 0);

    impl_obj_list_files<FakeClient>(&client, "s3://bucket/p", 1, &entries, &num_entries);
    EXPECT_EQ(client.seen_is_recursive, 1);
}

TEST(ListFilesImpl, EmptyResultsYieldNullptr)
{
    FakeClient client; // empty to_return, Success

    ObjectFileEntry_t* entries = sentinel;
    unsigned num_entries = 123;

    const auto ret = impl_obj_list_files<FakeClient>(&client, "s3://bucket/", 1, &entries, &num_entries);

    EXPECT_EQ(ret, common::ResponseCode::Success);
    EXPECT_EQ(num_entries, 0u);
    EXPECT_EQ(entries, nullptr);

    // freeing an empty listing is a safe no-op (delete[] nullptr)
    impl_obj_free_file_list(entries, num_entries);
}

TEST(ListFilesImpl, ClientErrorDoesNotAllocate)
{
    FakeClient client;
    client.ret = common::ResponseCode::FileAccessError;
    client.to_return = {{"s3://bucket/a", 10}}; // would-be entries must not be allocated

    ObjectFileEntry_t* entries = sentinel;
    unsigned num_entries = 123;

    const auto ret = impl_obj_list_files<FakeClient>(&client, "s3://bucket/", 1, &entries, &num_entries);

    EXPECT_EQ(ret, common::ResponseCode::FileAccessError);
    EXPECT_EQ(entries, sentinel); // out_entries untouched -> caller has nothing to free
}

TEST(ListFilesImpl, ClientThrowsReturnsUnknownError)
{
    FakeClient client;
    client.should_throw = true;

    ObjectFileEntry_t* entries = sentinel;
    unsigned num_entries = 123;

    const auto ret = impl_obj_list_files<FakeClient>(&client, "s3://bucket/", 1, &entries, &num_entries);

    EXPECT_EQ(ret, common::ResponseCode::UnknownError);
    EXPECT_EQ(entries, sentinel);
}

TEST(ListFilesImpl, NullArgumentsRejected)
{
    FakeClient client;
    ObjectFileEntry_t* entries = nullptr;
    unsigned num_entries = 0;

    EXPECT_EQ(impl_obj_list_files<FakeClient>(nullptr, "s3://bucket/", 1, &entries, &num_entries), common::ResponseCode::InvalidParameterError);
    EXPECT_EQ(impl_obj_list_files<FakeClient>(&client, nullptr, 1, &entries, &num_entries), common::ResponseCode::InvalidParameterError);
    EXPECT_EQ(impl_obj_list_files<FakeClient>(&client, "s3://bucket/", 1, nullptr, &num_entries), common::ResponseCode::InvalidParameterError);
    EXPECT_EQ(impl_obj_list_files<FakeClient>(&client, "s3://bucket/", 1, &entries, nullptr), common::ResponseCode::InvalidParameterError);
}

} // namespace runai::llm::streamer::common::backend_api
