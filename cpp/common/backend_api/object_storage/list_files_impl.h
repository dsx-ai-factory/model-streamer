/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "common/backend_api/object_storage/object_storage.h"
#include "common/response/response.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::common::backend_api
{

namespace
{

ObjectFileEntry_t* make_entries(
    const std::vector<std::pair<std::string, size_t>>& results)
{
    size_t path_bytes = 0;
    for (const auto& r : results)
        path_bytes += r.first.size() + 1;

    const size_t entries_bytes = results.size() * sizeof(ObjectFileEntry_t);

    auto* buf       = new char[entries_bytes + path_bytes];
    auto* entries   = reinterpret_cast<ObjectFileEntry_t*>(buf);
    char* path_area = buf + entries_bytes;

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& path_str = results[i].first;
        entries[i].path = path_area;
        path_str.copy(path_area, path_str.size());
        path_area[path_str.size()] = '\0';
        path_area += path_str.size() + 1;
        entries[i].size = results[i].second;
    }

    return entries;
}

} // anonymous namespace

template<typename ClientT>
ResponseCode_t impl_obj_list_files(
    ObjectClientHandle_t client_handle,
    const char* prefix,
    int is_recursive,
    ObjectFileEntry_t** out_entries,
    unsigned* out_num_entries)
{
    if (!client_handle || !prefix || !out_entries || !out_num_entries)
    {
        return common::ResponseCode::InvalidParameterError;
    }

    try
    {
        auto* client = static_cast<ClientT*>(client_handle);
        std::vector<std::pair<std::string, size_t>> results;

        const auto ret = client->list_files(prefix, is_recursive, results);
        if (ret != common::ResponseCode::Success)
        {
            return ret;
        }

        *out_num_entries = static_cast<unsigned>(results.size());

        if (results.empty())
        {
            *out_entries = nullptr;
            return common::ResponseCode::Success;
        }

        *out_entries = make_entries(results);
        return common::ResponseCode::Success;
    }
    catch (const std::exception& e)
    {
        LOG(ERROR) << "Failed to list files: " << e.what();
    }
    return common::ResponseCode::UnknownError;
}

inline void impl_obj_free_file_list(ObjectFileEntry_t* entries, unsigned /* num_entries */)
{
    delete[] reinterpret_cast<char*>(entries);
}

} // namespace runai::llm::streamer::common::backend_api
