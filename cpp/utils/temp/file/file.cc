/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/temp/file/file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <system_error>
#include <utility>

#include "utils/fd/fd.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::utils::temp
{

Path::Path(const std::string & dir, const std::string & name) :
    name(name),
    path(dir + (name.empty() ? "" : "/") + name)
{}

Path::Path(Path && other)
{
    *this = std::move(other);
}

Path & Path::operator=(Path && other)
{
    _delete();

    name = other.name;
    path = other.path;

    other.path = other.name = "";

    return *this;
}

Path::~Path()
{
    try
    {
        _delete();
    }
    catch (...)
    {}
}

bool Path::exists(const std::string & path)
{
    return ::access(path.c_str(), F_OK) != -1;
}

bool Path::is_directory(const std::string & path)
{
    struct stat stat = {};

    PASSERT(::stat(path.c_str(), &stat) == 0) << "Failed querying stats of '" << path << "'";

    return S_ISDIR(stat.st_mode);
}

void Path::_delete()
{
    if (path.empty() || !exists(path))
    {
        return;
    }

    ASSERT(!Path::is_directory(path)) << "Removing directory is not implemented";

    PASSERT(::unlink(path.c_str()) != -1) << "Failed removing file '" << path << "'";
}

Dir::Dir(const std::string & dir, const std::string & name) :
    name(name),
    path(dir + (name.empty() ? "" : "/") + name)
{
    // An empty name collapses path to dir, which would make _delete() recursively
    // remove the caller-supplied directory instead of an owned temp subdirectory
    ASSERT(!name.empty()) << "Temporary directory name must not be empty";
    std::filesystem::create_directories(path);
}

Dir::Dir(Dir && other)
{
    *this = std::move(other);
}

Dir & Dir::operator=(Dir && other)
{
    _delete();

    name = other.name;
    path = other.path;

    other.path = other.name = "";

    return *this;
}

Dir::~Dir()
{
    try
    {
        _delete();
    }
    catch (...)
    {}
}

void Dir::_delete()
{
    if (path.empty())
    {
        return;
    }

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec)
    {
        LOG(ERROR) << "Failed removing directory '" << path << "': " << ec.message();
    }
}

File::File(const std::string & dir, const std::string & name, const std::vector<uint8_t> & data) :
    _path(dir, name),
    name(_path.name),
    path(_path.path)
{
    utils::Fd::write(path, data, O_WRONLY | O_CREAT, 0777);
}

File::File(const std::vector<uint8_t> & data) : File(".", random::string(), data)
{}

} // namespace runai::llm::streamer::utils::temp
