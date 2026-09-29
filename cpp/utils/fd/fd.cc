/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/fd/fd.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <map>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::utils
{

namespace
{

constexpr size_t CHUNK_SIZE = 16384;

struct Set
{
    Set()
    {
        FD_ZERO(&_set);
    }

    Set(const std::set<Fd const *> & fds) : Set()
    {
        for (auto fd : fds)
        {
            set(fd->fd());
        }
    }

    void set(int fd)
    {
        FD_SET(fd, &_set);
    }

    bool isset(int fd) const
    {
        return FD_ISSET(fd, &_set);
    }

    operator fd_set * ()
    {
        return &_set;
    }

 private:
    fd_set _set;
};

} // namespace

Fd::Fd(int fd) :
    _fd(fd)
{}

Fd::Fd(Fd && other) :
    _fd(other._fd)
{
    other._fd = -1;
}

Fd & Fd::operator=(Fd && other)
{
    // close the current fd if needed
    _close();

    // copy the attributes of the other fd
    _fd = other._fd;

    // mark the other one as moved
    other._fd = -1;

    return *this;
}

Fd::~Fd()
{
    try
    {
        _close();
    }
    catch (...)
    {}
}

int Fd::fd() const
{
    return _fd;
}

Fd::operator int() const
{
    return fd();
}

size_t Fd::read(size_t size, /* out */ void * data, Read mode, size_t chunk_size)
{
    size_t total = 0;

    while (total < size)
    {
        const auto count = _os_read(
            reinterpret_cast<void *>(reinterpret_cast<size_t>(data) + total),
            std::min(chunk_size, size - total));

        if (count == -1)
        {
            if (errno == EINTR)
            {
                LOG(SPAM) << "Received EINTR while reading from fd " << _fd;
                continue;
            }

            PASSERT(false) << "Failed reading from fd " << _fd;
        }

        if (count == 0)
        {
            if (mode == Read::Eof)
            {
                break;
            }

            PASSERT(false) << "Trying to read from EOF " << _fd;
        }

        total += count;

        if (mode == Read::UpTo)
        {
            break;
        }
    }

    if (mode == Read::Exactly)
    {
        ASSERT(total == size);
    }

    return total;
}

size_t Fd::read(size_t size, /* out */ void * data, Read mode)
{
    return read(size, data, mode, CHUNK_SIZE);
}

void Fd::write(void const * const data, size_t size, size_t chunk_size)
{
    size_t total = 0;

    while (total < size)
    {
        const auto written = _os_write(
            reinterpret_cast<const void *>(reinterpret_cast<size_t>(data) + total),
            std::min(chunk_size, size - total));

        if (written == -1)
        {
            if (errno == EINTR)
            {
                LOG(SPAM) << "Received EINTR while writing to fd " << _fd;
                continue;
            }

            PASSERT(false) << "Failed writing to fd " << _fd;
        }

        PASSERT(written) << "Failed writing to closed device " << _fd;

        total += written;
    }

    ASSERT(total == size);
}

void Fd::write(void const * const data, size_t size)
{
    write(data, size, CHUNK_SIZE);
}

ssize_t Fd::_os_read(void * buf, size_t count)
{
    return ::read(_fd, buf, count);
}

ssize_t Fd::_os_write(void const * buf, size_t count)
{
    return ::write(_fd, buf, count);
}

void Fd::_close()
{
    if (_fd != -1)
    {
        CHECK(::close(_fd) == 0) << "Failed closing fd " << _fd;
    }
}

size_t Fd::size() const
{
    struct stat st = {};
    PASSERT(::fstat(_fd, &st) != -1) << "cannot get file size";
    return st.st_size;
}

void Fd::seek(off_t offset) const
{
    PASSERT(static_cast<size_t>(offset) <= size()) << "trying to read offset after EOF";
    PASSERT(::lseek(_fd, offset, SEEK_SET) != -1) << "failed seeking file";
}

std::vector<uint8_t> Fd::read(size_t size, Read mode)
{
    auto buffer = std::vector<uint8_t>(size);
    const auto got = read(size, /* out */ reinterpret_cast<void *>(buffer.data()), mode);
    buffer.resize(got);
    return buffer;
}

void Fd::write(const std::vector<uint8_t> & data)
{
    write(data.data(), data.size());
}

std::vector<uint8_t> Fd::read(const std::string & path)
{
    Fd fd(::open(path.c_str(), O_RDONLY));
    ASSERT(fd.fd() != -1) << "failed to open '" << path << "'";

    return fd.read(fd.size(), Read::Exactly);
}

std::vector<uint8_t> Fd::read(const std::string & path, off_t offset, size_t bytes)
{
    Fd fd(::open(path.c_str(), O_RDONLY));
    ASSERT(fd.fd() != -1) << "failed to open '" << path << "'";

    fd.seek(offset);
    return fd.read(bytes, Read::Exactly);
}

void Fd::write(const std::string & path, const std::vector<uint8_t> & data, int flags, mode_t mode)
{
    Fd fd(::open(path.c_str(), flags, mode));
    ASSERT(fd != -1) << "Failed opening '" << path << "'";
    fd.write(data.data(), data.size());
}

int Fd::open_for_read(const std::string & path, int extra_flags)
{
    const int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | extra_flags);
    if (fd < 0)
    {
        return -1;
    }

    struct stat st;
    if (::fstat(fd, &st) != 0)
    {
        const int error = errno;
        ::close(fd);
        errno = error;
        return -1;
    }

    if (S_ISREG(st.st_mode) || S_ISBLK(st.st_mode) || S_ISCHR(st.st_mode))
    {
        return fd;
    }

    // Named here rather than left to the caller: "Invalid argument" on an open that succeeded reads as
    // a bug in us, and the path alone does not say what it is.
    LOG(ERROR) << path << " is a "
               << (S_ISDIR(st.st_mode) ? "directory" : S_ISFIFO(st.st_mode) ? "FIFO"
                                                     : S_ISSOCK(st.st_mode) ? "socket" : "file type")
               << ", which cannot be read at an offset";

    ::close(fd);
    errno = EINVAL;
    return -1;
}

bool Fd::exists(const std::string & path)
{
    return ::access(path.c_str(), F_OK) != -1;
}


} // namespace runai::llm::streamer::utils
