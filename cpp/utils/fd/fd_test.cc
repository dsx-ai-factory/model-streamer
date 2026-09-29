/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/fd/fd.h"

#include <gtest/gtest.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>

#include <string>
#include <utility>
#include <vector>

#include "utils/logging/logging.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::utils
{

namespace
{

struct Helper : Fd
{
    Helper(const std::string & path, int mode) :
        Fd(::open(path.c_str(), mode, 0777))
    {
        ASSERT(_fd != -1) << "Failed opening '" << path << "'";
    }
};

struct Invalid : Fd
{
    Invalid() : Fd(random::number<int>(100, 200))
    {}
};

static bool is_fd_open(const std::string & name)
{
    const int status = ::system("ls -la /proc/self/fd > ./tmp.txt");

    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), EXIT_SUCCESS);

    const auto buffer = Fd::read("./tmp.txt");
    const auto str = std::string(buffer.begin(), buffer.end());
    return (str.find(name) != std::string::npos);
}

} // namespace

TEST(Creation, Empty)
{
    EXPECT_EQ(Fd(), -1);
}

TEST(Creation, Default)
{
    const auto temp = temp::File();

    bool is_open_;
    EXPECT_NO_THROW(is_fd_open(temp.name));
    ASSERT_FALSE(is_open_);

    const auto helper = Helper(temp.path, O_RDONLY);

    ASSERT_TRUE(is_fd_open(temp.name));

    {
        const Fd fd(helper.fd());

        EXPECT_TRUE(is_fd_open(temp.name));
    }

    EXPECT_FALSE(is_fd_open(temp.name));
}

TEST(Creation, RValue)
{
    const auto temp = temp::File();

    ASSERT_FALSE(is_fd_open(temp.name));

    auto helper = Helper(temp.path, O_RDONLY);

    ASSERT_TRUE(is_fd_open(temp.name));

    {
        auto fd = Fd(std::move(helper));

        EXPECT_TRUE(is_fd_open(temp.name));
    }

    EXPECT_FALSE(is_fd_open(temp.name));
}

TEST(Creation, Move)
{
    const auto temp_a = temp::File();
    const auto temp_b = temp::File();

    auto helper_a = Helper(temp_a.path, O_RDONLY);
    auto helper_b = Helper(temp_b.path, O_RDONLY);

    // they both should exist
    ASSERT_TRUE(is_fd_open(temp_a.name));
    ASSERT_TRUE(is_fd_open(temp_b.name));

    {
        Fd fd_a(helper_a.fd());
        Fd fd_b(helper_b.fd());

        // they both should still exist
        ASSERT_TRUE(is_fd_open(temp_a.name));
        ASSERT_TRUE(is_fd_open(temp_b.name));

        fd_b = std::move(fd_a);

        // `a` should still exist
        EXPECT_TRUE(is_fd_open(temp_a.name));

        // `b` should exist
        EXPECT_FALSE(is_fd_open(temp_b.name));
    }

    EXPECT_FALSE(is_fd_open(temp_a.name));
    EXPECT_FALSE(is_fd_open(temp_b.name));
}

TEST(Creation, Move_Empty)
{
    temp::File temp;

    Helper helper(temp.path, O_RDONLY);

    auto fd = Fd(-1);

    ASSERT_NE(helper, -1);
    ASSERT_EQ(fd, -1);

    fd = std::move(helper);

    EXPECT_EQ(helper, -1);
    EXPECT_NE(fd, -1);
}

TEST(Read_Write, Buffer)
{
    temp::Path path;

    const auto data = random::buffer();

    {
        Helper writer(path.path, O_WRONLY | O_CREAT);
        writer.write(data);
    }

    {
        Helper reader(path.path, O_RDONLY);
        const auto read = reader.read(data.size());

        EXPECT_EQ(data, read);
    }
}

TEST(Read, Failure)
{
    temp::Path path;

    const auto data = random::buffer();

    {
        Helper writer(path.path, O_WRONLY | O_CREAT);
        writer.write(data);
    }

    {
        Helper reader(path.path, O_RDONLY);
        EXPECT_THROW(reader.read(data.size() + 1), std::exception);
    }
}

TEST(Read, UpTo)
{
    const auto len = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(len);
    const auto file = utils::temp::File(data);

    for (auto size : { len, len + 1, len * 2, utils::random::number(len * 2, len * 10) })
    {
        auto fd = Helper(file.path, O_RDONLY);

        const auto read = fd.read(size, Fd::Read::UpTo);

        EXPECT_EQ(read.size(), data.size());
        EXPECT_EQ(read, data);
    }
}

TEST(Read, UpTo_Eof)
{
    const auto len = random::number(100, 1000);
    const auto data = random::buffer(len);
    const auto file = temp::File(data);

    for (auto size : { len, len + 1, len * 2, random::number(len * 2, len * 10) })
    {
        auto fd = Helper(file.path, O_RDONLY);

        const auto read = fd.read(size * 2, Fd::Read::Eof);

        EXPECT_EQ(read.size(), data.size());
        EXPECT_EQ(read, data);
    }
}

TEST(Read, UpTo_Eof_Empty)
{
    const auto len = random::number(100, 1000);
    const auto data = std::vector<uint8_t>{};
    const auto file = temp::File(data);

    for (auto size : { len, len + 1, len * 2, random::number(len * 2, len * 10) })
    {
        auto fd = Helper(file.path, O_RDONLY);

        const auto read = fd.read(size * 2, Fd::Read::Eof);
        EXPECT_EQ(read.size(), 0);
    }
}

TEST(Seek, Sanity)
{
    const auto len = random::number(100, 1000);
    const auto data = random::buffer(len);
    const auto file = temp::File(data);
    ASSERT_FALSE(is_fd_open(file.name));

    auto fd = Helper(file.path, O_RDONLY);
    ASSERT_TRUE(is_fd_open(file.name));

    const unsigned index = random::number(10);
    const off_t offset = index * sizeof(int);

    fd.seek(offset);

    const auto buffer = fd.read(sizeof(int));
    EXPECT_EQ(buffer.at(0), data.at(offset));
}

TEST(Seek, InputValdation)
{
    const auto len = random::number(100, 1000);
    const auto data = random::buffer(len);
    const auto file = temp::File(data);
    ASSERT_FALSE(is_fd_open(file.name));

    auto fd = Helper(file.path, O_RDONLY);
    ASSERT_TRUE(is_fd_open(file.name));

    const unsigned offset = random::number(data.size() + 1, data.size() * 10);
    EXPECT_THROW(fd.seek(offset), std::exception);
}

TEST(Size, Sanity)
{
    const auto data = random::buffer();
    const auto temp = temp::File(data);
    ASSERT_FALSE(is_fd_open(temp.name));

    const auto helper = Helper(temp.path, O_RDONLY);
    ASSERT_TRUE(is_fd_open(temp.name));

    Fd fd(helper.fd());
    EXPECT_TRUE(is_fd_open(temp.name));

    EXPECT_EQ(fd.size(), data.size());
}

// A FIFO must not be opened, because the OPEN is what hangs.
//
// Measured on Linux 6.8: open(fifo, O_RDONLY) does not return until something opens the write end. In
// the streamer that call is made by the thread serving a whole mount, so one such path stops every
// read on it, permanently and with no error anywhere.
//
// This is the test that made the branch real. Every other file type this refuses can be reasoned
// about; a FIFO can simply be created.
TEST(OpenForRead, A_Fifo_Is_Refused_Rather_Than_Blocking_Forever)
{
    const std::string path = "/tmp/runai_fifo_" + random::string();
    ASSERT_EQ(::mkfifo(path.c_str(), 0600), 0);

    // No writer exists, and none ever will - so a blocking open here would never return and this test
    // would hang rather than fail. That is the behaviour being prevented.
    const int fd = Fd::open_for_read(path);

    EXPECT_EQ(fd, -1) << "a FIFO cannot be read at an offset and must be refused";
    EXPECT_EQ(errno, EINVAL);

    if (fd >= 0)
    {
        ::close(fd);
    }
    ::unlink(path.c_str());
}

TEST(OpenForRead, A_Directory_Is_Refused)
{
    // open(O_RDONLY) SUCCEEDS on a directory, so nothing before the read notices. Each reader then
    // finds out somewhere different - libaio refuses the whole io_submit with EINVAL, io_uring
    // completes the read with -EINVAL, and pread returns EISDIR. Refusing here makes all three say the
    // same thing.
    const int fd = Fd::open_for_read("/tmp");

    EXPECT_EQ(fd, -1);
    EXPECT_EQ(errno, EINVAL);

    if (fd >= 0)
    {
        ::close(fd);
    }
}

TEST(OpenForRead, A_Regular_File_Is_Opened_And_Reads_Normally)
{
    const auto data = random::buffer(1024);
    const std::string path = "/tmp/runai_open_for_read_" + random::string();

    {
        Fd out(::open(path.c_str(), O_CREAT | O_WRONLY, 0600));
        ASSERT_NE(out.fd(), -1);
        out.write(data);
    }

    const int fd = Fd::open_for_read(path);
    ASSERT_GE(fd, 0);

    Fd owned(fd);
    const auto got = owned.read(data.size(), Fd::Read::Exactly);

    // O_NONBLOCK is set on this descriptor and must not change what a read of a regular file does.
    EXPECT_EQ(got, data);

    ::unlink(path.c_str());
}

TEST(OpenForRead, A_Missing_Path_Reports_Its_Own_Errno)
{
    const int fd = Fd::open_for_read("/tmp/runai_definitely_missing_" + random::string());

    EXPECT_EQ(fd, -1);
    EXPECT_EQ(errno, ENOENT) << "a missing file must not be reported as a refused type";
}

} // namespace runai::llm::streamer::utils
