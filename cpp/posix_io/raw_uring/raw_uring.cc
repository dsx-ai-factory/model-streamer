#include "posix_io/raw_uring/raw_uring.h"

#include <liburing.h>       // for the register opcodes and the rsrc structs only
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

namespace runai::llm::streamer::posix_io
{

namespace
{

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif

#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif

} // namespace

bool raw_can_register_buffer(size_t bytesize)
{
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));

    const int fd = ::syscall(__NR_io_uring_setup, 8, &params);
    if (fd < 0)
    {
        return false;
    }

    void * buffer = nullptr;
    if (::posix_memalign(&buffer, 4096, bytesize) != 0)
    {
        ::close(fd);
        return false;
    }

    struct io_uring_rsrc_register reg;
    std::memset(&reg, 0, sizeof(reg));
    reg.nr = 1;
    reg.flags = IORING_RSRC_REGISTER_SPARSE;

    bool registered = false;
    if (::syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS2, &reg, sizeof(reg)) >= 0)
    {
        struct iovec iov;
        iov.iov_base = buffer;
        iov.iov_len = bytesize;

        struct io_uring_rsrc_update2 update;
        std::memset(&update, 0, sizeof(update));
        update.data = reinterpret_cast<__u64>(&iov);
        update.nr = 1;

        // The update returns the number of slots filled, so anything but a negative errno is a yes.
        registered = ::syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS_UPDATE, &update,
                               sizeof(update)) >= 0;
    }

    // The table goes with the ring, so closing it is the whole cleanup.
    ::close(fd);
    ::free(buffer);
    return registered;
}

} // namespace runai::llm::streamer::posix_io
