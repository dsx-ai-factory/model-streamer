#ifndef NV_FILE_STREAMER_DEVICE_H
#define NV_FILE_STREAMER_DEVICE_H

// Plain C that also compiles as C++, because this header ships in the SDK tarball and a C program
// must be able to include it.

typedef enum NvFileStreamerDeviceType
{
    NV_FILE_STREAMER_DEVICE_CPU  = 0,
    NV_FILE_STREAMER_DEVICE_CUDA = 1,
} NvFileStreamerDeviceType;

typedef struct NvFileStreamerDevice
{
    // CPU is zero, so a zeroed struct asks for the host.
    NvFileStreamerDeviceType type;

    // The CUDA ordinal, typed as CUdevice is. Ignored when type is NV_FILE_STREAMER_DEVICE_CPU.
    int id;
} NvFileStreamerDevice;

#endif // NV_FILE_STREAMER_DEVICE_H
