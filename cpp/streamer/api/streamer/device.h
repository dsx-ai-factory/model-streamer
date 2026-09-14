#ifndef RUNAI_FILE_STREAMER_DEVICE_H
#define RUNAI_FILE_STREAMER_DEVICE_H

// Plain C that also compiles as C++, because this header ships in the SDK tarball and a C program
// must be able to include it.

typedef enum RunaiFileStreamerDeviceType
{
    RUNAI_FILE_STREAMER_DEVICE_CPU  = 0,
    RUNAI_FILE_STREAMER_DEVICE_CUDA = 1,
} RunaiFileStreamerDeviceType;

typedef struct RunaiFileStreamerDevice
{
    // CPU is zero, so a zeroed struct asks for the host.
    RunaiFileStreamerDeviceType type;

    // The CUDA ordinal, typed as CUdevice is. Ignored when type is RUNAI_FILE_STREAMER_DEVICE_CPU.
    int id;
} RunaiFileStreamerDevice;

#endif // RUNAI_FILE_STREAMER_DEVICE_H
