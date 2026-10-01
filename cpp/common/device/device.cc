#include "common/device/device.h"

namespace runai::llm::streamer::common
{

bool operator==(const Device & left, const Device & right)
{
    // The ordinal is not compared for the host, where it means nothing - otherwise two host devices
    // built differently would differ.
    return left.type == right.type && (left.is_host() || left.id == right.id);
}

bool operator!=(const Device & left, const Device & right)
{
    return !(left == right);
}

bool operator<(const Device & left, const Device & right)
{
    if (left.type != right.type)
    {
        return left.type < right.type;
    }

    return left.is_host() ? false : left.id < right.id;
}

std::ostream & operator<<(std::ostream & os, DeviceType type)
{
    switch (type)
    {
    case DeviceType::Cpu:
        return os << "cpu";
    case DeviceType::Cuda:
        return os << "cuda";
    }
    return os << "device type " << static_cast<int>(type);
}

std::ostream & operator<<(std::ostream & os, const Device & device)
{
    os << device.type;
    if (!device.is_host())
    {
        os << ":" << device.id;
    }
    return os;
}

} // namespace runai::llm::streamer::common
