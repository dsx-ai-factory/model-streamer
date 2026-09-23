// Read performance into DEVICE memory, and whether the shape of the destination matters.
//
// Rates are AGGREGATE - every device reads the whole file, so N devices move N times the bytes and
// the table reports the sum, with the per-device figure beside it. GB/s is decimal (1e9) and GiB/s
// is binary; both are printed because mixing them has already caused one wrong conclusion here.
//
// Two layouts, because they produce different work for the streamer:
//
//   one       every range writes into one large cuMemAlloc, tiled in order. The ranges are adjacent
//             in both the file and the destination, so the Assigner coalesces them into ONE transfer
//             and the reads are large.
//   many      every range has its own cuMemAlloc. Nothing is adjacent, so each range is its own
//             transfer - the layout a framework produces when it allocates per tensor.
//
// And 1, 2 and 4 devices, each with its own submission: a device is per submission, so N devices is
// N concurrent submissions through one streamer.
//
// Every arm is VERIFIED: the device memory is copied back and compared byte for byte. A fast wrong
// answer is the failure mode that matters here - a copy to the wrong offset costs nothing at run
// time and is invisible in a rate.
//
// ITERATIONS, on ONE streamer. The pinned pool is built once and reused for the life of the streamer,
// so the first submission pays for it and every later one does not. A real model is far larger than
// anything measured here and a real service serves request after request, so the number that matters
// is the STEADY one - the first is reported beside it to show what the pool costs.
//
// Run it where there is a driver. It reports and exits non-zero if any arm fails to verify.

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "device/cuda/cuda_device.h"
#include "device/cuda/cuda_lib.h"
#include "streamer/impl/streamer/streamer.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::impl
{

namespace
{

using CopyDtoH = CUresult (*)(void *, CUdeviceptr, size_t);

CopyDtoH read_back_fn()
{
    void * const handle = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    return handle == nullptr ? nullptr : reinterpret_cast<CopyDtoH>(::dlsym(handle, "cuMemcpyDtoH_v2"));
}

double seconds_since(const std::chrono::steady_clock::time_point & start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// One device's destinations: either a single allocation or one per range.
struct Destinations
{
    std::shared_ptr<device::Device> device;
    std::vector<void *> blocks;     // what has to be freed
    std::vector<char *> per_range;  // where each range goes

    void free_all()
    {
        for (auto * block : blocks)
        {
            device->device_free(block);
        }
        blocks.clear();
        per_range.clear();
    }
};

bool allocate(Destinations & out, const std::shared_ptr<device::Device> & device,
              bool single, size_t total, size_t range_bytesize, unsigned ranges)
{
    out.device = device;

    if (single)
    {
        void * block = nullptr;
        if (device->device_alloc(total, &block) != common::ResponseCode::Success)
        {
            return false;
        }
        out.blocks.push_back(block);
        for (unsigned i = 0; i < ranges; ++i)
        {
            out.per_range.push_back(static_cast<char *>(block) + i * range_bytesize);
        }
        return true;
    }

    for (unsigned i = 0; i < ranges; ++i)
    {
        void * block = nullptr;
        if (device->device_alloc(range_bytesize, &block) != common::ResponseCode::Success)
        {
            return false;
        }
        out.blocks.push_back(block);
        out.per_range.push_back(static_cast<char *>(block));
    }
    return true;
}

struct Arm
{
    std::string layout;
    unsigned devices = 0;
    bool single = false;
    double first = 0;        // GB/s on the first submission, which builds the pool
    double steady = 0;       // GB/s averaged over every later one
    bool verified = false;
    std::string note;
};

Arm run_arm(const std::shared_ptr<device::Backend> & backend, const std::string & path,
            const std::vector<uint8_t> & data, unsigned device_count, bool single,
            size_t range_bytesize, CopyDtoH read_back, unsigned iterations)
{
    Arm arm;
    arm.devices = device_count;
    arm.single = single;
    arm.layout = single ? "one buffer" : "many buffers";

    const size_t total = data.size();
    const unsigned ranges = static_cast<unsigned>(total / range_bytesize);

    std::vector<Destinations> destinations(device_count);
    for (unsigned d = 0; d < device_count; ++d)
    {
        std::shared_ptr<device::Device> device;
        if (backend->open_device(d, device) != common::ResponseCode::Success ||
            device->bind_thread() != common::ResponseCode::Success ||
            !allocate(destinations[d], device, single, total, range_bytesize, ranges))
        {
            arm.note = "could not allocate on device " + std::to_string(d);
            for (auto & destination : destinations) { destination.free_all(); }
            return arm;
        }
    }

    Streamer streamer;

    std::vector<std::vector<FileRanges>> requests(device_count);
    for (unsigned d = 0; d < device_count; ++d)
    {
        requests[d].resize(1);
        requests[d][0].path = path;
        for (unsigned i = 0; i < ranges; ++i)
        {
            requests[d][0].ranges.push_back(
                ReadRange{ i * range_bytesize, range_bytesize, destinations[d].per_range[i] });
        }
    }

    unsigned failures = 0;
    double steady_seconds = 0;

    // One round is one request per device, issued together and drained together - what a service does
    // when every rank asks at once. The streamer, and so the pool, lives across every round.
    for (unsigned round = 0; round < iterations; ++round)
    {
        const auto start = std::chrono::steady_clock::now();

        for (unsigned d = 0; d < device_count; ++d)
        {
            SubmissionId id = 0;
            const auto code = streamer.async_request(requests[d], common::Device::cuda(d), &id);
            if (code != common::ResponseCode::Success)
            {
                arm.note = "submission for device " + std::to_string(d) + " refused";
                for (auto & destination : destinations) { destination.free_all(); }
                return arm;
            }
        }

        for (unsigned i = 0; i < ranges * device_count; ++i)
        {
            bool done = false;
            if (streamer.response(120000, done).ret != common::ResponseCode::Success)
            {
                ++failures;
            }
        }

        const auto elapsed = seconds_since(start);
        const auto rate = (static_cast<double>(total) * device_count) / elapsed / 1e9;

        if (round == 0)
        {
            arm.first = rate;
        }
        else
        {
            steady_seconds += elapsed;
        }
    }

    if (iterations > 1)
    {
        arm.steady = (static_cast<double>(total) * device_count * (iterations - 1)) / steady_seconds / 1e9;
    }

    if (failures != 0)
    {
        arm.note = std::to_string(failures) + " ranges failed";
        for (auto & destination : destinations) { destination.free_all(); }
        return arm;
    }

    // Byte for byte, on every device. A copy landing at the wrong offset costs nothing at run time
    // and would otherwise show only as a good number.
    arm.verified = true;
    std::vector<char> landed(range_bytesize);
    for (unsigned d = 0; d < device_count && arm.verified; ++d)
    {
        destinations[d].device->bind_thread();
        for (unsigned i = 0; i < ranges; ++i)
        {
            if (read_back(landed.data(), reinterpret_cast<CUdeviceptr>(destinations[d].per_range[i]),
                          range_bytesize) != CUDA_SUCCESS)
            {
                arm.verified = false;
                arm.note = "read back failed on device " + std::to_string(d);
                break;
            }
            if (std::memcmp(landed.data(), data.data() + i * range_bytesize, range_bytesize) != 0)
            {
                arm.verified = false;
                arm.note = "device " + std::to_string(d) + " range " + std::to_string(i) + " differs";
                break;
            }
        }
    }

    for (auto & destination : destinations)
    {
        destination.free_all();
    }
    return arm;
}

// The same read to HOST memory. Without it a device rate cannot be read: it says nothing about the
// copy path if the file system was the limit all along.
// `aligned` decides whether this is a CONTROL or a curiosity. The worker reads directly only when
// the destination is congruent with the file offset, and a staging buffer is page aligned - so a
// pageable destination quietly reads BUFFERED, out of page cache, and its rate says nothing about
// what the device path is up against.
Arm run_host_arm(const std::string & path, const std::vector<uint8_t> & data, size_t range_bytesize,
                 bool aligned, unsigned iterations)
{
    Arm arm;
    arm.layout = aligned ? "host aligned" : "host pageable";
    arm.devices = 0;

    const size_t total = data.size();
    const unsigned ranges = static_cast<unsigned>(total / range_bytesize);

    std::vector<char> pageable;
    char * destination_base = nullptr;
    void * owned = nullptr;

    if (aligned)
    {
        owned = ::aligned_alloc(4096, total);
        if (owned == nullptr)
        {
            arm.note = "could not allocate an aligned destination";
            return arm;
        }
        destination_base = static_cast<char *>(owned);
    }
    else
    {
        pageable.resize(total);
        destination_base = pageable.data();
    }

    Streamer streamer;
    std::vector<FileRanges> request(1);
    request[0].path = path;
    for (unsigned i = 0; i < ranges; ++i)
    {
        request[0].ranges.push_back(
            ReadRange{ i * range_bytesize, range_bytesize, destination_base + i * range_bytesize });
    }

    unsigned failures = 0;
    double steady_seconds = 0;

    // The SAME rounds as a device arm. One round would report a destination being faulted in for the
    // first time - 16 GiB of fresh anonymous pages - as though it were a steady rate, and the device
    // arms fault their staging once and reuse it. That is not a difference in the readers.
    for (unsigned round = 0; round < iterations; ++round)
    {
        const auto start = std::chrono::steady_clock::now();

        SubmissionId id = 0;
        if (streamer.async_request(request, common::Device::host(), &id) != common::ResponseCode::Success)
        {
            arm.note = "submission refused";
            ::free(owned);
            return arm;
        }

        for (unsigned i = 0; i < ranges; ++i)
        {
            bool done = false;
            if (streamer.response(120000, done).ret != common::ResponseCode::Success)
            {
                ++failures;
            }
        }

        const auto elapsed = seconds_since(start);
        if (round == 0)
        {
            arm.first = static_cast<double>(total) / elapsed / 1e9;
        }
        else
        {
            steady_seconds += elapsed;
        }
    }

    arm.steady = iterations > 1
        ? (static_cast<double>(total) * (iterations - 1)) / steady_seconds / 1e9
        : arm.first;
    arm.verified = failures == 0 && std::memcmp(destination_base, data.data(), total) == 0;
    if (!arm.verified)
    {
        arm.note = failures != 0 ? "ranges failed" : "bytes differ";
    }
    ::free(owned);
    return arm;
}

} // namespace

} // namespace runai::llm::streamer::impl

int main(int argc, char ** argv)
{
    using namespace runai::llm::streamer;

    size_t total = 2ul << 30;          // 2 GiB
    size_t range_bytesize = 8ul << 20; // 8 MiB, the default chunk
    unsigned iterations = 5;
    std::vector<unsigned> device_counts = { 1, 2, 4 };

    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string name = argv[i];
        if (name == "--bytes")        { total = std::stoull(argv[i + 1]); }
        else if (name == "--range")   { range_bytesize = std::stoull(argv[i + 1]); }
        else if (name == "--iterations") { iterations = static_cast<unsigned>(std::stoul(argv[i + 1])); }
        else if (name == "--devices")
        {
            device_counts.clear();
            std::string list = argv[i + 1];
            size_t pos = 0;
            while (!list.empty())
            {
                pos = list.find(',');
                device_counts.push_back(static_cast<unsigned>(std::stoul(list.substr(0, pos))));
                if (pos == std::string::npos) { break; }
                list = list.substr(pos + 1);
            }
        }
    }

    const auto backend = device::cuda::backend();
    if (backend == nullptr)
    {
        std::cerr << "no CUDA driver on this host\n";
        return 77;
    }

    unsigned available = 0;
    backend->device_count(available);

    const auto read_back = impl::read_back_fn();
    if (read_back == nullptr)
    {
        std::cerr << "cuMemcpyDtoH_v2 could not be resolved\n";
        return 1;
    }

    std::cerr << "devices available: " << available
              << "  file " << (total >> 20) << " MiB"
              << "  range " << (range_bytesize >> 20) << " MiB"
              << "  " << (total / range_bytesize) << " ranges"
              << "  " << iterations << " rounds on one streamer\n";

    const auto data = utils::random::buffer(total);
    utils::temp::File file(data);

    // Warm the page cache once, so every arm sees the same storage and the comparison is about the
    // GPU path rather than about which arm happened to read cold.
    {
        std::vector<char> warm(1ul << 20);
        FILE * f = ::fopen(file.path.c_str(), "rb");
        while (f != nullptr && ::fread(warm.data(), 1, warm.size(), f) > 0) {}
        if (f != nullptr) { ::fclose(f); }
    }

    std::vector<impl::Arm> arms;

    // Two host arms. The ALIGNED one is the control - same strategy, same reads, only the copy to the
    // device missing. The pageable one is there to show what it costs to get that wrong.
    for (const bool aligned : { true, false })
    {
        arms.push_back(impl::run_host_arm(file.path, data, range_bytesize, aligned, iterations));
        std::cerr << "  " << arms.back().layout << "   "
                  << std::fixed << std::setprecision(2) << arms.back().first << " GB/s  "
                  << (arms.back().verified ? "verified" : ("NOT VERIFIED: " + arms.back().note)) << "\n";
    }

    for (const auto count : device_counts)
    {
        if (count > available)
        {
            std::cerr << "skipping " << count << " devices; only " << available << " here\n";
            continue;
        }
        for (const bool single : { true, false })
        {
            arms.push_back(impl::run_arm(backend, file.path, data, count, single, range_bytesize,
                                         read_back, iterations));
            const auto & arm = arms.back();
            std::cerr << "  " << arm.devices << " device(s), "
                      << (arm.single ? "one buffer " : "many buffers")
                      << "  first " << std::fixed << std::setprecision(2) << arm.first
                      << "  steady " << arm.steady << " GB/s"
                      << "  " << (arm.verified ? "verified" : ("NOT VERIFIED: " + arm.note)) << "\n";
        }
    }

    std::cout << "\ndevices  layout        first GB/s  steady GB/s  steady GiB/s  per-dev GiB/s  verified\n";
    bool all_verified = true;
    for (const auto & arm : arms)
    {
        const double gib = arm.steady * 1e9 / (1024.0 * 1024.0 * 1024.0);
        const double per_device = arm.devices == 0 ? gib : gib / arm.devices;

        std::cout << std::setw(7) << (arm.devices == 0 ? std::string("-") : std::to_string(arm.devices)) << "  "
                  << std::setw(13) << std::left << arm.layout << std::right
                  << std::setw(10) << std::fixed << std::setprecision(2) << arm.first
                  << std::setw(13) << arm.steady
                  << std::setw(14) << gib
                  << std::setw(15) << per_device
                  << "   " << (arm.verified ? "yes" : "NO") << "\n";
        all_verified = all_verified && arm.verified;
    }

    return all_verified ? 0 : 1;
}
