# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.file_streamer import file_streamer as file_streamer_module
from runai_model_streamer.file_streamer.device_support import DeviceSupport
from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.file_streamer.requests_iterator import FileChunks
from runai_model_streamer.file_streamer.tests.test_device_ring_path import FAKE_STREAM, FakeCudaEvent
from runai_model_streamer.libstreamer import (
    RUNAI_FILE_STREAMER_DEVICE_CPU,
    RUNAI_FILE_STREAMER_DEVICE_CUDA,
)

RANGE_SIZE = 8
RANGES = 12
WARNS_FROM = "runai_model_streamer.file_streamer.file_streamer"


class TestDeviceRouting(unittest.TestCase):
    """Which ring stream_files builds, and what it tells the library about the destination.

    Forced through the device_support seam, so these run on a host with no device. The slots are then
    CPU tensors while the submission declares CUDA, so what the library is told does not describe the
    memory - these tests read the DECISION, and the ones that consume are the host-ring fallbacks,
    where the two agree again. Data through a device ring is covered by test_device_ring_path, and
    end to end on real CUDA at the bottom of this file."""

    # CUDA because it is a type the C API serves: an invented value would be refused by
    # runai_request, which is a different failure from the one under test.
    SUPPORT = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeCudaEvent,
                            current_stream=FAKE_STREAM)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "w.bin")
        self.expected = {i: chr(97 + i % 26) * RANGE_SIZE for i in range(RANGES)}
        with open(self.path, "w") as handle:
            handle.write("".join(self.expected[i] for i in range(RANGES)))
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def chunks(self):
        return [FileChunks.contiguous(0, self.path, 0, [RANGE_SIZE] * RANGES)]

    def decide(self, free_bytes, supported=True, device="cpu", consume=False):
        """Submit one stream with the device seam forced, and report what stream_files decided.

        A ring whose slots are released with an event IS the device ring: a host slot is reusable the
        moment its copy returns, so that property is the one the two rings cannot share."""
        submitted = []
        real_request = file_streamer_module.runai_request

        def recording_request(streamer, *args, **kwargs):
            submitted.append(args[-2:])        # device_type, device_id - passed positionally
            return real_request(streamer, *args, **kwargs)

        got = None
        with patch.object(file_streamer_module, "device_support",
                          return_value=self.SUPPORT if supported else None), \
                patch.object(file_streamer_module, "device_free_memory", return_value=free_bytes), \
                patch.object(file_streamer_module, "runai_request", recording_request), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size",
                             return_value=4096) as probe:
            with FileStreamer() as fs:
                fs.stream_files(self.chunks(), None, device)
                device_ring = fs.requests_iterator.slots.reuse_needs_event
                if consume:
                    got = {index: bytes(buffer.numpy().tobytes()).decode()
                           for _file_id, index, buffer in fs.get_chunks()}
        return device_ring, fs._device, set(submitted), got, probe

    def ring_for(self, **kwargs):
        """The ring object itself, for the two rules stream_files' docstring states but that nothing
        else here would notice being broken."""
        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size", return_value=4096):
            with FileStreamer() as fs:
                fs.stream_files(self.chunks(), None, "cpu", **kwargs)
                ring = fs.requests_iterator
                shape = (ring.num_buffers, ring.buffer_size, ring.slots.reuse_needs_event,
                         ring._pool._owned)
                fs.drain_live_submissions()
        return shape

    def test_memory_limit_does_not_reach_the_device_ring(self):
        """memory_limit caps HOST memory. A device ring is bounded by the device budget instead, so
        passing one must not change its shape - the docstring says so and nothing else checks it."""
        without = self.ring_for()
        with_limit = self.ring_for(memory_limit=RANGE_SIZE)     # absurdly small, host-only knob
        self.assertEqual(with_limit, without,
                         "memory_limit changed the device ring; it caps host memory only")

    def test_a_device_ring_is_never_owned(self):
        """owned applies to a host destination only. A device slot is reused, and the pool refuses an
        owned slot released with an event - so owned=True here would fail mid-stream, not here."""
        for owned in (True, False):
            with self.subTest(owned=owned):
                *_, reuse_needs_event, pool_owned = self.ring_for(owned=owned)
                self.assertTrue(reuse_needs_event, "not a device ring")
                self.assertFalse(pool_owned, f"a device ring was built owned for owned={owned}")

    def test_a_device_with_room_gets_a_device_ring(self):
        device_ring, device, submitted, _got, _probe = self.decide(free_bytes=1 << 30)
        self.assertTrue(device_ring)
        self.assertEqual(device, (RUNAI_FILE_STREAMER_DEVICE_CUDA, 0))
        self.assertEqual(submitted, {(RUNAI_FILE_STREAMER_DEVICE_CUDA, 0)},
                         "the library has to be told the destinations are device memory")

    def test_a_device_without_room_falls_back_to_the_host_ring(self):
        """Free memory below two of the largest range. Not an error - the caller still gets its data,
        read into host memory and copied up."""
        with self.assertLogs(WARNS_FROM, "WARNING") as logs:
            device_ring, device, submitted, got, _ = self.decide(free_bytes=RANGE_SIZE, consume=True)
        self.assertFalse(device_ring)
        self.assertEqual(got, self.expected)
        self.assertEqual(device, (RUNAI_FILE_STREAMER_DEVICE_CPU, 0))
        self.assertEqual(submitted, {(RUNAI_FILE_STREAMER_DEVICE_CPU, 0)},
                         "host memory declared as a device has the library read our addresses as "
                         "device pointers")
        self.assertIn("no device ring", logs.output[0])

    def test_a_device_torch_cannot_measure_falls_back(self):
        """None is unknown, not empty. Nothing to size a ring from, so the host path - and the
        warning claims nothing about how much memory the device has."""
        with self.assertLogs(WARNS_FROM, "WARNING") as logs:
            device_ring, _device, submitted, got, _ = self.decide(free_bytes=None, consume=True)
        self.assertFalse(device_ring)
        self.assertEqual(got, self.expected)
        self.assertEqual(submitted, {(RUNAI_FILE_STREAMER_DEVICE_CPU, 0)})
        self.assertIn("cannot measure", logs.output[0])

    def test_a_destination_the_table_does_not_serve_is_the_host_path_with_no_warning(self):
        """An absent vendor is how it always worked, so there is nothing to tell the user about."""
        with patch.object(file_streamer_module.logger, "warning") as warning:
            device_ring, _device, submitted, got, _ = self.decide(
                free_bytes=1 << 30, supported=False, consume=True)
        self.assertFalse(device_ring)
        self.assertEqual(got, self.expected)
        self.assertEqual(submitted, {(RUNAI_FILE_STREAMER_DEVICE_CPU, 0)})
        warning.assert_not_called()

    def test_the_device_branch_never_probes_the_mounts(self):
        """pread cannot target device memory, so the library stages through its own pinned host
        buffers - nothing direct lands on these slots and the block is never read. The probe is a
        syscall per mount, so it is asked only where the answer is used."""
        *_, probe = self.decide(free_bytes=1 << 30)
        probe.assert_not_called()

    def test_the_host_branch_still_probes_the_mounts(self):
        """The other half of the pair: moving the probe must not have dropped it. A host ring places
        its ranges congruently with their file offsets and needs the measured block to do it."""
        with self.assertLogs(WARNS_FROM, "WARNING"):
            *_, probe = self.decide(free_bytes=RANGE_SIZE)
        probe.assert_called_once()

    def test_the_ring_is_released_when_the_context_exits(self):
        """A device ring holds up to its whole budget in device memory. Keeping it past the load
        would compete with the model it was built for - and runai_end has joined the workers by
        then, so nothing can still be writing into the slots."""
        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30):
            with FileStreamer() as fs:
                fs.stream_files(self.chunks(), None, "cpu")
                self.assertIsNotNone(fs.requests_iterator)
        self.assertIsNone(fs.requests_iterator, "the ring outlived the context")

    def test_free_memory_is_read_after_the_previous_ring_is_freed(self):
        """Sizing a ring against a device the last ring is still holding under-reports what is free,
        so the next ring is smaller than the device can afford - or is refused outright."""
        seen = []

        def recording_free(device):
            seen.append(fs.requests_iterator)
            return 1 << 30

        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", recording_free), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size", return_value=4096):
            with FileStreamer() as fs:
                fs.stream_files(self.chunks(), None, "cpu")
                self.assertIsNotNone(fs.requests_iterator, "no ring was built to free")
                # Drained rather than consumed: the forced row declares CUDA while these slots are
                # CPU tensors, so the library rightly refuses a transfer. The guard only needs the
                # responses retired, which is what the drain does.
                fs.drain_live_submissions()
                fs.stream_files(self.chunks(), None, "cpu")      # the second stream is the test
        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[1], "free memory was read while the previous ring still held its slots")

    def test_a_device_with_no_ordinal_sends_zero(self):
        """torch reports index None for a device type with no ordinals. 0 is then the only device
        there is - None would be refused by runai_request, inside the stream."""
        self.assertIsNone(torch.empty(0, device="cpu").device.index)
        _ring, device, _submitted, _got, _ = self.decide(free_bytes=1 << 30)
        self.assertEqual(device[1], 0)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestDeviceRoutingOnCuda(unittest.TestCase):
    """The same routing with nothing faked: a real device, a real ring in its memory, real bytes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "w.bin")
        size = 4096
        self.payload = b"".join(bytes([i]) * size for i in range(RANGES))
        with open(self.path, "wb") as handle:
            handle.write(self.payload)
        self.size = size
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def test_a_cuda_destination_is_read_into_cuda_memory(self):
        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(0, self.path, 0, [self.size] * RANGES)],
                            None, "cuda:0")
            self.assertTrue(fs.requests_iterator.slots.reuse_needs_event, "not a device ring")
            self.assertEqual(fs._device, (RUNAI_FILE_STREAMER_DEVICE_CUDA, 0))
            for _file_id, index, buffer in fs.get_chunks():
                self.assertEqual(buffer.device.type, "cuda", f"range {index} came back on the host")
                self.assertEqual(bytes(buffer.to("cpu").numpy().tobytes()),
                                 self.payload[index * self.size:(index + 1) * self.size],
                                 f"range {index} has the wrong bytes")

    def stream_once(self, streamer):
        streamer.stream_files([FileChunks.contiguous(0, self.path, 0, [self.size] * RANGES)],
                              None, "cuda:0")
        ring = streamer.requests_iterator.num_buffers * streamer.requests_iterator.buffer_size
        for _file_id, _index, _buffer in streamer.get_chunks():
            pass                               # consume without retaining: clones are not the ring
        return ring

    def test_the_device_memory_comes_back(self):
        """Measured, not asserted structurally: the ring is torch memory, so freeing it returns the
        bytes to torch's allocator, which is where the engine loading the model takes them from."""
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        with FileStreamer() as fs:
            ring = self.stream_once(fs)
            self.assertGreaterEqual(torch.cuda.memory_allocated() - before, ring,
                                    "the ring was not in device memory at all")
        self.assertEqual(torch.cuda.memory_allocated(), before,
                         "device memory was still held after the context exited")

    def test_a_second_stream_does_not_hold_two_rings(self):
        """Rebinding evaluates the new ring before dropping the old, so the previous one is freed
        first. Without that both exist at the peak, which on a device is twice the budget."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with FileStreamer() as fs:
            ring = self.stream_once(fs)
            self.stream_once(fs)
            peak = torch.cuda.max_memory_allocated()
        self.assertLess(peak, 2 * ring, f"peak {peak} held two rings of {ring}")


if __name__ == "__main__":
    unittest.main()
