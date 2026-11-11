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
from runai_model_streamer.file_streamer.requests_iterator import (
    RANGE_OFFSET_ALIGNMENT,
    FileChunks,
    FilesRequestsIteratorWithBuffer,
    RunaiStreamerMemoryLimitException,
)
from runai_model_streamer.file_streamer.tests.test_device_ring_path import (
    FAKE_STREAM,
    FakeCudaEvent,
    FakeStream,
)
from runai_model_streamer.libstreamer import (
    RUNAI_FILE_STREAMER_DEVICE_CPU,
    RUNAI_FILE_STREAMER_DEVICE_CUDA,
)

RANGE_SIZE = 8
RANGES = 12
WARNS_FROM = "runai_model_streamer.file_streamer.file_streamer"


class TestDictatedSlotShape(unittest.TestCase):
    """stream_files(slot_shape=...) on BOTH branches it reaches.

    Distributed streaming dictates a shape because the size each peer derives is floored by the
    largest range in its OWN share, so peers derive different sizes and would lay the same ranges
    out differently. These drive the real stream_files, not the iterator underneath it: the
    constructor already took an explicit shape before this existed, so a test that called it
    directly would pass whether or not stream_files honours the argument at all."""

    SUPPORT = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeCudaEvent,
                            current_stream=FAKE_STREAM)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "w.bin")
        with open(self.path, "w") as handle:
            handle.write("x" * (RANGE_SIZE * RANGES))
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def chunks(self):
        return [FileChunks.contiguous(0, self.path, 0, [RANGE_SIZE] * RANGES)]

    def ring_shape(self, supported, slot_shape):
        """(buffer_size, num_buffers, is_device_ring) of the ring stream_files actually built.

        The third element matters: both branches honour slot_shape, so a device branch that ignored
        it would fall through to the host one and produce the SAME shape. Only the kind of ring says
        which branch ran."""
        # Sizing is made to RAISE: if either branch consults it, the dictated shape is being
        # ignored and the test fails loudly rather than by comparing numbers that happen to match.
        def must_not_be_called(*args, **kwargs):
            raise AssertionError("sizing was consulted although a shape was dictated")

        with patch.object(file_streamer_module, "device_support",
                          return_value=self.SUPPORT if supported else None), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "device_ring_sizing", must_not_be_called), \
                patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", must_not_be_called), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size", return_value=4096):
            with FileStreamer() as fs:
                fs.stream_files(self.chunks(), None, "cpu", slot_shape=slot_shape)
                ring = fs.requests_iterator
                shape = (ring.buffer_size, ring.num_buffers, ring.slots.reuse_needs_event)
                fs.drain_live_submissions()
        return shape

    def test_the_device_branch_uses_the_dictated_shape(self):
        # Deliberately not what sizing would choose for these ranges, and a DEVICE ring: a device
        # branch that ignored the shape would fall through to the host one and look identical.
        # The pair is also pinned the right way round here: swap the unpacking and buffer_size
        # becomes 5, below the largest range, which stream_files refuses before building a ring.
        self.assertEqual(self.ring_shape(supported=True, slot_shape=(RANGE_SIZE * 3, 5)),
                         (RANGE_SIZE * 3, 5, True))

    def test_the_host_fallback_uses_the_dictated_shape(self):
        # A device the C++ layer does not serve still takes the shape, on the host ring.
        self.assertEqual(self.ring_shape(supported=False, slot_shape=(RANGE_SIZE * 2, 7)),
                         (RANGE_SIZE * 2, 7, False))


class TestADictatedShapeIsValidated(unittest.TestCase):
    """stream_files refuses a shape that cannot work, at the boundary where it arrives.

    Every other way to a ring goes through a sizing function that floors the buffer at the largest
    range. A dictated shape has no floor, so this is the only path that can carry an impossible one -
    and TestAnImpossibleBufferTruncatesSilently shows what it would do."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "w.bin")
        with open(self.path, "w") as handle:
            handle.write("x" * (RANGE_SIZE * RANGES))
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def stream_with(self, slot_shape):
        chunks = [FileChunks.contiguous(0, self.path, 0, [RANGE_SIZE] * RANGES)]
        with patch.object(file_streamer_module, "runai_probe_direct_block_size", return_value=4096):
            with FileStreamer() as fs:
                fs.stream_files(chunks, None, "cpu", slot_shape=slot_shape)
                fs.drain_live_submissions()

    def test_a_buffer_too_small_for_the_largest_range_is_refused(self):
        with self.assertRaises(RunaiStreamerMemoryLimitException) as caught:
            self.stream_with((RANGE_SIZE - 1, 4))
        self.assertIn(str(RANGE_SIZE), str(caught.exception),
                      "the refusal must name the range that does not fit")

    def test_the_boundary_is_exact(self):
        # Exactly the largest range is enough; one byte less is not.
        self.stream_with((RANGE_SIZE, 4))
        with self.assertRaises(RunaiStreamerMemoryLimitException):
            self.stream_with((RANGE_SIZE - 1, 4))

    def test_a_ring_needs_a_buffer(self):
        with self.assertRaises(RunaiStreamerMemoryLimitException):
            self.stream_with((RANGE_SIZE * 2, 0))


class TestARingThatCannotBeAllocated(unittest.TestCase):
    """What a failed device-ring allocation looks like, which depends on WHO chose the shape.

    A dictated shape is an agreement between peers, so a shape that will not fit is REFUSED and the
    refusal says what was asked for - a rank that quietly took the host ring instead would lay its
    ranges out differently from the peers it broadcasts to.

    A shape this layer DERIVED has no agreement to explain, so the allocator's own error travels out
    untouched."""

    SUPPORT = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeCudaEvent,
                            current_stream=FAKE_STREAM)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "w.bin")
        with open(self.path, "w") as handle:
            handle.write("x" * (RANGE_SIZE * RANGES))
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def test_the_refusal_names_the_shape_the_device_and_what_was_free(self):
        chunks = [FileChunks.contiguous(0, self.path, 0, [RANGE_SIZE] * RANGES)]

        def out_of_memory(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=7 << 20), \
                patch.object(file_streamer_module, "device_slots", out_of_memory), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size", return_value=4096):
            with FileStreamer() as fs:
                with self.assertRaises(RunaiStreamerMemoryLimitException) as caught:
                    fs.stream_files(chunks, None, "cpu", slot_shape=(RANGE_SIZE * 2, 4))

        message = str(caught.exception)
        # The three things the allocator's own message does not say. The device is pinned BY
        # POSITION - a bare "cpu" would pass on a message that named no device at all, since the
        # word is this test's own argument. (A distinctive device string would be better still, but
        # the device is resolved for real through torch, which has no CUDA in this environment.)
        self.assertIn("4 x", message, "the refusal must name the shape asked for")
        self.assertIn("7.0 MiB", message, "and what was free")
        self.assertIn("on cpu:", message, "and which device it tried to build on, in its own slot")
        self.assertIsInstance(caught.exception.__cause__, torch.cuda.OutOfMemoryError)

    def test_a_derived_shape_keeps_the_allocator_s_own_error(self):
        """No slot_shape, so nothing was agreed and nothing is explained.

        An engine that catches OutOfMemoryError to retry at a lower memory fraction has to keep
        catching it - converting this would break a caller that never asked for a dictated shape."""
        chunks = [FileChunks.contiguous(0, self.path, 0, [RANGE_SIZE] * RANGES)]

        def out_of_memory(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=7 << 20), \
                patch.object(file_streamer_module, "device_slots", out_of_memory), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size", return_value=4096):
            with FileStreamer() as fs:
                with self.assertRaises(torch.cuda.OutOfMemoryError):
                    fs.stream_files(chunks, None, "cpu")


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

    def test_the_ring_waits_on_the_stream_before_it_submits(self):
        """A fresh slot can be a cached block whose previous owner still has work queued on this
        stream. Torch reuses it relying on stream order, and the library writes on its own stream,
        which that order does not cover - so only this wait keeps a late write off a filled slot.

        ONCE, and before the first address goes out: after that the blocks are the ring's own and
        nothing else can queue work against them."""
        order = []
        real_request = file_streamer_module.runai_request

        def recording_request(streamer, *args, **kwargs):
            order.append("submit")
            return real_request(streamer, *args, **kwargs)

        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "runai_request", recording_request), \
                patch.object(FakeStream, "synchronize", lambda _stream: order.append("sync")):
            with FileStreamer() as fs:
                fs.stream_files(self.chunks(), None, "cpu")
                fs.drain_live_submissions()

        self.assertIn("submit", order, "nothing was submitted, so the order proves nothing")
        self.assertEqual(order.count("sync"), 1,
                         f"the stream was waited on {order.count('sync')} times, not once")
        self.assertEqual(order[0], "sync", "an address was submitted before the stream was waited on")

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


class TestASentFromRing(unittest.TestCase):
    """stream_files(sent_from=True): a ring whose slots this caller will broadcast.

    Forced through the device_support seam, so it runs on a host with no device."""

    SUPPORT = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeCudaEvent,
                            current_stream=FAKE_STREAM)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "w.bin")
        with open(self.path, "w") as handle:
            handle.write("".join(chr(97 + i % 26) * RANGE_SIZE for i in range(RANGES)))
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def chunks(self):
        return [FileChunks.contiguous(0, self.path, 0, [RANGE_SIZE] * RANGES)]

    # What the group agreed: two slots, each holding two of these ranges once they are spaced.
    SHAPE = (RANGE_OFFSET_ALIGNMENT * 2, 2)

    def stream(self, sent_from, supported=True, slot_shape=SHAPE):
        """The ring stream_files built: its alignment, and whether it is the device one.

        Free memory is not a parameter: a dictated shape replaces the sizing, so it decides
        nothing here."""
        with patch.object(file_streamer_module, "device_support",
                          return_value=self.SUPPORT if supported else None), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size",
                             return_value=4096):
            with FileStreamer() as fs:
                # "cpu" with the seam forced: the slots are CPU tensors while the submission
                # declares CUDA, as in TestDeviceRouting above. What is read here is the DECISION.
                fs.stream_files(self.chunks(), device="cpu", sent_from=sent_from,
                                slot_shape=slot_shape)
                ring = fs.requests_iterator
                fs.drain_live_submissions()
                return ring.range_alignment, ring.slots.reuse_needs_event

    def test_a_sent_from_ring_is_placed_by_the_rule(self):
        self.assertEqual(self.stream(sent_from=True), (RANGE_OFFSET_ALIGNMENT, True))

    def test_an_ordinary_ring_packs_one_after_another(self):
        self.assertEqual(self.stream(sent_from=False), (1, True))

    def test_the_ranges_are_spaced_by_the_rule_in_the_ring_it_built(self):
        """Not only the setting: the shape holds two ranges a slot once they are spaced, where a
        tight layout would have fitted far more."""
        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size",
                             return_value=4096):
            with FileStreamer() as fs:
                fs.stream_files(self.chunks(), device="cpu", sent_from=True,
                                slot_shape=self.SHAPE)
                # The requests stream_files already submitted: they hold the ring's slots, so
                # asking for another would find none free.
                request = next(iter(fs.live_requests.values()))

                self.assertEqual(fs.requests_iterator.range_offsets_of(request),
                                 [0, RANGE_OFFSET_ALIGNMENT])
                fs.drain_live_submissions()

    def test_a_device_the_library_cannot_write_into_is_refused(self):
        """Never a silent host ring: its ranges are placed congruent with their file offsets, which
        depends on the mount and on this rank's address, so a peer cannot reproduce it."""
        with self.assertRaisesRegex(ValueError, "cannot read into that device"):
            self.stream(sent_from=True, supported=False)

    def test_a_sender_without_an_agreed_shape_is_refused(self):
        """A derived size is floored by the largest range in THIS share, so peers would split the
        ranges across slots differently - and the alignment says nothing about which ranges are in
        a slot, only where they sit inside one."""
        with self.assertRaisesRegex(ValueError, "slot shape its peers agreed"):
            self.stream(sent_from=True, slot_shape=None)

    def test_an_ordinary_stream_still_falls_back(self):
        """Neither refusal applies to a caller that did not ask to send from the ring.

        The HOST ring, said by what its slots are: a host slot is reusable the moment its copy
        returns, so it needs no event, where a device slot does."""
        alignment, needs_event = self.stream(sent_from=False, supported=False, slot_shape=None)

        self.assertEqual(alignment, 1)
        self.assertFalse(needs_event, "this is the device ring, so nothing fell back")

    def test_a_sender_is_refused_when_the_shape_cannot_be_allocated(self):
        """The other way out of the device branch. A sender must not reach the host ring by it
        either - and that protection comes from the dictated shape, so it is pinned here."""
        def out_of_memory(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("out of memory")

        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size",
                             return_value=4096), \
                patch.object(file_streamer_module, "FilesRequestsIteratorWithBuffer",
                             side_effect=out_of_memory):
            with FileStreamer() as fs:
                with self.assertRaises(RunaiStreamerMemoryLimitException):
                    fs.stream_files(self.chunks(), device="cpu", sent_from=True,
                                    slot_shape=self.SHAPE)
