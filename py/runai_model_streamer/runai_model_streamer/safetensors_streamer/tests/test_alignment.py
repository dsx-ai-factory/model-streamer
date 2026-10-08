import unittest
from unittest.mock import patch

import numpy as np
import torch

from runai_model_streamer.safetensors_streamer import safetensors_pytorch
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import (
    SafetensorMetadata,
    create_torch_tensor,
    requires_alignment_copy,
)


def uint8_at(pool, offset, nbytes):
    """A uint8 tensor over pool[offset:offset+nbytes], built the way FileStreamer builds one.

    from_numpy, not a torch slice: that is what gives storage_offset 0 and lets a misaligned address
    through the dtype view unnoticed. A torch slice raises instead, so a fixture built that way would
    exercise a path the product does not take.
    """
    return torch.from_numpy(pool[offset:offset + nbytes]).view(1, -1)


def metadata(name, shape, dtype, nbytes):
    # data_offsets must agree with shape x element size - SafetensorMetadata validates it, which is
    # what would catch a fixture describing a tensor the bytes cannot hold.
    return SafetensorMetadata(name, {
        "shape": shape,
        "dtype": dtype,
        "data_offsets": [0, nbytes],
    })


class TestElementSize(unittest.TestCase):
    """Both ways of asking torch for bytes-per-element must agree.

    The fallback runs on no torch we test against - 2.0 is the only one without `dtype.itemsize` - so
    without this it would be dead code that nobody notices breaking.
    """

    # EVERY dtype this torch supports, not a sample. The map grows with the torch it runs on - F4 is
    # float4_e2m1fn_x2, a sub-byte packed type, and a sub-byte type is exactly where "bytes per
    # element" could mean two different things. A hardcoded list would never reach it.
    DTYPES = tuple(safetensors_pytorch.safetensors_to_torch_dtype.values())

    def test_the_fallback_agrees_with_the_attribute(self):
        with patch.object(safetensors_pytorch, "_DTYPE_HAS_ITEMSIZE", False):
            safetensors_pytorch._element_sizes.clear()
            fallback = {d: safetensors_pytorch.element_size_of(d) for d in self.DTYPES}

        expected = {d: torch.tensor([], dtype=d).element_size() for d in self.DTYPES}
        self.assertEqual(fallback, expected)
        self.assertGreater(len(expected), 10, "the dtype map looks empty - is the fixture right?")

    def test_the_attribute_path_does_not_build_tensors(self):
        if not safetensors_pytorch._DTYPE_HAS_ITEMSIZE:
            self.skipTest("this torch has no dtype.itemsize, so there is no fast path to check")

        safetensors_pytorch._element_sizes.clear()
        for dtype in self.DTYPES:
            safetensors_pytorch.element_size_of(dtype)

        self.assertEqual(safetensors_pytorch._element_sizes, {})

    def test_the_fallback_memoises_by_dtype(self):
        with patch.object(safetensors_pytorch, "_DTYPE_HAS_ITEMSIZE", False):
            safetensors_pytorch._element_sizes.clear()
            for _ in range(5):
                safetensors_pytorch.element_size_of(torch.bfloat16)

            self.assertEqual(safetensors_pytorch._element_sizes, {torch.bfloat16: 2})


class TestRequiresAlignmentCopy(unittest.TestCase):
    """A tensor's address comes from where the ring put it, and under O_DIRECT the ring cannot always
    choose: a range's address must be congruent to its file offset modulo the block, so a tensor at an
    odd file offset gets an odd address."""

    def setUp(self):
        # 2-byte aligned base, so the offset alone decides alignment.
        self.pool = np.zeros(4096, dtype=np.uint8)
        self.assertEqual(self.pool.ctypes.data % 2, 0)

    def test_an_aligned_tensor_needs_no_copy(self):
        # The case that must stay free: handing back the buffer is the whole point of owning it.
        self.assertFalse(
            requires_alignment_copy(uint8_at(self.pool, 0, 8), metadata("w", [4], "BF16", 8))
        )

    def test_a_misaligned_tensor_needs_a_copy(self):
        buffer = uint8_at(self.pool, 1, 8)
        self.assertEqual(buffer.data_ptr() % 2, 1, "the fixture must actually be misaligned")

        self.assertTrue(requires_alignment_copy(buffer, metadata("w", [4], "BF16", 8)))

    def test_a_single_byte_dtype_never_needs_a_copy(self):
        # Any address suits fp8 - and its odd SIZES are what push later tensors onto odd addresses in
        # the first place.
        self.assertFalse(
            requires_alignment_copy(uint8_at(self.pool, 1, 4), metadata("w", [4], "F8_E4M3", 4))
        )

    def test_a_zero_element_tensor_needs_no_copy(self):
        # create_torch_tensor builds it from torch.empty and never looks at the buffer, so asking
        # about its address would be answering a question nobody asked.
        self.assertFalse(
            requires_alignment_copy(uint8_at(self.pool, 1, 0), metadata("w", [0], "BF16", 0))
        )

    def test_the_dtype_view_hides_the_problem(self):
        # Why the predicate exists at all: torch's rule is storage_offset, which from_numpy makes 0,
        # so the view SUCCEEDS on a misaligned address and the tensor looks correct. If this ever
        # starts raising, the check can go.
        buffer = uint8_at(self.pool, 1, 8)
        tensor = create_torch_tensor(buffer, metadata("w", [4], "BF16", 8))

        self.assertEqual(tensor.data_ptr() % 2, 1)

    def test_a_copy_carries_the_same_bytes(self):
        # A copy that moved the window would satisfy every address check and still hand back the
        # wrong weights.
        payload = np.arange(1, 9, dtype=np.uint8)
        self.pool[1:9] = payload

        copied = uint8_at(self.pool, 1, 8).clone()
        tensor = create_torch_tensor(copied, metadata("w", [4], "BF16", 8))

        self.assertEqual(tensor.data_ptr() % 2, 0)
        self.assertTrue(np.array_equal(tensor.view(torch.uint8).numpy().reshape(-1), payload))



class TestSessionCountsAlignmentCopies(unittest.TestCase):
    """The streamer owns the copy and the count, so create_torch_tensor stays pure.

    The predicate is tested above; these drive get_tensors() with it stubbed, so the wiring is
    exercised without having to force a real misalignment out of the ring's geometry.
    """

    def streamer_over(self, buffers_and_metadata):
        from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
            SafetensorsStreamer,
        )

        streamer = SafetensorsStreamer()
        streamer.files_to_tensors_metadata = {
            0: {index: meta for index, (_, meta) in enumerate(buffers_and_metadata)}
        }
        chunks = [(0, index, buffer) for index, (buffer, _) in enumerate(buffers_and_metadata)]
        streamer.file_streamer.get_chunks = lambda: iter(chunks)
        return streamer

    def fixture(self, count):
        pool = np.zeros(4096, dtype=np.uint8)
        return pool, [(uint8_at(pool, index * 8, 8), metadata(f"w{index}", [4], "BF16", 8))
                      for index in range(count)]

    def test_nothing_is_copied_when_every_address_is_usable(self):
        _, items = self.fixture(3)
        streamer = self.streamer_over(items)

        yielded = list(streamer.get_tensors())

        self.assertEqual(len(yielded), 3)
        self.assertEqual(streamer.unaligned_copies, 0)
        # The same memory, not a copy of it.
        for (buffer, _), (_, tensor) in zip(items, yielded):
            self.assertEqual(tensor.data_ptr(), buffer.data_ptr())

    @patch("runai_model_streamer.safetensors_streamer.safetensors_pytorch.requires_alignment_copy",
           return_value=True)
    def test_a_copy_is_made_and_counted(self, _predicate):
        _, items = self.fixture(3)
        streamer = self.streamer_over(items)

        yielded = list(streamer.get_tensors())

        self.assertEqual(streamer.unaligned_copies, 3)
        for (buffer, _), (_, tensor) in zip(items, yielded):
            self.assertNotEqual(tensor.data_ptr(), buffer.data_ptr())

    @patch("runai_model_streamer.safetensors_streamer.safetensors_pytorch.requires_alignment_copy",
           return_value=False)
    def test_a_misaligned_tensor_that_slips_through_is_refused(self, _predicate):
        """The post-check, with the pre-check stubbed to miss it.

        It exists because making a copy aligned relies on torch's allocator, which is an
        implementation detail. A model that is quietly wrong is worse than a load that fails."""
        pool = np.zeros(4096, dtype=np.uint8)
        items = [(uint8_at(pool, 1, 8), metadata("w", [4], "BF16", 8))]
        streamer = self.streamer_over(items)

        with self.assertRaises(ValueError) as raised:
            list(streamer.get_tensors())

        self.assertIn("cannot be read from", str(raised.exception))

    def test_a_zero_element_tensor_passes_the_post_check(self):
        # create_torch_tensor builds it from torch.empty, whose data_ptr is 0 - which the post-check
        # must read as aligned rather than as a tensor to refuse.
        pool = np.zeros(4096, dtype=np.uint8)
        items = [(uint8_at(pool, 1, 0), metadata("w", [0], "BF16", 0))]
        streamer = self.streamer_over(items)

        yielded = list(streamer.get_tensors())

        self.assertEqual(len(yielded), 1)
        self.assertEqual(yielded[0][1].numel(), 0)
        self.assertEqual(streamer.unaligned_copies, 0)

    @patch("runai_model_streamer.safetensors_streamer.safetensors_pytorch.requires_alignment_copy",
           return_value=True)
    def test_the_count_belongs_to_the_session(self, _predicate):
        # A module-level counter would carry the first streamer's total into the second's log, and
        # would lose increments under a threaded consumer.
        _, first_items = self.fixture(2)
        first = self.streamer_over(first_items)
        list(first.get_tensors())

        _, second_items = self.fixture(1)
        second = self.streamer_over(second_items)
        list(second.get_tensors())

        self.assertEqual(first.unaligned_copies, 2)
        self.assertEqual(second.unaligned_copies, 1)


if __name__ == "__main__":
    unittest.main()
