import json
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save

from runai_model_streamer.safetensors_streamer.safetensors_pytorch import (
    SafetensorMetadata,
    create_torch_tensor,
)


@unittest.skipUnless(hasattr(torch, "float4_e2m1fn_x2"), "PyTorch has no packed F4 dtype")
class TestPackedFloat4(unittest.TestCase):
    def test_reference_writer_round_trip(self):
        # Compare with the reference reader, including empty last/non-last axes.
        for shape in ((2, 3), (4,), (0, 3), (2, 0)):
            with self.subTest(shape=shape):
                count = 1
                for dim in shape:
                    count *= dim
                raw = torch.arange(count, dtype=torch.uint8).reshape(shape)
                try:
                    encoded = save({"packed": raw.view(torch.float4_e2m1fn_x2)})
                except KeyError:
                    self.skipTest("Installed Safetensors does not support writing F4")
                header_size = int.from_bytes(encoded[:8], "little")
                header = json.loads(encoded[8:8 + header_size])
                metadata = SafetensorMetadata("packed", header["packed"])
                buffer = torch.tensor(list(encoded[8 + header_size:]), dtype=torch.uint8)

                actual = create_torch_tensor(buffer, metadata)
                with tempfile.NamedTemporaryFile(suffix=".safetensors") as fixture:
                    fixture.write(encoded)
                    fixture.flush()
                    with safe_open(fixture.name, framework="pt") as reference:
                        expected = reference.get_tensor("packed")
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_rejects_unrepresentable_shape(self):
        for shape in ([], [3], [2, 3]):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "even last dimension"):
                SafetensorMetadata("packed", {"dtype": "F4", "shape": shape, "data_offsets": [0, 3]})

    def test_rejects_incorrect_byte_count(self):
        with self.assertRaisesRegex(ValueError, "Corrupted Tensor"):
            SafetensorMetadata("packed", {"dtype": "F4", "shape": [2, 6], "data_offsets": [0, 12]})
