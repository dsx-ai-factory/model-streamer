# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import struct
import tempfile
import unittest

import torch

from runai_model_streamer.safetensors_streamer.safetensors_streamer import SafetensorsStreamer


def write_safetensors(path, header_entries, data):
    """Write a safetensors file with the header keys in EXACTLY the given order.

    Hand-built rather than produced by the reference library, which assigns data_offsets in key order and
    so can never emit the ordering this test is about. The header is padded to an 8 byte boundary, as the
    reference writer does.
    """
    header = json.dumps(dict(header_entries)).encode("utf-8")
    header += b" " * (-len(header) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(data)


class TestMetadataOrdering(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zero_element_tensor_sharing_a_start_offset_is_not_rejected(self):
        """A zero element tensor shares its start offset with whatever follows it.

        data_offsets [16, 16] and [16, 48] both start at 16, so sorting by start alone leaves the order to
        the header's key order via Python's stable sort. With the real tensor first, the gap check computes
        16 + 32 = 48 against a next start of 16 and rejects the file as overlapping - a valid file refused
        because of key order. Sorting by (start, size) puts the empty tensor first, where it belongs.

        The keys below are deliberately in the order the reference writer would never produce.
        """
        values = torch.arange(12, dtype=torch.float32)
        data = values[:4].numpy().tobytes() + values[4:].numpy().tobytes()   # 16 + 32 = 48 bytes
        path = os.path.join(self.temp_dir, "model.safetensors")

        write_safetensors(
            path,
            [
                ("first", {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}),
                ("second", {"dtype": "F32", "shape": [8], "data_offsets": [16, 48]}),
                ("empty", {"dtype": "F32", "shape": [0, 3], "data_offsets": [16, 16]}),
            ],
            data,
        )

        with SafetensorsStreamer() as streamer:
            streamer.stream_file(path)
            # clone: the yielded tensor is a VIEW into a ring buffer that is recycled once the
            # generator advances, so anything compared after the loop must own its data
            tensors = {name: tensor.clone() for name, tensor in streamer.get_tensors()}

        self.assertEqual(sorted(tensors), ["empty", "first", "second"])
        self.assertEqual(tuple(tensors["empty"].shape), (0, 3))
        self.assertEqual(tensors["empty"].numel(), 0)
        self.assertTrue(torch.equal(tensors["first"], values[:4]))
        self.assertTrue(torch.equal(tensors["second"], values[4:]))


if __name__ == "__main__":
    unittest.main()
