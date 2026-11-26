# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""For every dtype of the streamer's table, the streamer yields what `safe_open` reads from the same file.

A file of a scalar, a vector and a 3-D tensor is written per dtype from `gen_models.special_pattern`. The bytes are
compared, not the values: NaN != NaN and 0.0 == -0.0. `owned` is varied by the rows of test_synthetic_models.py.
"""

import os
import sys
import tempfile
import unittest

import torch
from safetensors import __version__ as SAFETENSORS_VERSION
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_models as gen  # noqa: E402
from runai_model_streamer import SafetensorsStreamer  # noqa: E402
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype  # noqa: E402

CUBE = (2, 3, 4)  # in values; the last dimension of a packed dtype is multiplied, so that it stays even


def element_bytes(st_name, count):
    """count elements cycling through the special patterns (bytes, for a packed dtype)."""
    size = gen.DTYPES[st_name][0]
    pattern = gen.special_pattern(st_name)
    elements = [pattern[i:i + size] for i in range(0, len(pattern), size)]
    return b"".join(elements[i % len(elements)] for i in range(count))


def shapes(st_name):
    """name -> shape in values, as a header records it. A packed dtype has no scalar."""
    packed = gen.PACKED_VALUES.get(st_name, 1)
    pattern_elements = len(gen.special_pattern(st_name)) // gen.DTYPES[st_name][0]
    result = {"vector": (pattern_elements * packed,), "cube": CUBE[:-1] + (CUBE[-1] * packed,)}
    if packed == 1:
        result["scalar"] = ()
    return result


def build_tensors(st_name):
    torch_dtype = safetensors_to_torch_dtype[st_name]
    packed = gen.PACKED_VALUES.get(st_name, 1)
    tensors = {}
    for name, shape in shapes(st_name).items():
        values = 1
        for dimension in shape:
            values *= dimension
        raw = torch.tensor(list(element_bytes(st_name, values // packed)), dtype=torch.uint8)
        tensors[name] = raw.view(torch_dtype).reshape(gen.packed_shape(st_name, shape))
    return tensors


def raw_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)  # reshape: a 0-d tensor cannot be viewed


class AgainstSafetensors(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="against-safetensors-")

    def tearDown(self):
        self.directory.cleanup()

    def test_the_table_was_found(self):
        self.assertGreater(len(safetensors_to_torch_dtype), 10, "the dtype table looks empty")

    def cannot_compare(self, st_name, action, error):
        self.fail("%s: safetensors %s cannot %s (%s: %s), so the streamer's dtype is not compared with it" % (
            st_name, SAFETENSORS_VERSION, action, type(error).__name__, error))

    def reference(self, st_name, path):
        try:
            with safe_open(path, framework="pt") as handle:
                return {name: handle.get_tensor(name) for name in handle.keys()}
        except Exception as error:
            self.cannot_compare(st_name, "read the file", error)

    def streamed(self, path):
        result = {}
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(path, None, "cpu", owned=True)
            for name, tensor in streamer.get_tensors():
                result[name] = tensor
        return result


def make_case(st_name):
    def case(self):
        self.assertIn(st_name, gen.DTYPES,
                      "%s is in the streamer's table and the generator does not know it: add it to DTYPES in "
                      "gen_models.py" % st_name)
        path = os.path.join(self.directory.name, "%s.safetensors" % st_name)
        try:
            save_file(build_tensors(st_name), path)
        except Exception as error:
            self.cannot_compare(st_name, "write it", error)
        for name, written in gen.header_dtypes(path).items():
            self.assertEqual(written, st_name, "%s: safetensors %s wrote %s for tensor %s, so no %s file was read" % (
                st_name, SAFETENSORS_VERSION, written, name, st_name))

        theirs = self.reference(st_name, path)
        ours = self.streamed(path)

        self.assertEqual(sorted(ours), sorted(theirs), "%s: the tensors differ" % st_name)
        for name, their in theirs.items():
            our = ours[name]
            where = "%s, tensor %s (safetensors %s, torch %s)" % (st_name, name, SAFETENSORS_VERSION, torch.__version__)
            self.assertEqual(our.dtype, their.dtype, "dtype of %s" % where)
            self.assertEqual(tuple(our.shape), tuple(their.shape), "shape of %s" % where)
            self.assertTrue(torch.equal(raw_bytes(our), raw_bytes(their)), "bytes of %s" % where)
        print("%s: %d tensors, the same as safe_open" % (st_name, len(theirs)))
    return case


for _name in sorted(safetensors_to_torch_dtype):
    setattr(AgainstSafetensors, "test_dtype_%s" % _name, make_case(_name))


if __name__ == "__main__":
    unittest.main()
