#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dtypes that the generator writes and the installed streamer accepts.

    dtype_check.py

Prints them, comma separated, on stdout (what gen_models.py takes as --dtypes). Fails when the streamer accepts a dtype
that the generator does not know and has not listed as a gap, so that a new dtype is not left untested without notice.
The report goes to stderr. It needs the installed streamer, for its dtype table.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_models  # noqa: E402
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype  # noqa: E402


def main():
    streamer = set(safetensors_to_torch_dtype)
    covered = sorted(streamer & set(gen_models.DTYPES))
    print("streamer dtypes: %s" % " ".join(sorted(streamer)), file=sys.stderr)
    print("not generated:   %s" % (" ".join(sorted(set(gen_models.DTYPES) - streamer)) or "-"), file=sys.stderr)
    unknown = sorted(streamer - set(gen_models.DTYPES) - gen_models.KNOWN_GAPS)
    print("not known to the generator: %s" % (" ".join(unknown) or "-"), file=sys.stderr)
    if unknown:
        print("ERROR: the streamer accepts %s, which the generator cannot write: add it to DTYPES in "
              "tests/gpu/suites/streaming/gen_models.py" % " ".join(unknown), file=sys.stderr)
        return 1
    print(",".join(covered))
    return 0


if __name__ == "__main__":
    sys.exit(main())
