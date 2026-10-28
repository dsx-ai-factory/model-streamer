#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A small real model for the smoke test of the real-GPU tests.

The model is downloaded from the Hugging Face hub at a fixed revision on every run, checked against a recorded size and
SHA-256, and described by a manifest in the format of gen_models.py: the name, dtype, shape, position and CRC32 of
every tensor. The CRC32 is computed here from the bytes of the file, not by the streamer, so the rows machinery of
test_synthetic_models.py can stream the model and compare it like any other. The download and the manifest are timed
(timing-real.jsonl), to see what the download adds to a run.

    real_model.py fetch --out DIR [--url URL]

writes DIR/real_smoke/model.safetensors, DIR/real_smoke.manifest.json and DIR/timing-real.jsonl.

Needs only the standard library. Runs under Python 3.8 or later.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_timer import PhaseTimer, dump_jsonl, stamp  # noqa: E402

MIB = 1 << 20
CHUNK = 8 * MIB

# Apache-2.0, not gated. Pinned by revision, size and SHA-256, so a change upstream fails the run instead of changing
# what is expected.
MODEL = {
    "repo": "Qwen/Qwen2.5-0.5B",
    "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "filename": "model.safetensors",
    "size": 988097824,
    "sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
MODEL_NAME = "real_smoke"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 60


class RealModelError(Exception):
    pass


def default_url(model=MODEL):
    return "https://huggingface.co/%s/resolve/%s/%s" % (model["repo"], model["revision"], model["filename"])


def download(url, destination, size, sha256, attempts=DOWNLOAD_ATTEMPTS):
    """Download url to destination and check its size and SHA-256. A failed attempt is repeated; the last error is
    raised. Returns the number of bytes."""
    last_error = None
    for attempt in range(1, attempts + 1):
        partial = destination + ".part"
        try:
            digest = hashlib.sha256()
            done = 0
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(partial, "wb") as out:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
            if done != size:
                raise RealModelError("downloaded %d bytes, expected %d" % (done, size))
            if digest.hexdigest() != sha256:
                raise RealModelError("sha256 %s, expected %s" % (digest.hexdigest(), sha256))
            os.replace(partial, destination)
            return done
        except (OSError, RealModelError) as error:
            last_error = error
            stamp("download attempt %d of %d failed: %s" % (attempt, attempts, error))
            if os.path.exists(partial):
                os.unlink(partial)
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise RealModelError("could not download %s: %s" % (url, last_error))


def read_header(path):
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return header_len, header


def build_manifest(directory, filename, name=MODEL_NAME):
    """The manifest of the safetensors file directory/filename, in the format of gen_models.py. Raises if the offsets
    leave a hole or the file does not end with the last tensor."""
    path = os.path.join(directory, filename)
    header_len, header = read_header(path)
    metadata = header.pop("__metadata__", None)
    data_start = 8 + header_len
    size = os.path.getsize(path)
    # By (start, end): an empty tensor starts where the next one does, and has to come first whatever the header order.
    ordered = sorted(header.items(), key=lambda item: tuple(item[1]["data_offsets"]))
    entries, position = [], 0
    with open(path, "rb") as handle:
        for key, (tensor_name, tensor) in enumerate(ordered):
            start, end = tensor["data_offsets"]
            if start != position:
                raise RealModelError("%s: tensor %s starts at %d, the previous one ended at %d" % (
                    filename, tensor_name, start, position))
            handle.seek(data_start + start)
            crc, left = 0, end - start
            while left:
                chunk = handle.read(min(left, CHUNK))
                if not chunk:
                    raise RealModelError("%s: file ends inside tensor %s" % (filename, tensor_name))
                crc = zlib.crc32(chunk, crc)
                left -= len(chunk)
            entries.append({"name": tensor_name, "dtype": tensor["dtype"], "shape": tensor["shape"], "start": start,
                            "end": end, "crc32": crc, "gen": "real", "key": key})
            position = end
    if data_start + position != size:
        raise RealModelError("%s: %d bytes after the last tensor" % (filename, size - data_start - position))
    sizes = [e["end"] - e["start"] for e in entries]
    return {
        "model": name, "source": MODEL, "dtypes": sorted({e["dtype"] for e in entries}),
        "num_files": 1, "num_tensors": len(entries), "total_data_bytes": sum(sizes),
        "largest_tensor_bytes": max(sizes or [0]),
        "files": [{"path": filename, "size": size, "header_len": header_len, "data_start": data_start,
                   "header_mode": "real", "metadata": metadata, "tensors": entries}],
    }


def fetch(out_dir, url=None, model=MODEL, name=MODEL_NAME):
    timer = PhaseTimer()
    directory = os.path.join(out_dir, name)
    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, model["filename"])
    with timer.phase("real_model.download", nbytes=model["size"]):
        download(url or default_url(model), destination, model["size"], model["sha256"])
    with timer.phase("real_model.manifest", nbytes=model["size"]):
        manifest = build_manifest(directory, model["filename"], name)
    with open(os.path.join(out_dir, name + ".manifest.json"), "w") as handle:
        json.dump(manifest, handle)
    dump_jsonl(timer.records, os.path.join(out_dir, "timing-real.jsonl"))
    seconds = next(r["seconds"] for r in timer.records if r["phase"] == "real_model.download")
    stamp("%s: %d tensors, %.2f GiB downloaded in %.1fs (%.1f MiB/s)" % (
        name, manifest["num_tensors"], model["size"] / 2**30, seconds, model["size"] / MIB / max(seconds, 1e-9)))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_parser = sub.add_parser("fetch")
    fetch_parser.add_argument("--out", required=True)
    fetch_parser.add_argument("--url", default=None, help="where to download from (default: the pinned revision)")
    args = parser.parse_args(argv)
    try:
        fetch(args.out, args.url)
    except RealModelError as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
