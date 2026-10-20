#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic safetensors models for the real-GPU tests.

Four models, each isolating one kind of stress:

    mixed          every dtype, ranks 0-5, edge sizes, odd names, 4 uneven shards with different alignment
    many_tensors   about 100,000 tiny tensors and a multi-MB header in one file
    giant_tensor   a tensor of exactly 2 GiB and one of 2004 MiB (the largest tensor seen in a real model)
    many_files     5,000 small files, some header-only, some holding a single 1-byte tensor

The files are written by hand, not through safetensors.save_file, so the layout is under control: header
padding (and so the alignment of every tensor start), header-only files (and zero-sized dimensions, see EMPTY_TENSORS).

Every byte is a function of (seed, model, tensor key), drawn from numpy's PCG64 bit generator. numpy
guarantees that the raw stream of a bit generator never changes, unlike the Generator methods built on it,
so the bytes of a tensor depend only on its seed, model and key.

The layout (shapes, sizes, names, shard split) is drawn with Generator methods, which numpy may change between
versions. So a seed reproduces a model exactly only under the same numpy version. Each manifest records
numpy_version and generator_version; to rebuild an old model, install that numpy and run the generator with
the same seed. Each tensor has its own stream, so one tensor can be regenerated alone.

Alongside each model goes a manifest (<out>/<model>.manifest.json): for every tensor its name, dtype, shape,
position in the file and the CRC32 of its bytes, computed while the data is generated. The test compares what
reaches the GPU with it.

    gen_models.py generate --out DIR --seed N [--scale 1.0] [--files 5000] [--models NAME,NAME,...]
                           [--dtypes F32,U16,...] [--no-verify]
    gen_models.py verify   --out DIR [--models NAME,NAME,...]

The model names are the four above; by default all of them. Each model is written to <out>/<name>/.

Runs under the devcontainer's Python 3.8. Needs only numpy.
"""

import argparse
import json
import math
import os
import struct
import sys
import zlib
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_timer import PhaseTimer, dump_jsonl, report  # noqa: E402

GENERATOR_VERSION = 2

# Tensors with zero elements are left out for now: the streamer yields them on the CPU even when a CUDA device
# was requested (https://github.com/dsx-ai-factory/model-streamer/issues/260), so every row of the mixed model
# failed on them. None of 184 real checkpoints surveyed contains one. Once that issue is closed, set this to
# True: the generator then draws zero dimensions again, the mixed model gets its two mandatory empty tensors,
# and check_coverage requires one.
EMPTY_TENSORS = False

MIB = 1 << 20
GIB = 1 << 30
CHUNK = 64 * MIB  # data is produced and written in chunks of this size; a multiple of 8

# safetensors dtype name -> (element size in bytes, how the bytes are drawn)
DTYPES = {
    "F64": (8, "random"),
    "F32": (4, "random"),
    "F16": (2, "random"),
    "BF16": (2, "random"),
    "F8_E4M3": (1, "random"),
    "F8_E5M2": (1, "random"),
    "F8_E8M0": (1, "random"),
    "C64": (8, "random"),
    "I64": (8, "random"),
    "I32": (4, "random"),
    "I16": (2, "random"),
    "I8": (1, "random"),
    "U64": (8, "random"),
    "U32": (4, "random"),
    "U16": (2, "random"),
    "U8": (1, "random"),
    "BOOL": (1, "bool"),
}
DTYPE_NAMES = sorted(DTYPES)

# F4 (two 4-bit values per byte, so the shape in the header is not the shape of the torch tensor) is not here yet:
# the streamer rejects such a file (https://github.com/dsx-ai-factory/model-streamer/issues/234). The test run fails
# when the streamer accepts a dtype that is neither in DTYPES nor here, so that a new dtype is never left untested
# without someone deciding so.
KNOWN_GAPS = frozenset({"F4"})

# The dtypes the models are drawn from: all of DTYPES, or the ones given with --dtypes. The test run passes the
# ones the streamer accepts with the installed torch, which has no unsigned 16/32/64 bit or float8_e8m0 before
# 2.3 and 2.7 or so. Every one of them is forced into the mixed model, and check_coverage requires it.
active_dtypes = list(DTYPE_NAMES)


def use_dtypes(names):
    unknown = sorted(set(names) - set(DTYPES))
    if unknown or not names:
        raise GenerationError("unknown dtype(s) %s; the dtypes are %s" % (unknown or [names], ", ".join(DTYPE_NAMES)))
    active_dtypes[:] = sorted(set(names))

# U8 tensor sizes that sit on or next to the boundaries the read path cares about: 4 KiB pages, the 64 KiB
# direct-I/O block, and the 8 MiB read chunk.
EDGE_SIZES = (1, 4095, 4096, 4097, 64 * 1024 - 1, 64 * 1024, 64 * 1024 + 1,
              8 * MIB - 1, 8 * MIB, 8 * MIB + 1)

# The mixed model keeps every tensor below this so the memory limit used by the tests can stay small: a tensor
# must fit in one host buffer, and a distributed rank only gets its share of the limit.
MIXED_MAX_TENSOR = 24 * MIB

# The number is part of every seed, so it stays with a model whatever the model is called.
MODEL_IDS = {"mixed": 1, "many_tensors": 2, "giant_tensor": 3, "many_files": 4}
MODEL_NAMES = tuple(MODEL_IDS)

TensorSpec = namedtuple("TensorSpec", "name dtype shape gen key")
ShardPlan = namedtuple("ShardPlan", "filename tensors header_mode metadata")


class GenerationError(Exception):
    pass


class VerifyError(Exception):
    pass


def parse_models(text):
    """The models named in a comma separated list, in the order given."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODEL_IDS]
    if unknown or not names:
        raise GenerationError("unknown model(s) %s; the models are %s" % (unknown or [text], ", ".join(MODEL_NAMES)))
    return names


def elements(shape):
    return math.prod(shape) if shape else 1


def nbytes_of(spec):
    return elements(spec.shape) * DTYPES[spec.dtype][0]


# ----------------------------------------------------------------------------------------------------------
# bytes of a tensor
# ----------------------------------------------------------------------------------------------------------

def special_pattern(dtype):
    """Bit patterns a float or integer copy could mangle: NaNs with payloads, infinities, negative zero,
    denormals, the extremes. One element of each, back to back. For 1-byte types, every possible value."""
    size, gen = DTYPES[dtype]
    if gen == "bool":
        return bytes([0, 1])
    patterns = {
        1: list(range(256)),
        2: [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7FFF, 0xFFFF, 0x0001, 0x8001, 0x03FF,
            0x7F80, 0xFF80, 0x7FC0, 0x7F81],
        4: [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFFFFFFF,
            0x00000001, 0x80000001, 0x007FFFFF, 0x7FFFFFFF],
        8: [0, 1 << 63, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0x7FF0000000000001,
            0xFFFFFFFFFFFFFFFF, 1, 0x000FFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF],
    }[size]
    return b"".join(value.to_bytes(size, "little") for value in patterns)


def iter_tensor_bytes(seed, model, spec):
    """The bytes of a tensor as uint8 arrays of at most CHUNK bytes. The same arguments always give the
    same bytes."""
    total = nbytes_of(spec)
    if total == 0:
        return
    if spec.gen == "pattern":
        yield np.frombuffer(special_pattern(spec.dtype), dtype=np.uint8)
        return
    bit_generator = np.random.PCG64(np.random.SeedSequence([seed, MODEL_IDS[model], spec.key]))
    done = 0
    while done < total:
        size = min(CHUNK, total - done)
        raw = bit_generator.random_raw(-(-size // 8)).view(np.uint8)[:size]
        if spec.gen == "bool":
            raw = raw & 1
        yield raw
        done += size


# ----------------------------------------------------------------------------------------------------------
# plans: which tensors, in which files
# ----------------------------------------------------------------------------------------------------------

_PROJ = ("q", "k", "v", "o", "gate", "up", "down")
_NAME_TEMPLATES = (
    "model.layers.{l}.self_attn.{p}_proj.weight",
    "model.layers.{l}.mlp.{p}_proj.weight",
    "model.layers.{l}.input_layernorm.weight",
    "model.layers.{l}.mlp.experts.{e}.{p}_proj.weight",
    "transformer.h.{l}.attn.c_attn.bias",
    "encoder.block.{l}.layer.0.SelfAttention.{p}.weight",
    "model.layers.{l}.mlp.{p}_proj.qweight",
    "model.layers.{l}.mlp.{p}_proj.scales",
    "model.layers.{l}.mlp.{p}_proj.g_idx",
    "model.layers.{l}.mlp.{p}_proj.weight_scale",
)


def realistic_name(key, used):
    name = _NAME_TEMPLATES[key % len(_NAME_TEMPLATES)].format(l=key // 5, e=key % 64, p=_PROJ[key % 7])
    if name in used:
        name += ".dup%d" % key
    used.add(name)
    return name


def random_shape(rng, itemsize, max_bytes):
    """A shape of rank 0 to 5 whose size is log-uniform between one element and max_bytes. With EMPTY_TENSORS,
    about 2% of the non-scalar shapes get a zero dimension, which makes an empty tensor."""
    rank = int(rng.integers(0, 6))
    if rank == 0:
        return ()
    size = int(math.exp(rng.uniform(0.0, math.log(max(max_bytes, 2)))))
    remaining = max(1, size // itemsize)
    dims = []
    for _ in range(rank - 1):
        dim = int(rng.integers(1, min(8, remaining) + 1))
        dims.append(dim)
        remaining //= dim
    dims.append(max(1, remaining))
    if EMPTY_TENSORS and rng.random() < 0.02:
        dims[int(rng.integers(0, rank))] = 0
    return tuple(dims)


def random_tensor(rng, key, used, max_bytes, name=None):
    dtype = active_dtypes[int(rng.integers(0, len(active_dtypes)))]
    shape = random_shape(rng, DTYPES[dtype][0], max_bytes)
    gen = DTYPES[dtype][1]
    return TensorSpec(name or realistic_name(key, used), dtype, shape, gen, key)


def mandatory_tensors_mixed(first_key, used):
    """Everything the mixed model must contain whatever the random draw gives: every dtype as a scalar, as a tensor of
    rank 1-5 and as a pattern tensor; empty tensors (with EMPTY_TENSORS); the edge sizes; odd names."""
    specs = []
    key = first_key

    def add(name, dtype, shape, gen):
        nonlocal key
        assert name not in used, name
        used.add(name)
        specs.append(TensorSpec(name, dtype, tuple(shape), gen, key))
        key += 1

    for index, dtype in enumerate(active_dtypes):
        gen = DTYPES[dtype][1]
        rank = 1 + index % 5
        add("mandatory.%s.scalar" % dtype.lower(), dtype, (), gen)
        add("mandatory.%s.rank%d" % (dtype.lower(), rank), dtype, [2 + j % 3 for j in range(rank)], gen)
        add("mandatory.%s.special" % dtype.lower(), dtype,
            (len(special_pattern(dtype)) // DTYPES[dtype][0],), "pattern")
    if EMPTY_TENSORS:
        add("mandatory.empty.f32", "F32", (0,), "random")
        add("mandatory.empty.i8", "I8", (3, 0, 2), "random")
    for size in EDGE_SIZES:
        add("mandatory.edge.%d" % size, "U8", (size,), "random")
    add("mandatory.name.long." + "x" * 280 + ".weight", "F32", (3,), "random")
    add("mandatory.name.unicode.tënsor.日本語.\U0001F642", "F32", (3,), "random")
    add("mandatory.name/with slashes and spaces", "F32", (3,), "random")
    return specs, key


def plan_mixed(seed, scale):
    rng = np.random.default_rng([seed, MODEL_IDS["mixed"], 0xFFFF])
    used = set()
    budget = int(GIB * scale)
    tensors, key = mandatory_tensors_mixed(0, used)
    total = sum(nbytes_of(t) for t in tensors)
    if total > budget:
        raise GenerationError("model mixed: a budget of %d bytes cannot fit the mandatory tensors, which take %d "
                              "bytes; raise --scale" % (budget, total))
    while total < budget:
        spec = random_tensor(rng, key, used, min(MIXED_MAX_TENSOR, budget - total))
        tensors.append(spec)
        total += nbytes_of(spec)
        key += 1

    order = list(rng.permutation(len(tensors)))

    def size(index):
        return nbytes_of(tensors[index])

    first = next(i for i in order if 1024 <= size(i) <= 4 * MIB)
    small, small_bytes = [], 0
    for i in order:
        if i != first and 0 < size(i) <= 2048 and small_bytes + size(i) <= 512 * 1024:
            small.append(i)
            small_bytes += size(i)
    taken = set([first] + small)
    rest = [i for i in order if i not in taken]
    rest_bytes = sum(size(i) for i in rest)
    mid, mid_bytes = [], 0
    for i in rest:
        if mid_bytes + size(i) > rest_bytes // 5:
            break
        mid.append(i)
        mid_bytes += size(i)
    if not mid:
        mid = rest[:1]
    big = rest[len(mid):]
    if not small or not big:
        raise GenerationError("model mixed: could not form four non-empty shards")

    def shard(name, indexes, mode, metadata):
        return ShardPlan(name, [tensors[i] for i in indexes], mode, metadata)

    return [
        shard("model-00001-of-00004.safetensors", [first], "aligned", {"format": "pt"}),
        shard("model-00002-of-00004.safetensors", small, "odd", None),
        shard("model-00003-of-00004.safetensors", mid, "pad8", {"format": "pt"}),
        shard("model-00004-of-00004.safetensors", big, "odd", None),
    ]


def plan_many_tensors(seed, scale):
    rng = np.random.default_rng([seed, MODEL_IDS["many_tensors"], 0xFFFF])
    used = set()
    count = max(1000, int(100_000 * scale))
    dtypes = ["BF16", "F32", "U8", "F16", "I32"]
    tensors = []
    for key in range(count):
        dtype = dtypes[int(rng.integers(0, len(dtypes)))]
        if key % 5 == 4:
            name = realistic_name(key, used).rsplit(".", 1)[0] + ".weight_scale"
            if name in used:
                name += ".dup%d" % key
            used.add(name)
            tensors.append(TensorSpec(name, "F32", (1,), "random", key))
            continue
        elements_count = int(math.exp(rng.uniform(0.0, math.log(128))))
        shape = (elements_count,) if key % 3 else (1 + elements_count % 7, max(1, elements_count // 4))
        tensors.append(TensorSpec(realistic_name(key, used), dtype, shape, "random", key))
    for extra in range(max(2, int(24 * scale))):
        key = count + extra
        tensors.append(TensorSpec("big.%d" % extra, "U8", (4 * MIB,), "random", key))
    order = rng.permutation(len(tensors))
    return [ShardPlan("model.safetensors", [tensors[i] for i in order], "odd", {"format": "pt"})]


def plan_giant_tensor(seed, scale):
    rng = np.random.default_rng([seed, MODEL_IDS["giant_tensor"], 0xFFFF])
    giant_bytes = max(1, int(2 * GIB * scale))
    rows = max(1, int(round(128256 * scale)))
    used = set()
    key = 0

    def smalls(count):
        nonlocal key
        out = []
        for _ in range(count):
            out.append(random_tensor(rng, key, used, 64 * 1024))
            key += 1
        return out

    first = smalls(3) + [TensorSpec("big.2GiB.u8", "U8", (giant_bytes,), "random", 1000)] + smalls(2)
    second = [TensorSpec("model.embed_tokens.weight", "BF16", (rows, 8192), "random", 1001)] + smalls(3)
    return [
        ShardPlan("model-00001-of-00002.safetensors", first, "aligned", {"format": "pt"}),
        ShardPlan("model-00002-of-00002.safetensors", second, "odd", None),
    ]


def plan_many_files(seed, scale, files):
    rng = np.random.default_rng([seed, MODEL_IDS["many_files"], 0xFFFF])
    modes = ("aligned", "pad8", "odd")
    shards = []
    key = 0
    for index in range(files):
        kind = rng.random()
        # One of each is forced, so a small --files never lacks a header-only or a single-byte file by chance.
        if index == 0:
            kind = 0.0
        elif index == 1:
            kind = 0.02
        tensors = []
        if kind < 0.01:
            pass  # header only, no tensors
        elif kind < 0.03:
            tensors.append(TensorSpec("f%05d.single" % index, "U8", (1,), "random", key))
            key += 1
        else:
            max_bytes = 10 * MIB if kind > 0.995 else 64 * 1024
            for j in range(int(rng.integers(1, 4))):
                spec = random_tensor(rng, key, set(), max_bytes, name="f%05d.t%d" % (index, j))
                tensors.append(spec)
                key += 1
        metadata = {"format": "pt"} if rng.random() < 0.5 else None
        shards.append(ShardPlan("model-%05d-of-%05d.safetensors" % (index + 1, files), tensors,
                                modes[int(rng.integers(0, 3))], metadata))
    return shards


def make_plan(model, seed, scale, files):
    if model == "mixed":
        return plan_mixed(seed, scale)
    if model == "many_tensors":
        return plan_many_tensors(seed, scale)
    if model == "giant_tensor":
        return plan_giant_tensor(seed, scale)
    return plan_many_files(seed, scale, files)


# ----------------------------------------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------------------------------------

def build_header(shard, offsets):
    header = {}
    if shard.metadata is not None:
        header["__metadata__"] = shard.metadata
    for spec, (start, end) in zip(shard.tensors, offsets):
        header[spec.name] = {"dtype": spec.dtype, "shape": list(spec.shape), "data_offsets": [start, end]}
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    length = len(raw)
    if shard.header_mode == "aligned":
        # the data starts on a 4096 boundary of the file
        padded = length + (-(8 + length)) % 4096
    elif shard.header_mode == "pad8":
        padded = length + (-length) % 8
    else:
        # a data start that is not a multiple of 8, so tensors are misaligned for their dtype
        padded = length + (3 - length) % 8
    return raw + b" " * (padded - length)


def write_shard(path, shard, seed, model):
    offsets = []
    position = 0
    for spec in shard.tensors:
        size = nbytes_of(spec)
        offsets.append((position, position + size))
        position += size
    header = build_header(shard, offsets)
    entries = []
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(header)))
        out.write(header)
        for spec, (start, end) in zip(shard.tensors, offsets):
            crc = 0
            for chunk in iter_tensor_bytes(seed, model, spec):
                out.write(chunk)
                crc = zlib.crc32(chunk, crc)
            entries.append({"name": spec.name, "dtype": spec.dtype, "shape": list(spec.shape),
                            "start": start, "end": end, "crc32": crc, "gen": spec.gen, "key": spec.key})
    return {"path": os.path.basename(path), "size": 8 + len(header) + position, "header_len": len(header),
            "data_start": 8 + len(header), "header_mode": shard.header_mode, "metadata": shard.metadata,
            "tensors": entries}


def generate_model(args):
    model, out_dir, seed, scale, files, verify, dtypes = args
    use_dtypes(dtypes)  # in this process: the pool may spawn workers that do not share the parent's state
    timer = PhaseTimer()
    directory = os.path.join(out_dir, model)
    os.makedirs(directory, exist_ok=True)
    with timer.phase("%s.plan" % model):
        plan = make_plan(model, seed, scale, files)
    manifest_files = []
    with timer.phase("%s.generate+write" % model):
        for shard in plan:
            entry = write_shard(os.path.join(directory, shard.filename), shard, seed, model)
            manifest_files.append(entry)
    data_bytes = sum(e["end"] - e["start"] for f in manifest_files for e in f["tensors"])
    timer.records[-1]["bytes"] = data_bytes
    manifest = {
        "generator_version": GENERATOR_VERSION, "numpy_version": np.__version__, "seed": seed, "model": model, "scale": scale,
        "dtypes": list(active_dtypes),
        "num_files": len(manifest_files),
        "num_tensors": sum(len(f["tensors"]) for f in manifest_files),
        "total_data_bytes": data_bytes,
        "largest_tensor_bytes": max([e["end"] - e["start"] for f in manifest_files for e in f["tensors"]] or [0]),
        "files": manifest_files,
    }
    with timer.phase("%s.manifest" % model):
        with open(os.path.join(out_dir, model + ".manifest.json"), "w") as handle:
            json.dump(manifest, handle)
    if verify:
        with timer.phase("%s.verify" % model, nbytes=data_bytes):
            verify_model(out_dir, model)
    return model, timer.records


# ----------------------------------------------------------------------------------------------------------
# verification: the fixtures are checked before anything is streamed, so a failure in the test is never a
# failure of the fixture
# ----------------------------------------------------------------------------------------------------------

def load_manifest(out_dir, model):
    with open(os.path.join(out_dir, model + ".manifest.json")) as handle:
        return json.load(handle)


def verify_model(out_dir, model):
    manifest = load_manifest(out_dir, model)
    directory = os.path.join(out_dir, model)
    problems = []
    for entry in manifest["files"]:
        path = os.path.join(directory, entry["path"])
        try:
            size = os.path.getsize(path)
            if size != entry["size"]:
                problems.append("%s: size %d, manifest says %d" % (path, size, entry["size"]))
                continue
            with open(path, "rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
                if header_len != entry["header_len"]:
                    problems.append("%s: header length %d, manifest says %d" % (path, header_len,
                                                                               entry["header_len"]))
                    continue
                header = json.loads(handle.read(header_len).decode("utf-8"))
                header.pop("__metadata__", None)
                expected_names = [t["name"] for t in entry["tensors"]]
                if sorted(header) != sorted(expected_names):
                    problems.append("%s: header names differ from the manifest" % path)
                    continue
                position = 0
                for tensor in entry["tensors"]:
                    declared = header[tensor["name"]]
                    if (declared["dtype"] != tensor["dtype"] or declared["shape"] != tensor["shape"]
                            or declared["data_offsets"] != [tensor["start"], tensor["end"]]
                            or tensor["start"] != position):
                        problems.append("%s: tensor %s differs from the manifest or leaves a hole"
                                        % (path, tensor["name"]))
                    position = tensor["end"]
                    handle.seek(entry["data_start"] + tensor["start"])
                    crc, left = 0, tensor["end"] - tensor["start"]
                    while left:
                        chunk = handle.read(min(left, CHUNK))
                        if not chunk:
                            problems.append("%s: file ends inside tensor %s" % (path, tensor["name"]))
                            break
                        crc = zlib.crc32(chunk, crc)
                        left -= len(chunk)
                    if crc != tensor["crc32"]:
                        problems.append("%s: tensor %s differs from what was written (crc32)"
                                        % (path, tensor["name"]))
        except (OSError, ValueError, KeyError) as error:
            problems.append("%s: %s" % (path, error))
    problems.extend(check_determinism(manifest))
    if problems:
        raise VerifyError("model %s: %d problem(s):\n  %s" % (model, len(problems), "\n  ".join(problems[:20])))


def check_determinism(manifest, samples=3):
    """Regenerate a few small tensors from the seed and compare with the manifest."""
    problems = []
    tensors = [t for f in manifest["files"] for t in f["tensors"] if 0 < t["end"] - t["start"] <= 8 * MIB]
    step = max(1, len(tensors) // samples)
    for tensor in tensors[::step][:samples]:
        spec = TensorSpec(tensor["name"], tensor["dtype"], tuple(tensor["shape"]), tensor["gen"], tensor["key"])
        crc = 0
        for chunk in iter_tensor_bytes(manifest["seed"], manifest["model"], spec):
            crc = zlib.crc32(chunk, crc)
        if crc != tensor["crc32"]:
            problems.append("tensor %s does not regenerate from the seed" % tensor["name"])
    return problems


def check_coverage(manifests):
    """Across the models: every dtype, every rank 0-5, every edge size, and empty tensors if EMPTY_TENSORS (if
    not, none may be present)."""
    dtypes, ranks, edges, empty = set(), set(), set(), False
    for manifest in manifests:
        for entry in manifest["files"]:
            for tensor in entry["tensors"]:
                dtypes.add(tensor["dtype"])
                ranks.add(len(tensor["shape"]))
                if tensor["dtype"] == "U8" and len(tensor["shape"]) == 1:
                    edges.add(tensor["shape"][0])
                empty = empty or tensor["end"] == tensor["start"]
    missing = []
    wanted = set().union(*(manifest["dtypes"] for manifest in manifests))
    missing += ["dtype " + d for d in sorted(wanted) if d not in dtypes]
    missing += ["rank %d" % r for r in range(6) if r not in ranks]
    missing += ["edge size %d" % s for s in EDGE_SIZES if s not in edges]
    if EMPTY_TENSORS and not empty:
        missing.append("an empty tensor")
    if missing:
        raise VerifyError("the models do not cover: " + ", ".join(missing))
    if empty and not EMPTY_TENSORS:
        raise VerifyError("the models contain an empty tensor, which EMPTY_TENSORS = False leaves out")


# ----------------------------------------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------------------------------------

def run_generate(args):
    models = parse_models(args.models)
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()] if args.dtypes else list(DTYPE_NAMES)
    use_dtypes(dtypes)
    jobs = [(m, args.out, args.seed, args.scale, args.files, not args.no_verify, dtypes) for m in models]
    print("generating models %s, seed %d, scale %g, numpy %s, generator version %d, in %s" % (
        ",".join(models), args.seed, args.scale, np.__version__, GENERATOR_VERSION, args.out), flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for model, model_records in pool.map(generate_model, jobs):
            records.extend(model_records)
            manifest = load_manifest(args.out, model)
            print("model %s: %d files, %d tensors, %.2f GiB" % (
                model, manifest["num_files"], manifest["num_tensors"], manifest["total_data_bytes"] / GIB),
                flush=True)
    if not args.no_verify and set(models) == set(MODEL_NAMES):
        check_coverage([load_manifest(args.out, m) for m in models])
    dump_jsonl(records, os.path.join(args.out, "timing-gen.jsonl"))
    print(report(records), flush=True)


def run_verify(args):
    models = parse_models(args.models)
    for model in models:
        verify_model(args.out, model)
    if set(models) == set(MODEL_NAMES):
        check_coverage([load_manifest(args.out, m) for m in models])
    print("fixtures in %s verified" % args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", required=True)
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--scale", type=float, default=1.0,
                     help="size of the mixed, many_tensors and giant_tensor models relative to the full size")
    gen.add_argument("--files", type=int, default=5000, help="number of files of the many_files model")
    gen.add_argument("--models", default=",".join(MODEL_NAMES), help="comma separated model names")
    gen.add_argument("--dtypes", default="", help="comma separated dtypes to draw from (default: all of them)")
    gen.add_argument("--no-verify", action="store_true")
    gen.set_defaults(func=run_generate)
    ver = sub.add_parser("verify")
    ver.add_argument("--out", required=True)
    ver.add_argument("--models", default=",".join(MODEL_NAMES), help="comma separated model names")
    ver.set_defaults(func=run_verify)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (GenerationError, VerifyError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
