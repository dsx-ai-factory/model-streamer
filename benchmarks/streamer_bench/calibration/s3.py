# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S3, read with s5cmd into /dev/null: a native client, so the ceiling is set by the network and the
server, not by Python or by the streamer under test."""
import os
import statistics
import subprocess
import sys
import time

from .. import GIB


def s3_endpoint():
    for k in ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def s5cmd(*args):
    endpoint = s3_endpoint()
    return ["s5cmd"] + (["--endpoint-url", endpoint] if endpoint else []) + list(args)


def s3_objects(uri):
    """(uri, size) of every object under the prefix; only the *.safetensors ones when there are any."""
    prefix = uri.rstrip("/")
    out = subprocess.run(s5cmd("ls", prefix + "/*"), capture_output=True, text=True, check=True).stdout
    objs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2].isdigit():
            objs.append((f"{prefix}/{parts[3]}", int(parts[2])))
    st = [o for o in objs if o[0].endswith(".safetensors")]
    return st or objs


def s3_read(objs, concurrency, part_mib):
    """Read every object at once, `concurrency` parts in flight each, into /dev/null."""
    procs = [subprocess.Popen(s5cmd("cat", "-c", str(concurrency), "-p", str(part_mib), u),
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for u, _ in objs]
    for p in procs:
        _, err = p.communicate()
        if p.returncode:
            raise RuntimeError(f"s5cmd failed: {err.decode()[-400:]}")
    return sum(n for _, n in objs)


def measure(meter, a):
    """s3_get_single     one object, one part in flight (the per-connection floor)
    s3_get_parallel   every object at once, the best of a sweep over parts in flight and part size
    s3_first_run      the first read, kept apart: connection setup, server warm-up and network burst
                      credits can make it differ. Not a ceiling.

    Returns (what was read, the best sweep setting) for the profile."""
    objs = s3_objects(a.s3_uri)
    if not objs:
        sys.exit(f"no objects under {a.s3_uri}")
    if getattr(a, "s3_max_gib", None):  # enough objects to measure the storage, not the whole of a large model
        kept, size = [], 0
        for obj in objs:
            if kept and size + obj[1] > a.s3_max_gib * GIB:
                break
            kept.append(obj)
            size += obj[1]
        objs = kept
    total = sum(n for _, n in objs)
    endpoint = s3_endpoint()
    print(f"s3: {len(objs)} objects, {total / GIB:.2f} GiB under {a.s3_uri}"
          f" via {endpoint or 'the default AWS endpoint'}")
    t0 = time.perf_counter()
    s3_read(objs, 8, 8)
    meter.record("s3_first_run", [total / GIB / (time.perf_counter() - t0)], "(8 x 8 MiB, not a ceiling)")
    largest = [max(objs, key=lambda o: o[1])]
    meter.rate("s3_get_single", lambda: s3_read(largest, 1, 8), largest[0][1])
    sweep = {}
    for c in (int(x) for x in a.s3_concurrency.split(",")):
        for p in (int(x) for x in a.s3_part_mib.split(",")):
            key = f"s3_get_c{c}_p{p}"
            meter.rate(key, lambda: s3_read(objs, c, p), total)
            sweep[key] = (c, p)
    best = max(sweep, key=lambda k: statistics.median(meter.raw[k]))
    meter.record("s3_get_parallel", meter.raw[best], f"best: {best}")
    params = {"uri": a.s3_uri, "endpoint": endpoint, "gib": round(total / GIB, 3),
              "tool": subprocess.run(["s5cmd", "version"], capture_output=True, text=True).stdout.strip()}
    return params, {"concurrency_per_object": sweep[best][0], "part_mib": sweep[best][1], "objects": len(objs)}
