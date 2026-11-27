# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The machine profile: finding or making one, and scoring a test against its ceilings."""
import json
import math
import os
import platform
import shutil
import subprocess
import sys

from . import CALIBRATE, DEFAULT_PROFILE, S3_CALIBRATE_GIB, TESTS, is_dist


def load_json(path):
    with open(path) as f:
        return json.load(f)


def profile_mismatch(profile, a, remote):
    """Why the profile does not describe this host and this model's storage, or None when it does."""
    host = profile.get("host", {})
    if host.get("hostname") != platform.node():
        return f"is from host {host.get('hostname')}, this is {platform.node()}"
    if remote:
        if not profile.get("ceilings_gibps", {}).get("s3_get_parallel"):
            return "has no S3 ceilings"
        bucket = lambda uri: uri[len("s3://"):].split("/")[0]  # one bucket is one storage, whatever the prefix
        if bucket((host.get("s3") or {}).get("uri", "")) != bucket(a.model_dir):
            return "was measured against another S3 bucket"
        from calibration.s3 import s3_endpoint
        if (host.get("s3") or {}).get("endpoint") != s3_endpoint():  # the same bucket name on another service
            return "was measured against another S3 endpoint"
        return None
    from calibration.host import fs_type
    mount = fs_type(a.model_dir)["mount"]
    measured = (host.get("storage") or {}).get("mount")
    if measured != mount:
        return f"was measured on {measured or 'no storage'}, the model is on {mount}"
    return None


def get_profile(a, remote):
    """(profile, its path), or (None, None) when the run goes without ceilings."""
    if a.profile:
        profile = load_json(a.profile)
        why = profile_mismatch(profile, a, remote)
        if why:
            print(f"warning: {a.profile} {why}; its ceilings may not apply", file=sys.stderr)
        return profile, os.path.abspath(a.profile)
    if a.no_calibrate:
        return None, None
    if os.path.exists(DEFAULT_PROFILE):
        profile = load_json(DEFAULT_PROFILE)
        why = profile_mismatch(profile, a, remote)
        if not why:
            print(f"using {DEFAULT_PROFILE} (--profile FILE to use another)")
            return profile, os.path.abspath(DEFAULT_PROFILE)
        print(f"{DEFAULT_PROFILE} {why}")
    hint = "run calibrate.py yourself and pass --profile FILE. Continuing without ceilings."
    where = a.scratch_dir if remote else a.model_dir  # the disk stage writes its test files here
    cmd = [sys.executable, CALIBRATE, "--path", where, "--out", DEFAULT_PROFILE]
    if remote:
        if not shutil.which("s5cmd"):
            print(f"no machine profile for this S3 model, and calibrating S3 needs s5cmd: {hint}", file=sys.stderr)
            return None, None
        # a short sweep over part of the model: enough for a ceiling without reading a large model many times
        cmd += ["--s3-uri", a.model_dir, "--s3-max-gib", str(S3_CALIBRATE_GIB), "--s3-concurrency", "4,8",
                "--s3-part-mib", "8,32"]
    if not shutil.which("fio"):
        print(f"no machine profile, and calibrating the disk needs fio on PATH (apt-get install fio): {hint}",
              file=sys.stderr)
        return None, None
    if not os.access(where, os.W_OK):
        print(f"no machine profile, and {where} cannot be written to calibrate on it: {hint}", file=sys.stderr)
        return None, None
    print(f"calibrating this machine on {where}{' and ' + a.model_dir if remote else ''} first; the profile is "
          f"kept in {DEFAULT_PROFILE} (--profile FILE reuses one, --no-calibrate skips this)", flush=True)
    if subprocess.run(cmd).returncode:
        print("warning: calibration failed; continuing without ceilings", file=sys.stderr)
        return None, None
    return load_json(DEFAULT_PROFILE), os.path.abspath(DEFAULT_PROFILE)


def ceilings_for(test, profile, remote=False):
    """(name of the ceiling this test is scored against, or None; the ceilings it may be nearest to).

    Host tests and the distributed tests are scored against the storage: disk_direct_read_parallel,
    or s3_get_parallel for an S3 source. A single-process GPU test is
    scored against the lower of that and h2d_pinned."""
    if not profile:
        return None, {}
    c = {k: v for k, v in profile["ceilings_gibps"].items() if v}
    storage = "s3_get_parallel" if remote else "disk_direct_read_parallel"
    if not is_dist(test) and TESTS[test]["device"] == "cuda":
        scored = [k for k in (storage, "h2d_pinned") if k in c]
        return (min(scored, key=c.get) if scored else None), c
    # the profile's h2d numbers are one stream to GPU 0, which says nothing about N ranks at once
    return (storage if storage in c else None), {k: v for k, v in c.items() if not k.startswith("h2d_")}


def nearest(achieved, ceilings):
    """The ceiling closest to what was achieved: a hint at what limited the run, not a proof."""
    if not ceilings or achieved <= 0:  # a rate that rounds to zero is near nothing
        return None
    return min(ceilings, key=lambda k: abs(math.log(ceilings[k] / achieved)))
