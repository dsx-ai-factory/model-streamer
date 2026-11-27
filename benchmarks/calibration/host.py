# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the profile says about the machine it was measured on."""
import os
import platform

from .memory import meminfo_gib, memory_budget_gib


def fs_type(path):
    """{"mount", "type"} of the filesystem that holds path, from /proc/mounts."""
    best, kind = "", "unknown"
    real = os.path.realpath(path)
    with open("/proc/mounts") as f:
        for line in f:
            _, mount, fstype = line.split()[:3]
            if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
                best, kind = mount, fstype
    return {"mount": best, "type": kind}


def read_text(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def host_info(path):
    info = {
        "hostname": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
        "cpus": len(os.sched_getaffinity(0)), "mem_total_gib": round(meminfo_gib("MemTotal") or 0, 1),
        "memory_budget_gib": round(memory_budget_gib(), 1),
        "thp": read_text("/sys/kernel/mm/transparent_hugepage/enabled"),
        "numa_nodes": len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")])
        if os.path.isdir("/sys/devices/system/node") else None,
        "storage": {"path": os.path.realpath(path), **fs_type(path)},
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def cpu_model():
    """(model name, sockets) from /proc/cpuinfo, or (None, None) where it does not say."""
    text = read_text("/proc/cpuinfo") or ""
    names = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("model name")]
    sockets = {line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("physical id")}
    return (names[0], len(sockets) or 1) if names else (None, None)


def memory_dimms(root="/sys/devices/system/edac/mc"):
    """{"type": "DDR5", "count": 32, "gib": 64.0} for the populated DIMMs, as the kernel's memory
    controller drivers report them, or None where they are not loaded. Needs no root, unlike the DMI
    tables."""
    import glob
    import re
    kinds, sizes = [], []
    for dimm in glob.glob(os.path.join(root, "mc*", "dimm*")):
        size = read_text(os.path.join(dimm, "size"))
        if not size or not size.strip().isdigit() or int(size) == 0:
            continue
        sizes.append(int(size))
        found = re.search(r"DDR\d?", read_text(os.path.join(dimm, "dimm_mem_type")) or "")
        if found:
            kinds.append(found.group(0))
    if not sizes or not kinds:
        return None
    return {"type": max(set(kinds), key=kinds.count), "count": len(sizes), "gib": round(max(sizes) / 1024, 1)}
