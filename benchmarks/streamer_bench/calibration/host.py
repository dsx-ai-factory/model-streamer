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
