# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn the runs into one summary per test, the table and the results file."""
import json
import platform
import statistics
import time

from . import LOADERS, group_of, loader_of, markdown
from .profile import ceilings_for, nearest


def summarize(name, runs, model, strategies, profile, remote, warm):
    """One test: the medians over its runs and the ceiling, then the runs themselves. A test is ok only
    when every run gave a result and none returned wrong data."""
    good = [r for r in runs if "failed" not in r]
    s = {"test": name, "loader": loader_of(name), "group": group_of(name),
         "ok": bool(good) and len(good) == len(runs) and all("errors" not in r for r in good)}
    if good:
        med = statistics.median(r["gibps"] for r in good)
        s.update(load_s=round(statistics.median(r["load_s"] for r in good), 2), gibps=round(med, 2),
                 gibps_min=min(r["gibps"] for r in good), gibps_max=max(r["gibps"] for r in good))
        if all("cpu_s" in r for r in good):
            s.update(cpu_s_per_gib=round(statistics.median(r["cpu_s"] for r in good) / model["gib"], 2),
                     peak_rss_gib=max(r["peak_rss_gib"] for r in good))
        if all("download_s" in r for r in good):
            s["download_s"] = round(statistics.median(r["download_s"] for r in good), 2)
        if strategies:
            s["fs_strategy"] = sorted(strategies)
        scored, candidates = ceilings_for(name, profile, remote, warm)
        if scored:
            s["ceiling"] = {"name": scored, "gibps": candidates[scored],
                            "pct": round(100 * med / candidates[scored], 1), "nearest": nearest(med, candidates)}
    s["runs"] = runs
    return s


def add_speedups(results):
    """Give each test its speedup over safetensors to the same destination, when both gave a result."""
    base = {s["group"]: s["load_s"] for s in results if s["loader"] == "safetensors" and "load_s" in s}
    for s in results:
        if "load_s" in s and s["load_s"] and s["group"] in base:
            s["speedup"] = round(base[s["group"]] / s["load_s"], 2)


def print_table(results, pkgs):
    print(f"\nstreamer: {', '.join(pkgs) or 'not loaded'}")
    print(f"{'test':20s} {'loader':15s} {'load s':>8s} {'GiB/s':>7s} {'speedup':>8s} {'% ceil':>7s} "
          f"{'cpu s/GiB':>10s} {'rss GiB':>8s}  valid")
    for s in results:
        if "gibps" not in s:
            print(f"{s['test']:20s} {LOADERS[s['loader']]:15s} FAILED")
            continue
        c = s.get("ceiling")
        print(f"{s['test']:20s} {LOADERS[s['loader']]:15s} {s['load_s']:8.2f} {s['gibps']:7.1f} "
              f"{(str(s['speedup']) + 'x') if 'speedup' in s else 'n/a':>8s} "
              f"{(str(round(c['pct'])) + '%') if c else 'n/a':>7s} "
              f"{_cell(s.get('cpu_s_per_gib'), '10.2f')} {_cell(s.get('peak_rss_gib'), '8.1f')}  "
              f"{'yes' if s['ok'] else 'NO'}")


def _cell(value, spec):
    width = int(spec.split(".")[0])
    return f"{'n/a':>{width}s}" if value is None else f"{value:{spec}}"


def calibration_of(profile, path):
    """What the results keep of the machine profile: everything but each stage's single runs."""
    if not profile:
        return None
    return {"path": path, **{k: v for k, v in profile.items() if k not in ("schema", "raw_gibps")}}


def report(a, runs_of, strategy_of, model, pkgs, versions, environment, env, profile, profile_path, remote):
    """Print the table and write the results file and its Markdown report. True when every test is ok."""
    results = [summarize(n, runs, model, strategy_of[n], profile, remote, a.warm) for n, runs in runs_of.items()]
    add_speedups(results)
    print_table(results, pkgs)
    settings = {"repeats": a.repeats, "warm": a.warm}
    if a.nproc:
        settings["nproc"] = a.nproc
    if env:
        settings["env"] = env
    out = {"schema": 3, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
           "streamer": pkgs, "versions": versions, "environment": environment, "model": model,
           "settings": settings, "calibration": calibration_of(profile, profile_path), "results": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out} and {markdown.write(out, a.out)}")
    return all(s["ok"] for s in results)
