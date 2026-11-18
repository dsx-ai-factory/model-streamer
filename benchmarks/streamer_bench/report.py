# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn the runs into one summary per test, the table and the results file."""
import json
import platform
import statistics
import time

from . import html
from .profile import ceilings_for, nearest


def summarize(name, runs, model, strategies, profile, remote, warm):
    """One test: the medians over its runs and the ceiling, then the runs themselves. A test is ok only
    when every run gave a result and none returned wrong data."""
    good = [r for r in runs if "failed" not in r]
    s = {"test": name, "ok": bool(good) and len(good) == len(runs) and all("errors" not in r for r in good)}
    if good:
        med = statistics.median(r["gibps"] for r in good)
        s.update(load_s=round(statistics.median(r["load_s"] for r in good), 2), gibps=round(med, 2),
                 gibps_min=min(r["gibps"] for r in good), gibps_max=max(r["gibps"] for r in good),
                 cpu_s_per_gib=round(statistics.median(r["cpu_s"] for r in good) / model["gib"], 2),
                 peak_rss_gib=max(r["peak_rss_gib"] for r in good))
        if strategies:
            s["fs_strategy"] = sorted(strategies)
        scored, candidates = ceilings_for(name, profile, remote, warm)
        if scored:
            s["ceiling"] = {"name": scored, "gibps": candidates[scored],
                            "pct": round(100 * med / candidates[scored], 1), "nearest": nearest(med, candidates)}
    s["runs"] = runs
    return s


def print_table(results, pkgs):
    print(f"\nstreamer: {', '.join(pkgs) or 'not loaded'}")
    print(f"{'test':24s} {'GiB/s':>7s} {'range':>13s} {'load s':>7s} {'% ceil':>7s} {'cpu s/GiB':>10s} "
          f"{'rss GiB':>8s}  valid  nearest ceiling")
    for s in results:
        if "gibps" not in s:
            print(f"{s['test']:24s} FAILED")
            continue
        rng = f"{s['gibps_min']:.1f}-{s['gibps_max']:.1f}"
        c = s.get("ceiling")
        print(f"{s['test']:24s} {s['gibps']:7.1f} {rng:>13s} {s['load_s']:7.2f} "
              f"{(str(round(c['pct'])) + '%') if c else 'n/a':>7s} "
              f"{s['cpu_s_per_gib']:10.2f} {s['peak_rss_gib']:8.1f}  {'yes' if s['ok'] else 'NO ':5s}  "
              f"{c['nearest'] if c else 'n/a'}")


def report(a, runs_of, strategy_of, model, pkgs, env, profile, profile_path, remote):
    """Print the table and write the results file and its HTML page. True when every test is ok."""
    results = [summarize(n, runs, model, strategy_of[n], profile, remote, a.warm) for n, runs in runs_of.items()]
    print_table(results, pkgs)
    settings = {"repeats": a.repeats, "warm": a.warm}
    if a.nproc:
        settings["nproc"] = a.nproc
    if env:
        settings["env"] = env
    out = {"schema": 2, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
           "streamer": pkgs, "profile": profile_path, "model": model, "settings": settings, "results": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out} and {html.write(out, a.out)}")
    return all(s["ok"] for s in results)
