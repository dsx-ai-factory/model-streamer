# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Results as a Markdown report: the test setting, then a table of load time, throughput and speedup per
loader. Several results files make one report, a block of rows per model.

    python3 -m streamer_bench.markdown bench_results.json                 writes bench_results.md again
    python3 -m streamer_bench.markdown a.json b.json --out report.md      one report for both
"""
import argparse
import json
import os

from . import LOADERS, TESTS


def _num(v, digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _on_s3(out):
    return out["model"]["path"].startswith("s3://")


def _model(out):
    return os.path.basename(out["model"]["path"].rstrip("/"))


def _loader(s):
    """The loader's name; "download + ..." when the model was first downloaded from S3 to local disk."""
    name = LOADERS.get(s.get("loader"), s["test"])
    return f"download + {name}" if "download_s" in s else name


def _load_time(s):
    time = _num(s["load_s"], 2)
    if "download_s" in s:
        time += f' ({_num(s["download_s"], 2)} download + {_num(s["load_s"] - s["download_s"], 2)} load)'
    return time


def _destination(out, group):
    """Where a group of tests loads to, in words: host memory, one GPU, or the GPUs of the ranks."""
    if group == "cpu":
        return "host memory"
    gpus = out.get("environment", {}).get("gpus") or []
    count = out["settings"].get("nproc", 1) if group == "dist" else 1
    return f"{count} × {gpus[0] if gpus else 'GPU'}"


def cases(outs):
    """[(results file, group, its tests)] in the order they were run: one block of table rows each."""
    found = []
    for out in outs:
        groups = list(dict.fromkeys(s["group"] for s in out["results"]))
        found += [(out, g, [s for s in out["results"] if s["group"] == g]) for g in groups]
    return found


def table(outs):
    lines = ["| Model | Destination | Loader | Load time (s) | Throughput (GiB/s) | Speedup |",
             "|---|---|---|---|---|---|"]
    for out, group, tests in cases(outs):
        best = max([s.get("speedup", 0) for s in tests] + [0])
        for i, s in enumerate(tests):
            first = f"| {_model(out)} | {_destination(out, group)} " if i == 0 else "| | "
            if "gibps" not in s:
                lines.append(f"{first}| {_loader(s)} | failed | | |")
                continue
            speedup = f'{s["speedup"]:.2f}x' if "speedup" in s else "n/a"
            if best > 1 and s.get("speedup") == best:
                speedup = f"**{speedup}**"
            invalid = "" if s["ok"] else " (invalid data)"
            lines.append(f'{first}| {_loader(s)} | {_load_time(s)}{invalid} | {_num(s["gibps"])} | {speedup} |')
    return "\n".join(lines)


def _each(outs, value):
    """The distinct values of something across the results files, in order, as one string."""
    return ", ".join(dict.fromkeys(str(v) for v in map(value, outs) if v not in (None, "", [])))


def _gpus(out):
    gpus = out.get("environment", {}).get("gpus")
    return f"{len(gpus)} × {gpus[0]}" if gpus else None


def _rate(out, ceiling):
    return ((out.get("calibration") or {}).get("ceilings_gibps") or {}).get(ceiling)


def _storage(out):
    storage = out.get("environment", {}).get("storage")
    if not storage:
        return None
    rate = _rate(out, "disk_direct_read_parallel")
    return f'{storage["type"]} on {storage["mount"]}' + (f", {rate:.0f} GiB/s parallel direct read" if rate else "")


def _s3(out):
    if not _on_s3(out):
        return None
    cal = out.get("calibration") or {}
    rate = _rate(out, "s3_get_parallel")
    endpoint = ((cal.get("host") or {}).get("s3") or {}).get("endpoint")
    return (endpoint or "S3") + (f", {rate:.0f} GiB/s parallel GET with s5cmd" if rate else "")


def _model_size(out):
    return f'{_model(out)} ({out["model"]["gib"]} GiB)' if "gib" in out["model"] else _model(out)


def _runs(out):
    st = out["settings"]
    return f'median of {st["repeats"]} per loader, {"warm" if st["warm"] else "cold"} page cache'


def setting(outs):
    """What the numbers were measured on and with."""
    versions = {}
    for out in outs:
        versions.update(out.get("versions", {}))
    pkgs = _each(outs, lambda o: ", ".join(o["streamer"]))
    rows = [
        ("Host", _each(outs, lambda o: o["host"])),
        ("GPUs", _each(outs, _gpus)),
        ("CPUs, memory", _each(outs, lambda o: f'{o["environment"].get("cpus")} CPUs, '
                                               f'{o["environment"].get("mem_total_gib")} GiB')),
        ("Kernel", _each(outs, lambda o: o["environment"].get("kernel"))),
        ("Storage", _each(outs, _storage)),
        ("S3", _each(outs, _s3)),
        ("Models", _each(outs, _model_size)),
        ("Runs", _each(outs, _runs)),
        ("Settings", _each(outs, lambda o: " ".join(f"{k}={v}" for k, v in o["settings"].get("env", {}).items()))),
        ("safetensors", versions.get("safetensors")),
        ("instanttensor", versions.get("instanttensor")),
        ("runai_model_streamer", pkgs and f'{versions.get("runai-model-streamer", "")} (imported from {pkgs})'),
        ("torch", versions.get("torch")),
    ]
    return "\n".join(f"- {k}: {v}" for k, v in rows if v)


def results(outs):
    """The tables: one for the models read from local storage and one for those on S3, each under a
    heading that names the storage and the rate measured for it."""
    parts = []
    for on_s3, what, ceiling, how in ((False, "local disk", "disk_direct_read_parallel", "parallel direct read"),
                                      (True, "S3", "s3_get_parallel", "parallel GET with s5cmd")):
        group = [o for o in outs if _on_s3(o) == on_s3]
        if not group:
            continue
        rates = [r for r in (_rate(o, ceiling) for o in group) if r]
        measured = f" ({min(rates):.0f} GiB/s {how})" if rates else ""
        parts.append(f"## Test on {what}{measured}\n\n{table(group)}")
    return "\n\n".join(parts)


def loads(outs):
    """How each test that ran loads the model, a line per test."""
    seen = dict.fromkeys(s["test"] for out in outs for s in out["results"])
    return "\n".join(f'- `{t}`: {TESTS[t]["about"]}' for t in seen if t in TESTS)


def render(outs):
    return f"""# Load benchmark

## Test setting

{setting(outs)}

{results(outs)}

- Throughput is the model's size over the load time.
- Speedup is against Safetensors to the same destination.
- With several GPUs, the load time is the time until the slowest GPU finishes.
- InstantTensor loads to GPUs only, so it has no host memory entry.
- Safetensors and InstantTensor read local files only. For a model on S3 their load time is the download
  to local disk with s5cmd plus the load from there, shown as the two parts; the streamer reads from S3
  directly.

## How each loader is run

{loads(outs)}
"""


def write(out, json_path):
    """Write one result's report next to its results file and return the report's path."""
    path = os.path.splitext(json_path)[0] + ".md"
    with open(path, "w") as f:
        f.write(render([out]))
    return path


def combine(json_paths, md_path):
    """One report for several results files. Returns its table."""
    outs = []
    for p in json_paths:
        with open(p) as f:
            outs.append(json.load(f))
    with open(md_path, "w") as f:
        f.write(render(outs))
    return table(outs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results files written by bench.py")
    ap.add_argument("--out", help="the report to write (default: next to a single results file, else bench_report.md)")
    a = ap.parse_args()
    target = a.out or (os.path.splitext(a.results[0])[0] + ".md" if len(a.results) == 1 else "bench_report.md")
    print(combine(a.results, target))
    print(f"\nwrote {target}")
