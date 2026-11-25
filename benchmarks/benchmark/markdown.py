# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Results as a Markdown report: the test setting, then a table of load time, throughput and speedup per
loader for the models on local disk, one for the loads inside vLLM and one for the models on S3. Several
results files make one report, a block of rows per model.

    python3 -m benchmark.markdown bench_results.json                 writes bench_results.md again
    python3 -m benchmark.markdown a.json b.json --out report.md      one report for both
"""
import argparse
import json
import os

from . import LOADERS


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
    count = out["settings"].get("nproc", 1) if group in ("dist", "vllm_dist") else 1
    return f"{count} × {gpus[0] if gpus else 'GPU'}"


def _in_vllm(group):
    return group.startswith("vllm")


def cases(outs, vllm=False):
    """[(results file, group, its tests)] in the order they were run: one block of table rows each.
    The loads inside vLLM are apart from the others."""
    found = []
    for out in outs:
        groups = list(dict.fromkeys(s["group"] for s in out["results"] if _in_vllm(s["group"]) == vllm))
        found += [(out, g, [s for s in out["results"] if s["group"] == g]) for g in groups]
    return found


def table(outs, vllm=False):
    found = cases(outs, vllm)
    versus = any("speedup" in s for _, _, tests in found for s in tests)  # nothing to compare with: no column
    lines = ["| Model | Destination | Loader | Load time (s) | Throughput (GiB/s) |" + (" Speedup |" if versus else ""),
             "|---|---|---|---|---|" + ("---|" if versus else "")]
    for out, group, tests in found:
        best = max([s.get("speedup", 0) for s in tests] + [0])
        for i, s in enumerate(tests):
            first = f"| {_model(out)} | {_destination(out, group)} " if i == 0 else "| | "
            if "gibps" not in s:
                lines.append(f"{first}| {_loader(s)} | failed | |" + (" |" if versus else ""))
                continue
            speedup = f'{s["speedup"]:.2f}x' if "speedup" in s else "n/a"
            invalid = "" if s["ok"] else " (invalid data)"
            cells = [_loader(s), _load_time(s) + invalid, _num(s["gibps"])] + ([speedup] if versus else [])
            if best > 1 and s.get("speedup") == best:  # the fastest loader's whole line stands out
                cells = [f"**{c}**" for c in cells]
            lines.append(first + "| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _each(outs, value):
    """The distinct values of something across the results files, in order, as one string."""
    return ", ".join(dict.fromkeys(str(v) for v in map(value, outs) if v not in (None, "", [])))


def _gpus(out):
    """The GPUs the tests loaded to: none for a run to host memory only."""
    used = dict.fromkeys(_destination(out, s["group"]) for s in out["results"] if s["group"] != "cpu")
    return ", ".join(used) or None


def _streamer(outs, versions):
    """The streamer's version: "local build" when it was not the installed package."""
    return versions.get("runai-model-streamer") if any(o["streamer"] for o in outs) else None


def _rate(out, ceiling):
    return ((out.get("calibration") or {}).get("ceilings_gibps") or {}).get(ceiling)


def _storage(out):
    storage = out.get("environment", {}).get("storage")
    if not storage:
        return None
    rate = _rate(out, "disk_direct_read_parallel")
    return storage["type"] + (f", {rate:.0f} GiB/s parallel direct read" if rate else "")


def _s3(out):
    if not _on_s3(out):
        return None
    rate = _rate(out, "s3_get_parallel")
    return f"{rate:.0f} GiB/s parallel GET with s5cmd" if rate else "not calibrated"


def _gpus_in_use(out):
    busy = out.get("environment", {}).get("gpus_in_use")
    return ", ".join(f"GPU {i} ({gib} GiB)" for i, gib in busy.items()) if busy else None


def _cpus(out):
    env = out.get("environment", {})
    model = f'{env["cpu_sockets"]} × {env["cpu_model"]}, ' if env.get("cpu_model") else ""
    return f'{model}{env.get("cpus")} CPUs'


def _memory(out):
    env = out.get("environment", {})
    dimms = env.get("memory_dimms") or {}
    kind = f' {dimms["type"]} ({dimms["count"]} × {dimms["gib"]:.0f} GiB)' if dimms else ""
    return f'{env.get("mem_total_gib")} GiB{kind}'


def _bytes(n):
    """A size in the unit that reads best, for tensors from a few bytes to gigabytes."""
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _dtypes(shares):
    """The data types that hold at least 1% of the bytes, with their share when there are several."""
    main = {k: v for k, v in shares.items() if v >= 0.01} or shares
    return ", ".join(k if len(main) == 1 else f"{k} {100 * v:.0f}%" for k, v in main.items())


def models(outs):
    """A row per model: its size, files and tensors, the tensors' average and largest size, and data types."""
    rows = {}
    for out in outs:
        m = out["model"]
        rows.setdefault(_model(out), "| {} | {} | {} | {} | {} | {} | {} |".format(
            _model(out), f'{m["gib"]} GiB' if "gib" in m else "n/a", m.get("files", "n/a"), m.get("tensors", "n/a"),
            _bytes(m["tensor_bytes_average"]) if "tensor_bytes_average" in m else "n/a",
            _bytes(m["tensor_bytes_largest"]) if "tensor_bytes_largest" in m else "n/a",
            _dtypes(m["dtypes"]) if m.get("dtypes") else "n/a"))
    return "\n".join(["| Model | Size | Files | Tensors | Average tensor | Largest tensor | Data type |",
                      "|---|---|---|---|---|---|---|", *rows.values()])


def _runs(out):
    return f'median of {out["settings"]["repeats"]} per loader'


def _page_cache(outs):
    """The page cache before the runs from local disk: evicted and checked before each. The runs that
    started with part of the model still in it are counted, over every results file: the page must not
    call them cold. An S3 model has no such check and no entry: the streamer does not read files, and the
    other loaders load what they have just downloaded."""
    local = [out for out in outs if not _on_s3(out)]
    if not local:
        return None
    left = sum(any("still in the page cache" in w for w in run.get("warnings", []))
               for out in local for s in out["results"] for run in s["runs"])
    return f'not fully evicted before {left} run{"s" if left > 1 else ""}' if left else "cold before every run"


def _settings(outs):
    """The variables set for the measured processes. One that had different values names the models each
    value was used for."""
    values = {}  # variable -> {value: [models]}
    for out in outs:
        for k, v in out["settings"].get("env", {}).items():
            models = values.setdefault(k, {}).setdefault(v, [])
            if _model(out) not in models:
                models.append(_model(out))
    return ", ".join(
        f"{k}={next(iter(by))}" if len(by) == 1 else
        f"{k}=" + ", ".join(f"{v} ({' and '.join(models)})" for v, models in by.items())
        for k, by in values.items())


def setting(outs):
    """What the numbers were measured on and with: the hardware, the storage's measured rates and the
    versions. Names that only mean something on the machine that ran it (host, mount point, endpoint,
    paths) are left to the results file."""
    versions = {}
    for out in outs:
        versions.update(out.get("versions", {}))
    ran = {s["loader"] for out in outs for s in out["results"]}
    rows = [
        ("GPUs", _each(outs, _gpus)),
        ("GPUs already in use by something else during the run", _each(outs, _gpus_in_use)),
        ("CPUs", _each(outs, _cpus)),
        ("Memory", _each(outs, _memory)),
        ("Kernel", _each(outs, lambda o: o["environment"].get("kernel"))),
        ("Storage", _each(outs, _storage)),
        ("S3", _each(outs, _s3)),
        ("Runs", _each(outs, _runs)),
        ("Page cache, for the models on local disk", _page_cache(outs)),
        ("Settings", _settings(outs)),
        ("safetensors", versions.get("safetensors") if "safetensors" in ran else None),
        ("instanttensor", versions.get("instanttensor") if "instanttensor" in ran else None),
        ("runai_model_streamer", _streamer(outs, versions)),
        ("vllm", versions.get("vllm") if any(_in_vllm(s["group"]) for o in outs for s in o["results"])
         else None),
        ("torch", versions.get("torch")),
    ]
    return "\n".join(f"- {k}: {v}" for k, v in rows if v)


def _download_note(group):
    """How the loaders that read only local files got an S3 model, when any of them ran."""
    names = list(dict.fromkeys(LOADERS["safetensors" if s["loader"].startswith("safetensors") else s["loader"]]
                               for out in group for s in out["results"] if "download_s" in s))
    if not names:
        return ""
    who = " and ".join(names) + (" read" if len(names) > 1 else " reads")
    them = "them" if len(names) > 1 else "it"
    best = next((b for b in (((o.get("calibration") or {}).get("params") or {}).get("s3_best") for o in group)
                 if b), None)
    how = (f', with the settings the calibration measured as fastest ({best["objects"]} files at once, '
           f'{best["concurrency_per_object"]} parts of {best["part_mib"]} MiB each)') if best else ""
    return (f"\n\n{who} local files only. For {them} the model is first downloaded to local disk with "
            f"s5cmd{how}, and then loaded as the download leaves it, in the page cache or not; the load time is the "
            "two together.")


def results(outs):
    """The tables: one for the models read from local storage, one for the loads inside vLLM and one for
    the models on S3, each under a heading that names the storage and the rate measured for it."""
    parts = []
    for on_s3, what, ceiling, how in ((False, "local disk", "disk_direct_read_parallel", "parallel direct read"),
                                      (True, "S3", "s3_get_parallel", "parallel GET with s5cmd")):
        group = [o for o in outs if _on_s3(o) == on_s3]
        if not group:
            continue
        rates = [r for r in (_rate(o, ceiling) for o in group) if r]
        measured = f" ({min(rates):.0f} GiB/s {how})" if rates else ""
        if cases(group):
            alone = [dict(o, results=[s for s in o["results"] if not _in_vllm(s["group"])]) for o in group]
            parts.append(f"## Test on {what}{measured}\n\n{table(group)}" + (_download_note(alone) if on_s3 else ""))
        if cases(group, vllm=True):
            inside = [dict(o, results=[s for s in o["results"] if _in_vllm(s["group"])]) for o in group]
            parts.append(f"## Test in vLLM, from {what}{measured}\n\n{table(group, vllm=True)}\n\n"
                         "The load time is vLLM's weight loading step, on the slowest rank when there are several."
                         + (_download_note(inside) if on_s3 else ""))
    return "\n\n".join(parts)


def render(outs):
    return f"""# Load benchmark

## Models

{models(outs)}

## Test setting

{setting(outs)}

{results(outs)}
"""


def write(out, json_path):
    """Write one result's report next to its results file and return the report's path."""
    path = os.path.splitext(json_path)[0] + ".md"
    with open(path, "w") as f:
        f.write(render([out]))
    return path


def combine(json_paths, md_path):
    """One report for several results files. Returns its tables, each under its heading."""
    outs = []
    for p in json_paths:
        with open(p) as f:
            outs.append(json.load(f))
    with open(md_path, "w") as f:
        f.write(render(outs))
    return results(outs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results files written by bench.py")
    ap.add_argument("--out", help="the report to write (default: next to a single results file, else bench_report.md)")
    a = ap.parse_args()
    target = a.out or (os.path.splitext(a.results[0])[0] + ".md" if len(a.results) == 1 else "bench_report.md")
    print(combine(a.results, target))
    print(f"\nwrote {target}")
