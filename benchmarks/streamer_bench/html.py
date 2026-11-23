# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Results as one self-contained HTML page: a table of load time, throughput and speedup per loader, with
the details folded away. Several results files make one page, a block of rows per model.

    python3 -m streamer_bench.html bench_results.json                  writes bench_results.html again
    python3 -m streamer_bench.html a.json b.json --out report.html     one page for both, and the same
                                                                       table as Markdown on stdout
"""
import argparse
import json
import os
from html import escape

from . import LOADERS, TESTS

CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --band:#f1f0ea; --bad:#d03b3b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --band:#242423; }
}
body { background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif; margin:0 auto; max-width:820px;
       padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 4px; } h3 { font-size:13px; margin:16px 0 6px; }
.meta { color:var(--ink2); margin:0 0 8px; overflow-wrap:anywhere; } .note { color:var(--muted); font-size:12px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th { color:var(--ink2); font-weight:500; } th:first-child, td:first-child { text-align:left; }
table.main { margin-top:20px; } table.main td.case { background:var(--band); text-align:left; white-space:normal; }
td.case small { color:var(--ink2); margin-left:6px; }
.name { position:relative; }
.info { display:inline-block; width:14px; height:14px; margin-left:6px; border:1px solid var(--muted);
        border-radius:50%; color:var(--ink2); font:italic 600 10px/14px Georgia,serif; text-align:center;
        vertical-align:1px; cursor:help; }
.info:hover, .info:focus { border-color:var(--ink); color:var(--ink); outline:none; }
.tip { display:none; position:absolute; left:0; bottom:calc(100% + 4px); z-index:2; width:min(340px, 78vw);
       padding:8px 10px; border-radius:6px; background:var(--ink); color:var(--bg); font-size:12px; line-height:1.4;
       white-space:normal; font-weight:400; }
.info:hover + .tip, .info:focus + .tip { display:block; }
.bad { color:var(--bad); font-weight:600; }
.wrap { overflow-x:auto; }
table.kv { width:auto; } table.kv td, table.kv th { text-align:left; white-space:normal; overflow-wrap:anywhere; }
table.kv td:first-child { color:var(--ink2); white-space:nowrap; }
details { margin:6px 0; } summary { cursor:pointer; } pre { white-space:pre-wrap; overflow-wrap:anywhere; }
details.fold { margin:0; border-top:1px solid var(--grid); padding:10px 0; } details.fold > summary { font-weight:600; }
.more { margin-top:40px; border-bottom:1px solid var(--grid); }
"""


def _num(v, digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _model(out):
    return os.path.basename(out["model"]["path"].rstrip("/"))


def _loader(s):
    return LOADERS.get(s.get("loader"), s["test"])


def _group(s):
    test = TESTS.get(s["test"], {})
    return s.get("group") or ("dist" if test.get("dist") else test.get("device"))


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
        groups = []
        for s in out["results"]:
            if _group(s) not in groups:
                groups.append(_group(s))
        found += [(out, g, [s for s in out["results"] if _group(s) == g]) for g in groups]
    return found


def _speedup(s, best):
    if "speedup" not in s:
        return "n/a"
    text = f'{s["speedup"]:.2f}x'
    return f"<b>{text}</b>" if best and s["speedup"] == best and best > 1 else text


def _main(outs):
    """The table: per model and destination, a row per loader."""
    rows = []
    for out, group, tests in cases(outs):
        size = f'{out["model"]["gib"]} GiB' if "gib" in out["model"] else ""
        rows.append(f'<tr><td class="case" colspan="4"><b>{escape(_model(out))}</b> to '
                    f'{escape(_destination(out, group))}<small>{escape(size)}</small></td></tr>')
        best = max([s.get("speedup", 0) for s in tests] + [0])
        for s in tests:
            about = TESTS.get(s["test"], {}).get("about", "")
            tip = (f'<span class="info" tabindex="0" aria-label="setup">i</span>'
                   f'<span class="tip" role="tooltip">{escape(about)}</span>') if about else ""
            name = f'<td class="name">{escape(_loader(s))}{tip}</td>'
            if "gibps" not in s:
                rows.append(f'<tr>{name}<td colspan="3" class="bad">&#10007; failed</td></tr>')
                continue
            bad = "" if s["ok"] else ' <span class="bad">&#10007; invalid</span>'
            rows.append(f'<tr>{name}<td>{_num(s["load_s"], 2)}{bad}</td><td>{_num(s["gibps"])}</td>'
                        f'<td>{_speedup(s, best)}</td></tr>')
    return ('<table class="main"><tr><th>Loader</th><th>Load time (s)</th><th>Throughput (GiB/s)</th>'
            "<th>Speedup</th></tr>" + "".join(rows) + "</table>")


def markdown(outs):
    """The same table as Markdown, for a document."""
    lines = ["| Model | Destination | Loader | Load time (s) | Throughput (GiB/s) | Speedup |",
             "|---|---|---|---|---|---|"]
    for out, group, tests in cases(outs):
        for i, s in enumerate(tests):
            first = f"| {_model(out)} | {_destination(out, group)} " if i == 0 else "| | "
            if "gibps" not in s:
                lines.append(f"{first}| {_loader(s)} | failed | | |")
                continue
            speedup = f'{s["speedup"]:.2f}x' if "speedup" in s else "n/a"
            lines.append(f'{first}| {_loader(s)} | {_num(s["load_s"], 2)} | {_num(s["gibps"])} | {speedup} |')
    return "\n".join(lines)


def _grid(head, rows):
    return ('<div class="wrap"><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")


def _numbers(results):
    """What the main table leaves out: the spread, CPU, memory, the ceiling and the read strategy."""
    rows = []
    for s in results:
        if "gibps" not in s:
            rows.append([escape(s["test"]), "failed"] + [""] * 6)
            continue
        c = s.get("ceiling") or {}
        rows.append([escape(s["test"]), f'{_num(s["gibps_min"])}&ndash;{_num(s["gibps_max"])}',
                     _num(s.get("cpu_s_per_gib"), 2), _num(s.get("peak_rss_gib")), escape(c.get("name", "n/a")),
                     f'{_num(c["pct"], 0)}%' if c else "n/a", escape(", ".join(s.get("fs_strategy", [])) or "n/a"),
                     "&#10003;" if s["ok"] else '<span class="bad">&#10007; no</span>'])
    return _grid(("test", "GiB/s range", "cpu s/GiB", "peak RSS GiB", "ceiling", "% of ceiling", "read strategy",
                  "valid"), rows)


def _runs(results):
    out = []
    for s in results:
        lines = []
        for i, r in enumerate(s["runs"]):
            if "failed" in r:
                lines.append(f"run {i}: FAILED\n{r['failed']}")
                continue
            line = f"run {i}: {r['load_s']} s, {r['gibps']} GiB/s, cpu {r['cpu_s']} s, peak RSS {r['peak_rss_gib']} GiB"
            if "rank_s" in r:
                line += f", ranks {r['rank_s']} s, peak GPU {r['peak_gpu_gib']} GiB"
            lines.append(line)
            lines += [f"  error: {e}" for e in r.get("errors", [])]
            lines += [f"  warning: {w}" for w in r.get("warnings", [])]
        out.append(f"<details><summary>{escape(s['test'])}: {len(s['runs'])} runs</summary>"
                   f"<pre>{escape(chr(10).join(lines))}</pre></details>")
    return "".join(out)


def _pairs(rows):
    """A two-column table of name and value; rows without a value are left out."""
    rows = [(k, v) for k, v in rows if v not in (None, "", [])]
    return ('<div class="wrap"><table class="kv">'
            + "".join(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in rows)
            + "</table></div>")


def _environment(out):
    env, ver, st = out.get("environment", {}), out.get("versions", {}), out["settings"]
    storage = env.get("storage") or {}
    return ("<h3>Machine</h3>" + _pairs([
        ("host", out["host"]), ("measured", out["created"]), ("kernel", env.get("kernel")), ("CPUs", env.get("cpus")),
        ("memory", f"{env['mem_total_gib']} GiB" if env.get("mem_total_gib") else None),
        ("GPUs", ", ".join(env.get("gpus") or [])),
        ("model", out["model"]["path"]),
        ("model storage", f"{storage.get('type')} on {storage.get('mount')}" if storage else None),
        ("transparent huge pages", env.get("thp")), ("Python", env.get("python"))])
        + "<h3>Versions</h3>" + _pairs([
            ("runai-model-streamer", ver.get("runai-model-streamer")),
            ("streamer imported from", ", ".join(out["streamer"])),
            ("instanttensor", ver.get("instanttensor")), ("safetensors", ver.get("safetensors")),
            ("torch", ver.get("torch"))])
        + "<h3>Settings</h3>" + _pairs([
            ("runs per test", st["repeats"]), ("page cache", "warm" if st["warm"] else "cold"),
            ("ranks", st.get("nproc"))] + [(f"env {k}", v) for k, v in st.get("env", {}).items()]))


def _calibration(out):
    cal = out.get("calibration")
    if not cal:
        return '<h3>Calibration</h3><p class="meta">No machine profile was used, so the tests have no ceilings.</p>'
    host, params = cal.get("host", {}), cal.get("params", {})
    storage = host.get("storage") or {}
    used = {s["ceiling"]["name"] for s in out["results"] if "ceiling" in s}
    rows = "".join(f"<tr><td>{escape(k)}</td><td>{_num(v, 2)}</td><td>{'scored against' if k in used else ''}</td></tr>"
                   for k, v in cal.get("ceilings_gibps", {}).items() if v is not None)
    notes = "".join(f'<p class="note">{escape(n)}</p>' for n in cal.get("notes", []))
    return ("<h3>Calibration</h3>" + _pairs([
        ("profile", cal.get("path")), ("measured", cal.get("created")), ("on host", host.get("hostname")),
        ("on storage", f"{storage.get('type')} on {storage.get('mount')}" if storage else None),
        ("disk test", f"{params['files']} files of {params['file_gib']} GiB" if params.get("files") else None),
        ("memory test", f"{params['mem_gib']} GiB, {params['threads']} threads" if params.get("threads") else None),
        ("runs per stage", params.get("repeats"))])
        + '<div class="wrap"><table class="kv"><tr><th>ceiling</th><th>GiB/s</th><th></th></tr>'
        + rows + "</table></div>" + notes)


def _details(out):
    dest = ", ".join(dict.fromkeys(_destination(out, _group(s)) for s in out["results"]))
    return (f'<details class="fold"><summary>{escape(_model(out))} to {escape(dest)}: runs, machine and '
            f'calibration</summary><h3>Numbers</h3>{_numbers(out["results"])}<h3>Runs</h3>{_runs(out["results"])}'
            f"{_environment(out)}{_calibration(out)}</details>")


def render(outs):
    first = outs[0]
    hosts = ", ".join(dict.fromkeys(o["host"] for o in outs))
    repeats = ", ".join(dict.fromkeys(str(o["settings"]["repeats"]) for o in outs))
    cache = "warm" if all(o["settings"]["warm"] for o in outs) else "cold"
    title = _model(first) if len(outs) == 1 else f"{len(outs)} models"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Load benchmark: {escape(title)}</title><style>{CSS}</style></head><body>
<h1>Load benchmark</h1>
<p class="meta">{escape(hosts)} &middot; {escape(first["created"][:10])} &middot; median of {escape(repeats)} runs
per loader, {cache} page cache</p>
{_main(outs)}
<p class="note">Throughput is the model's size over the load time. Speedup is against Safetensors to the same
destination. With several GPUs the load time is the slowest rank's. Point at the <i>i</i> next to a loader to see
how it is set up.</p>
<div class="more">
{"".join(_details(o) for o in outs)}
</div>
</body></html>
"""


def write(out, json_path):
    """Write one result's page next to its results file and return the page's path."""
    path = os.path.splitext(json_path)[0] + ".html"
    with open(path, "w") as f:
        f.write(render([out]))
    return path


def combine(json_paths, html_path):
    """One page for several results files. Returns the same table as Markdown."""
    outs = []
    for p in json_paths:
        with open(p) as f:
            outs.append(json.load(f))
    with open(html_path, "w") as f:
        f.write(render(outs))
    return markdown(outs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results files written by bench.py")
    ap.add_argument("--out", help="the page to write (default: next to a single results file, else bench_report.html)")
    a = ap.parse_args()
    target = a.out or (os.path.splitext(a.results[0])[0] + ".html" if len(a.results) == 1 else "bench_report.html")
    table = combine(a.results, target)
    print(table)
    print(f"\nwrote {target}")
