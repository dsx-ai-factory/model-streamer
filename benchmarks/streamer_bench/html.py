# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The results as one self-contained HTML page: a bar per test against its ceiling, then every number.

    python3 -m streamer_bench.html bench_results.json     writes bench_results.html again
"""
import json
import os
import sys
from html import escape

CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --bar:#2a78d6; --bad:#d03b3b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --bar:#3987e5; }
}
body { background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif; margin:0 auto; max-width:980px;
       padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:32px 0 8px; }
.meta { color:var(--ink2); margin:0; overflow-wrap:anywhere; } .note { color:var(--muted); font-size:12px; }
.chart { display:grid; grid-template-columns:max-content 1fr max-content; gap:10px 12px; align-items:center; }
.track { position:relative; height:14px; border-bottom:1px solid var(--grid); }
.bar { position:absolute; left:0; top:1px; height:12px; background:var(--bar); border-radius:0 4px 4px 0; }
.tick { position:absolute; top:-4px; height:22px; width:2px; background:var(--ink); }
.val { font-variant-numeric:tabular-nums; white-space:nowrap; } .val small { color:var(--ink2); }
.bad { color:var(--bad); font-weight:600; }
.wrap { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:5px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th { color:var(--ink2); font-weight:500; } th:first-child, td:first-child { text-align:left; }
table.kv { width:auto; } table.kv td, table.kv th { text-align:left; white-space:normal; overflow-wrap:anywhere; }
table.kv td:first-child { color:var(--ink2); white-space:nowrap; }
details { margin:6px 0; } summary { cursor:pointer; } pre { white-space:pre-wrap; overflow-wrap:anywhere; }
"""


def _num(v, digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _chart(results):
    if not results:
        return ""
    scale = max([s["gibps_max"] for s in results if "gibps" in s]
                + [s["ceiling"]["gibps"] for s in results if "ceiling" in s] + [1e-9])
    rows = []
    for s in results:
        name = escape(s["test"])
        if "gibps" not in s:
            rows.append(f'<div>{name}</div><div class="track"></div><div class="val bad">&#10007; failed</div>')
            continue
        c = s.get("ceiling")
        tip = f'{s["test"]}: {s["gibps"]} GiB/s, {s["load_s"]} s'
        tick, share = "", ""
        if c:
            tip += f', {c["pct"]}% of {c["name"]} ({c["gibps"]} GiB/s)'
            tick = f'<div class="tick" style="left:calc({100 * c["gibps"] / scale:.2f}% - 1px)"></div>'
            share = f' <small>{round(c["pct"])}% of {escape(c["name"])}</small>'
        bad = "" if s["ok"] else ' <span class="bad">&#10007; invalid</span>'
        rows.append(
            f'<div>{name}</div>'
            f'<div class="track" title="{escape(tip)}"><div class="bar" style="width:{100 * s["gibps"] / scale:.2f}%">'
            f'</div>{tick}</div>'
            f'<div class="val">{_num(s["gibps"])} GiB/s <small>{_num(s["load_s"], 2)} s</small>{share}{bad}</div>')
    return '<div class="chart">' + "".join(rows) + "</div>"


def _table(results):
    head = ("test", "GiB/s", "min", "max", "load s", "% of ceiling", "ceiling", "nearest ceiling", "cpu s/GiB",
            "peak RSS GiB", "strategy", "valid")
    rows = []
    for s in results:
        c = s.get("ceiling") or {}
        cells = [escape(s["test"])]
        if "gibps" in s:
            cells += [_num(s["gibps"], 2), _num(s["gibps_min"], 2), _num(s["gibps_max"], 2), _num(s["load_s"], 2),
                      _num(c.get("pct")), escape(c.get("name", "n/a")), escape(c.get("nearest") or "n/a"),
                      _num(s.get("cpu_s_per_gib"), 2), _num(s.get("peak_rss_gib")),
                      escape(", ".join(s.get("fs_strategy", [])))]
        else:
            cells += ["failed"] + [""] * 9
        cells.append("&#10003; yes" if s["ok"] else '<span class="bad">&#10007; no</span>')
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return ('<div class="wrap"><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join(rows) + "</table></div>")


def _runs(results):
    out = []
    for s in results:
        lines = []
        for i, r in enumerate(s["runs"]):
            if "failed" in r:
                lines.append(f"run {i}: FAILED\n{r['failed']}")
                continue
            line = f"run {i}: {r['load_s']} s, {r['gibps']} GiB/s"
            if "cpu_s" in r:
                line += f", cpu {r['cpu_s']} s, peak RSS {r['peak_rss_gib']} GiB"
            if "startup_s" in r:
                line += f", engine start {r['startup_s']} s, {r.get('gpu_gib', 'n/a')} GiB on the GPU"
            if "rank_s" in r:
                line += f", ranks {r['rank_s']} s, peak GPU {r['peak_gpu_gib']} GiB"
            lines.append(line)
            lines += [f"  error: {e}" for e in r.get("errors", [])]
            lines += [f"  warning: {w}" for w in r.get("warnings", [])]
        out.append(f"<details><summary>{escape(s['test'])}: {len(s['runs'])} runs</summary>"
                   f"<pre>{escape(chr(10).join(lines))}</pre></details>")
    return "".join(out)


def _vllm(out, dist):
    """A vLLM section: the same model loaded by vLLM with each of its loaders that was run, on one GPU
    or (dist) with tensor parallelism over the ranks."""
    results = [s for s in out["results"]
               if s["test"].startswith("vllm_") and s["test"].startswith("vllm_dist_") == dist]
    if not results:
        return ""
    speedup = out.get("vllm_dist_speedup" if dist else "vllm_speedup")
    headline = (f"With the streamer vLLM loads the model {speedup}x as fast as with its default loader."
                if speedup else "")
    version = out.get("versions", {}).get("vllm", "")
    title = (f"vLLM distributed model load by loader ({out['settings'].get('nproc')} ranks)" if dist
             else "vLLM model load by loader")
    return (f"<h2>{title}</h2>"
            f'<p class="meta">{escape(headline)}</p>{_chart(results)}'
            f'<p class="note">vLLM {escape(version)}. The time is the one vLLM logs as "Model loading took", '
            f"which covers building the model and loading its weights onto the GPU"
            f"{'; with several ranks it is the slowest rank' if dist else ''}. "
            f"The tensors are not validated in these tests.</p>")


def _pairs(rows):
    """A two-column table of name and value; rows without a value are left out."""
    rows = [(k, v) for k, v in rows if v not in (None, "", [])]
    return ('<div class="wrap"><table class="kv">'
            + "".join(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in rows)
            + "</table></div>")


def _environment(out):
    env, ver, st = out.get("environment", {}), out.get("versions", {}), out["settings"]
    storage = env.get("storage") or {}
    return ("<h2>Environment</h2>" + _pairs([
        ("host", out["host"]), ("kernel", env.get("kernel")), ("CPUs", env.get("cpus")),
        ("memory", f"{env['mem_total_gib']} GiB" if env.get("mem_total_gib") else None),
        ("GPUs", ", ".join(env.get("gpus") or [])),
        ("model storage", f"{storage.get('type')} on {storage.get('mount')}" if storage else None),
        ("transparent huge pages", env.get("thp")), ("Python", env.get("python"))])
        + "<h2>Versions</h2>" + _pairs([
            ("runai-model-streamer", ver.get("runai-model-streamer")),
            ("streamer imported from", ", ".join(out["streamer"])),
            ("vLLM", ver.get("vllm")), ("instanttensor", ver.get("instanttensor")),
            ("torch", ver.get("torch")), ("safetensors", ver.get("safetensors"))])
        + "<h2>Settings</h2>" + _pairs([
            ("runs per test", st["repeats"]), ("page cache", "warm" if st["warm"] else "cold"),
            ("ranks", st.get("nproc"))] + [(f"env {k}", v) for k, v in st.get("env", {}).items()]))


def _calibration(out):
    cal = out.get("calibration")
    if not cal:
        return ('<h2>Calibration</h2><p class="meta">No machine profile was used, so the tests have no '
                "ceilings.</p>")
    host, params = cal.get("host", {}), cal.get("params", {})
    storage = host.get("storage") or {}
    used = {s["ceiling"]["name"] for s in out["results"] if "ceiling" in s}
    rows = "".join(f"<tr><td>{escape(k)}</td><td>{_num(v, 2)}</td><td>{'scored against' if k in used else ''}</td></tr>"
                   for k, v in cal.get("ceilings_gibps", {}).items() if v is not None)
    notes = "".join(f'<p class="note">{escape(n)}</p>' for n in cal.get("notes", []))
    return ("<h2>Calibration</h2>" + _pairs([
        ("profile", cal.get("path")), ("measured", cal.get("created")), ("on host", host.get("hostname")),
        ("on storage", f"{storage.get('type')} on {storage.get('mount')}" if storage else None),
        ("disk test", f"{params['files']} files of {params['file_gib']} GiB" if params.get("files") else None),
        ("memory test", f"{params['mem_gib']} GiB, {params['threads']} threads" if params.get("threads") else None),
        ("runs per stage", params.get("repeats"))])
        + '<div class="wrap"><table class="kv"><tr><th>ceiling</th><th>GiB/s</th><th></th></tr>'
        + rows + "</table></div>" + notes)


def render(out):
    m, st = out["model"], out["settings"]
    settings = [f"{st['repeats']} runs per test", "warm page cache" if st["warm"] else "cold page cache"]
    if st.get("nproc"):
        settings.append(f"{st['nproc']} ranks")
    settings += [f"{k}={v}" for k, v in st.get("env", {}).items()]
    size = f" ({m['gib']} GiB, {m['tensors']} tensors)" if "gib" in m else ""
    has_ceiling = any("ceiling" in s for s in out["results"])
    note = ("The bar is the median load rate; the vertical mark is the ceiling the test is scored against."
            if has_ceiling else "The bar is the median load rate. No machine profile was used, so there are no ceilings.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Load benchmark: {escape(os.path.basename(m["path"].rstrip("/")))}</title><style>{CSS}</style></head><body>
<h1>Load benchmark</h1>
<p class="meta">{escape(m["path"])}{escape(size)}</p>
<p class="meta">{escape(out["host"])} &middot; {escape(out["created"])} &middot; {escape(", ".join(settings))}</p>
<p class="meta">streamer: {escape(", ".join(out["streamer"]) or "not loaded")}</p>
<h2>Load rate per test</h2>
{_chart([s for s in out["results"] if not s["test"].startswith("vllm_")])}
<p class="note">{note}</p>
{_vllm(out, False)}{_vllm(out, True)}
<h2>All numbers</h2>
{_table(out["results"])}
<h2>Runs</h2>
{_runs(out["results"])}
{_environment(out)}
{_calibration(out)}
</body></html>
"""


def write(out, json_path):
    """Write the page next to the results file and return its path."""
    path = os.path.splitext(json_path)[0] + ".html"
    with open(path, "w") as f:
        f.write(render(out))
    return path


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        print("wrote", write(json.load(f), sys.argv[1]))
