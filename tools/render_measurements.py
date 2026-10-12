#!/usr/bin/env python
"""renders results/measurements.md from the json files the benches wrote
(design 1.12): a header block (lscpu model name and the flags that matter for
int8 inference, nproc, free -m, python and onnxruntime versions, the commit
read from .git), the parameter table (results/params.json), the component
bench tables (results/bench_components_<host>.json), the replay bench with the
real-time targets of design 1.10 next to the measured numbers and a pass/fail
per bucket (results/replay_bench_<host>.json), and a "laptop (user-run)"
section whose every cell reads [laptop: not yet measured] until the user
pastes the block the scripts print. no number in the output is typed here.

usage: python tools/render_measurements.py --out results/measurements.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOT_YET = "[laptop: not yet measured]"
FLAGS = ("avx2", "fma", "avx512f", "avx512bw", "avx512_vnni", "amx_tile", "amx_int8", "amx_bf16")
DURATIONS = ("2", "3.3", "5", "7.15", "10", "15")
TOKENS = ("16", "32", "64")
BUCKET_ORDER = ("le5s", "le8s", "le15s")
LAPTOP_COMMANDS = [
    "python tools/count_params.py --models models/ --heads heads/ --out results/params.md",
    "python tools/bench_components.py --threads 1,2,4 --clips bench_clips/ --out results/bench_components_$(hostname).json",
    'python tools/replay_bench.py --clips "proxy/*.wav" --n 20 --realtime --out results/replay_bench_$(hostname).json',
    "python tools/render_measurements.py --out results/measurements.md",
]


# -- header ----------------------------------------------------------------

def run(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def lscpu_info() -> dict:
    model, flags = "", ""
    for line in run(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            model = line.split(":", 1)[1].strip()
        elif line.startswith("Flags:"):
            flags = line.split(":", 1)[1]
    if not model or not flags:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name") and not model:
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not flags:
                    flags = line.split(":", 1)[1]
        except OSError:
            pass
    fs = set(flags.split())
    return {"model_name": model or "unknown", "flags": {f: f in fs for f in FLAGS}}


def free_m() -> str:
    out = run(["free", "-m"]).strip()
    if out:
        return out
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0]) // 1024
        return f"MemTotal {mem.get('MemTotal')} MB, MemAvailable {mem.get('MemAvailable')} MB (from /proc/meminfo)"
    except (OSError, ValueError):
        return "unknown"


def commit_hash(root: Path) -> str:
    """HEAD's commit read from the .git files; no git command is run."""
    git = root / ".git"
    try:
        if git.is_file():  # worktree: "gitdir: <path>"
            git = Path(git.read_text().split(":", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(":", 1)[1].strip()
        p = git / ref
        if p.exists():
            return p.read_text().strip()[:12] + f" ({ref.rsplit('/', 1)[-1]})"
        common = git / "commondir"
        base = (git / common.read_text().strip()).resolve() if common.exists() else git
        for line in (base / "packed-refs").read_text().splitlines() if (base / "packed-refs").exists() else []:
            if line.endswith(" " + ref):
                return line.split()[0][:12] + f" ({ref.rsplit('/', 1)[-1]})"
    except (OSError, ValueError, IndexError):
        pass
    return "unknown"


def header(root: Path) -> list[str]:
    import onnxruntime as ort
    cpu = lscpu_info()
    nproc = run(["nproc"]).strip() or str(os.cpu_count())
    flags = ", ".join(f"{f} {'yes' if v else 'no'}" for f, v in cpu["flags"].items())
    return ["## Container (this host)", "",
            f"- host: `{socket.gethostname()}`, rendered {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}, commit `{commit_hash(root)}`",
            f"- lscpu: {cpu['model_name']}; nproc {nproc}; flags of interest: {flags}",
            "- free -m:", "", "```", free_m(), "```", "",
            f"- python {platform.python_version()} ({platform.platform()}), onnxruntime {ort.__version__}", ""]


# -- tables ----------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fmt(v, nd: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def ms_m(n) -> str:
    return "-" if n is None else f"{n / 1e6:.3f} M"


def params_section(res: dict | None) -> list[str]:
    out = ["### Parameters (`results/params.json`, written by `tools/count_params.py`)", ""]
    if res is None:
        return out + ["[not yet measured: run `python tools/count_params.py`]", ""]
    out += [f"Generated {res.get('generated_at')}.", "", "| Component | File | Params | Trained | Note |", "|---|---|---|---|---|"]
    for r in res["rows"]:
        out.append(f"| {r['component']} | {r['file'] if r['file'] == '-' else '`' + r['file'] + '`'} | {ms_m(r['params'])} | {r['trained']} | {r['note']} |")
    pend = f"; pending: {', '.join(res['heads_pending'])}" if res.get("heads_pending") else ""
    out.append(f"| **Total on the inference path** | | **{ms_m(res['total_inference_path'])}** | | "
               f"{res['total_inference_path'] / res['cap'] * 100:.2f} % of the 6 B cap{pend} |")
    if res.get("piper"):
        p = res["piper"]
        out.append(f"| {p['component']} | `{p['file']}` | {ms_m(p['params'])} | {p['trained']} | {p['note']} |")
        out.append(f"| Total with Piper enabled | | {ms_m(res['total_with_piper'])} | | |")
    return out + [""]


def bench_section(b: dict, path: Path) -> list[str]:
    m = b.get("machine", {})
    out = [f"### Component bench (`{path.name}`, written by `tools/bench_components.py`)", "",
           (f"Generated {b.get('generated_at')} on `{b.get('host')}`; {b.get('runs')} timed runs after {b.get('warmup')} warm-up, p50 / p95 in ms; "
           f"load average {b.get('loadavg_start')} at start, {b.get('loadavg_end')} at the end; wall {b.get('wall_s')} s. "
           f"Clips: {', '.join(f'{c['label']} s ({c['source']})' for c in b.get('clips', []))} from `{b.get('source_wav')}`."), ""]
    th = sorted(b.get("by_threads", {}), key=int)
    labels = [c["label"] for c in b.get("clips", [])] or list(DURATIONS)

    def table(title: str, key: str, cols: list[str], sub: str = "p50", nd: int = 1) -> list[str]:
        rows = [f"**{title}** ({sub}, ms)", "", "| threads | " + " | ".join(f"{c} s" if key != "text_ms" else f"{c} tok" for c in cols) + " |",
                "|---|" + "---|" * len(cols)]
        for t in th:
            r = b["by_threads"][t].get(key, {})
            rows.append(f"| {t} | " + " | ".join(fmt(r.get(c, {}).get(sub), nd) for c in cols) + " |")
        return rows + [""]

    out += table("SenseVoice-small int8, onnxruntime call", "sensevoice_ms", labels)
    out += table("SenseVoice-small int8, onnxruntime call", "sensevoice_ms", labels, "p95")
    out += table("SenseVoice total (front end + model + CTC decode)", "sensevoice_total_ms", labels)
    out += table("Front end (fbank + LFR + CMVN)", "frontend_ms", labels)
    out += table("Prosody (parselmouth)", "prosody_ms", labels)
    out += table("RoBERTa-base text expert by input length", "text_ms", list(TOKENS))
    out += table("RoBERTa-base text expert by input length", "text_ms", list(TOKENS), "p95")
    out += ["**VAD, heads, rules, selector, session loads**", "",
            "| threads | VAD per 512-sample frame (1 thread) | heads (p50 ms) | selector (p50 ms) | load: sensevoice / text / vad / heads / selector (s) |",
            "|---|---|---|---|---|"]
    for t in th:
        r = b["by_threads"][t]
        heads = ", ".join(f"{k} {fmt(v.get('p50'), 3)}" for k, v in r.get("heads_ms", {}).items())
        ld = r.get("load_s", {})
        out.append(f"| {t} | {fmt(r.get('vad_frame_ms', {}).get('p50'), 3)} / p95 {fmt(r.get('vad_frame_ms', {}).get('p95'), 3)} | {heads} | "
                   f"{fmt(r.get('selector_ms', {}).get('p50'), 3)} | {ld.get('sensevoice')} / {ld.get('text')} / {ld.get('vad')} / {ld.get('heads')} / {ld.get('selector')} |")
    out += ["", "Model files on disk: " + ", ".join(f"`{k}` {fmt(v / 2**20, 1)} MB" if v else f"`{k}` missing" for k, v in b.get("model_files", {}).items())
            + (("; heads: " + ", ".join(f"`{k}` {v / 1024:.0f} KB" for k, v in b["head_files"].items())) if b.get("head_files") else "") + ".", ""]
    if m:
        out += [(f"Machine recorded by the bench: {m.get('cpu', {}).get('model_name')}, nproc {m.get('cpu', {}).get('nproc')}, "
                f"python {m.get('python')}, onnxruntime {m.get('onnxruntime')}, numpy {m.get('numpy')}."), ""]
    return out


def pf(v) -> str:
    return "n/a" if v is None else ("pass" if v else "**FAIL**")


def replay_section(r: dict, path: Path) -> list[str]:
    c, res, comp, mem = r["config"], r["results"], r["compute"], r["memory"]
    out = [f"### Replay bench (`{path.name}`, written by `tools/replay_bench.py`)", "",
           (f"Generated {r.get('generated_at')} on `{r.get('host')}`: {c['n_clips']} clips from `{c['clip_source']}` "
           f"(durations p50 {fmt(c['clip_duration_s'].get('p50'), 2)} s, max {fmt(c['clip_duration_s'].get('max'), 2)} s), "
           f"{'real-time pace' if c['realtime'] else 'as fast as possible'}, {c['tail_s']} s of silence after each clip, one stream, "
           f"{c['warmup_clips']} warm-up clip(s) not measured; service with {c['threads']} SenseVoice threads, {c['text_threads']} RoBERTa threads, "
           f"endpoint silence {c['endpoint_silence_ms']} ms, provisionals {c['provisional_hz']} Hz over a {c['provisional_window_s']} s window, "
           f"fusion mode `{c['fusion_mode']}`, heads {c['heads'] or 'none'}. Models loaded in {r['load_s']} s. "
           f"Load average {r['loadavg_start']} at start, {r['loadavg_end']} at the end. "
           f"{res['n_finals']} finals, {res['n_responses']} responses, {res['n_provisionals']} provisionals, errors: {res['errors'] or 'none'}"
           f"{'' if r.get('finished_cleanly', True) else '; the run did not drain cleanly'}."), "",
           "**Final state after the endpoint, by utterance length (design 1.10 targets at 4 threads)**", "",
           ("| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | "
           "stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |"), "|---|---|---|---|---|---|---|---|---|"]
    for name in BUCKET_ORDER:
        b = res["by_bucket"].get(name)
        if b is None:
            continue
        t = b["target_ms"]
        if b["n"] == 0:
            out.append(f"| {b['description']} | 0 | no utterances in this bucket | {t['p50']:.0f} / {t['p95']:.0f} | - | - | - | - | n/a |")
            continue
        s, cl, q, st = b["service_total_ms"], b["client_ms"], b["response_ms"], b["stages_p50_ms"]
        out.append(f"| {b['description']} | {b['n']} | {fmt(s['p50'])} / {fmt(s['p95'])} | {t['p50']:.0f} / {t['p95']:.0f} | "
                   f"{fmt(cl.get('p50'))} / {fmt(cl.get('p95'))} | {fmt(q.get('p50'))} / {fmt(q.get('p95'))} | "
                   f"{fmt(st.get('frontend'))} / {fmt(st.get('sensevoice'))} / {fmt(st.get('prosody'))} / {fmt(st.get('text_encoder'))} / {fmt(st.get('heads'))} | "
                   f"{fmt(b.get('provisional_in_flight_p50_ms'))} | {pf(b['pass'])} |")
    a = res["all"]
    out += [(f"| all | {a['service_total_ms'].get('n', 0)} | {fmt(a['service_total_ms'].get('p50'))} / {fmt(a['service_total_ms'].get('p95'))} | - | "
            f"{fmt(a['client_ms'].get('p50'))} / {fmt(a['client_ms'].get('p95'))} | {fmt(a['response_ms'].get('p50'))} / {fmt(a['response_ms'].get('p95'))} | "
            f"{' / '.join(fmt(a['stages_p50_ms'].get(k)) for k in ('frontend', 'sensevoice', 'prosody', 'text_encoder', 'heads'))} | - | - |"), ""]
    rp, cad = res["reply_after_final_ms"], res["provisionals"]
    out += ["**Other targets of design 1.10**", "", "| target | measured | pass |", "|---|---|---|",
            f"| reply text within {rp['target_p95_ms']} ms of the final state (p95) | p50 {fmt(rp.get('p50'), 3)}, p95 {fmt(rp.get('p95'), 3)}, max {fmt(rp.get('max'), 3)} ms over {rp.get('n', 0)} replies | {pf(rp['pass'])} |",
            (f"| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 {fmt(cad['full_window_interval_s'].get('p50'), 3)} s (n {cad['full_window_interval_s'].get('n', 0)}), "
            f"tail-window gap p50 {fmt(cad['tail_window_interval_s'].get('p50'), 3)} s (n {cad['tail_window_interval_s'].get('n', 0)}), "
            f"full-to-tail switch gap p50 {fmt(cad.get('full_to_tail_switch_interval_s', {}).get('p50'), 3)} s, first provisional "
            f"{fmt(cad.get('first_provisional_offset_s', {}).get('p50'), 3)} s after speech start, {cad['n_full']} full + {cad['n_tail']} tail provisionals, "
            f"{cad['utterances_over_1p5s_with_a_provisional']} utterances over 1.5 s got one, dropped {cad['dropped_provisionals']}, provisional SenseVoice p50 {fmt(cad['sensevoice_ms'].get('p50'))} ms | {pf(cad['pass'])} |"),
            (f"| sustained compute below {comp['target_max']} s per second of audio | worker {fmt(comp['worker_compute_s_per_audio_s'], 3)} s per audio s "
            f"({fmt(comp['worker_compute_s_per_speech_s'], 3)} per speech s); whole process {fmt(comp['process_cpu_s_per_audio_s'], 3)} cpu s per audio s, "
            f"cpu % sampled mean {fmt(comp['process_cpu_percent_sampled'].get('mean'))} max {fmt(comp['process_cpu_percent_sampled'].get('max'))} | {pf(comp['pass'])} |"),
            f"| memory | rss {mem['rss_mb_after_load']} MB after load, peak {mem['peak_rss_mb']} MB during the run (service + client + monitor in one process) | - |",
            (f"| worker lag | max health queue_s {fmt(res.get('max_queue_s'), 3)} s; clips without a final: {res['clips_without_final'] or 'none'}; "
            f"clips split into several finals: {res['clips_with_several_finals'] or 'none'}; degraded finals {res['finals_degraded']}, empty transcripts {res['finals_empty_text']} | - |"), ""]
    out += ["Notes: " + " ".join(n + "." for n in r.get("notes", [])), ""]
    return out


def laptop_section() -> list[str]:
    out = ["## Laptop (user-run)", "",
           ("Every cell below stays `[laptop: not yet measured]` until the user runs the same scripts on the laptop and pastes the block they print "
           "(design 1.12). Nothing here is typed from memory."), "", "Commands:", "", "```"] + LAPTOP_COMMANDS + ["```", "",
           "| item | laptop |", "|---|---|",
           f"| lscpu model name, flags (avx512_vnni, amx) | {NOT_YET} |", f"| nproc, free -m, python, onnxruntime | {NOT_YET} |",
           f"| parameters on the inference path | {NOT_YET} |"]
    for t in ("1", "2", "4"):
        out.append(f"| SenseVoice p50 ms at {t} thread(s), {' / '.join(DURATIONS)} s | {NOT_YET} |")
    out += [f"| front end p50 ms, {' / '.join(DURATIONS)} s | {NOT_YET} |",
            f"| RoBERTa p50 ms at 16 / 32 / 64 tokens, 1 / 2 / 4 threads | {NOT_YET} |",
            f"| VAD per frame, heads, selector, session loads | {NOT_YET} |"]
    for name, desc in (("le5s", "up to 5 s"), ("le8s", "5 to 8 s"), ("le15s", "8 s to the 15 s cap")):
        out.append(f"| final after endpoint, {desc}: p50 / p95 ms and pass | {NOT_YET} |")
    out += [f"| reply after final p95 ms | {NOT_YET} |", f"| provisional cadence, dropped provisionals | {NOT_YET} |",
            f"| compute s per audio s, cpu %, peak rss | {NOT_YET} |", f"| 3-minute mic.py session (notes) | {NOT_YET} |", ""]
    return out


def render(root: Path, results: Path) -> str:
    out = ["# Measurements", "",
           ("Rendered by `python tools/render_measurements.py --out results/measurements.md` from the json files under `results/`; "
           "the header comes from `lscpu`, `nproc`, `free -m`, the running interpreter and `.git`. Targets are the real-time definition of "
           "design 1.10; every measured number was written by the script named in its section."), ""]
    out += header(root)
    out += params_section(load_json(results / "params.json"))
    benches = sorted(glob.glob(str(results / "bench_components_*.json")))
    if benches:
        for p in benches:
            b = load_json(Path(p))
            out += bench_section(b, Path(p)) if b else [f"[`{Path(p).name}` unreadable]", ""]
    else:
        out += ["### Component bench", "", "[not yet measured: run `python tools/bench_components.py --threads 1,2,4 --clips bench_clips/`]", ""]
    replays = sorted(glob.glob(str(results / "replay_bench_*.json")))
    if replays:
        for p in replays:
            r = load_json(Path(p))
            out += replay_section(r, Path(p)) if r else [f"[`{Path(p).name}` unreadable]", ""]
    else:
        out += ["### Replay bench", "", "[not yet measured: run `python tools/replay_bench.py --clips \"proxy/*.wav\" --n 20 --realtime`]", ""]
    out += laptop_section()
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="render results/measurements.md from the bench json files")
    ap.add_argument("--out", type=Path, default=ROOT / "results/measurements.md")
    ap.add_argument("--results", type=Path, default=None, help="directory of the json files (default: the --out directory)")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    results = a.results or a.out.parent
    md = render(a.root, results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    print(f"wrote {a.out} ({len(md.splitlines())} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
