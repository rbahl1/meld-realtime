#!/usr/bin/env python
"""renders results/measurements.md from the json files the benches wrote
(design 1.12): a block for the host that renders (lscpu model name and the
flags that matter for int8 inference, nproc, free -m, python and onnxruntime
versions, the commit read from .git), then a container section (the parameter
table from results/params.json and every bench json whose recorded host is
--container-host) and a laptop section (every bench json from any other host,
plus results/laptop_notes.md when it exists). the replay sections put the
real-time targets of design 1.10 next to the measured numbers with a pass/fail
per bucket. without laptop json files the laptop section is a table whose every
cell reads [laptop: not yet measured]. no number in the output is typed here.

usage: python tools/render_measurements.py --out results/measurements.md [--container-host vm]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
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
CONTAINER_HOST = "vm"  # hostname the build container recorded in the committed bench json files
LAPTOP_NOTES = "laptop_notes.md"
FLAGS = ("avx2", "fma", "avx512f", "avx512bw", "avx512_vnni", "amx_tile", "amx_int8", "amx_bf16")
DURATIONS = ("2", "3.3", "5", "7.15", "10", "15")
TOKENS = ("16", "32", "64")
BUCKET_ORDER = ("le5s", "le8s", "le15s")
LAPTOP_COMMANDS = [
    "uv run python tools/count_params.py --models models/ --heads heads/ --out results/params.md",
    "uv run python tools/bench_components.py --threads 1,2,4 --clips bench_clips/ --out results/bench_components_$(hostname).json",
    'uv run python tools/replay_bench.py --clips "proxy/*.wav" --n 20 --realtime --out results/replay_bench_$(hostname).json',
    "uv run python tools/bench_components.py --build-only --trim-silence --durations 6,9,12,15 --clips bench_clips/long",
    ('uv run python tools/replay_bench.py --clips "bench_clips/long/*.wav" --n 0 --repeat 3 --realtime --endpoint-silence-ms 500 '
     "--out results/replay_bench_$(hostname)_long.json"),
    "uv run python tools/render_measurements.py --out results/measurements.md",
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
    return ["## Host that rendered this file", "",
            f"- host `{socket.gethostname()}`, rendered {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}, commit `{commit_hash(root)}`",
            f"- CPU (lscpu): {cpu['model_name']}",
            f"- nproc: {nproc}",
            f"- flags of interest: {flags}",
            f"- python {platform.python_version()} ({platform.platform()}), onnxruntime {ort.__version__}",
            "- free -m:", "", "```", free_m(), "```", ""]


# -- tables ----------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clip_note(source: str) -> str:
    """one definition line on where the bench clips come from."""
    if "proxy" in source:
        return "- The proxy clips are Piper TTS of MELD dev text. Their latencies are real. Their emotion labels are not a result."
    if "bench_clips" in source:
        return "- The clips are cut and tiled from `models/sensevoice/test_wavs/en.wav` by `tools/bench_components.py`."
    if source.startswith("audio"):
        return ("- The clips are MELD utterances, converted to 16 kHz WAV by `tools/extract_audio.py`. They are cut at the end of the "
                "subtitle, so the detected end of speech can fall after the end of the clip. The buckets group finals by segment length, and the "
                "clip-attribution row shows where the segments and the clips differ.")
    return f"- The clips are the WAV files under `{source}`."


def counted(items: list, what: str) -> str:
    """a count with the items behind it, ex. "2 clips without a final (a.wav#0, b.wav#3)"."""
    if not items:
        return f"{what[0].upper()}{what[1:]}: none"
    return f"{len(items)} {what} ({', '.join(str(i) for i in items)})"


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
        out.append(f"| Total including Piper | | {ms_m(res['total_with_piper'])} | | |")
    return out + [""]


def bench_section(b: dict, path: Path) -> list[str]:
    m = b.get("machine", {})
    out = [f"### Component bench (`{path.name}`, written by `tools/bench_components.py`)", "",
           (f"Generated {b.get('generated_at')} on `{b.get('host')}` by `{b.get('command')}`. "
            f"Each cell is the p50 or p95 in ms of {b.get('runs')} timed runs after {b.get('warmup')} warm-up run. "
            f"The load average was {b.get('loadavg_start')} at the start and {b.get('loadavg_end')} at the end. "
            f"The bench took {b.get('wall_s')} s. "
            f"Clips: {', '.join(f'{c['label']} s ({c['source']})' for c in b.get('clips', []))} from `{b.get('source_wav')}`."), ""]
    th = sorted(b.get("by_threads", {}), key=int)
    labels = [c["label"] for c in b.get("clips", [])] or list(DURATIONS)

    def table(title: str, key: str, cols: list[str], sub: str = "p50", nd: int = 1) -> list[str]:
        rows = [f"#### {title} ({sub}, ms)", "", "| threads | " + " | ".join(f"{c} s" if key != "text_ms" else f"{c} tok" for c in cols) + " |",
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
    out += ["#### VAD, heads, rules, selector, session loads", "",
            "| threads | VAD per 512-sample frame (1 thread) | heads (p50 ms) | selector (p50 ms) | load: sensevoice / text / vad / heads / selector (s) |",
            "|---|---|---|---|---|"]
    for t in th:
        r = b["by_threads"][t]
        heads = ", ".join(f"{k} {fmt(v.get('p50'), 3)}" for k, v in r.get("heads_ms", {}).items())
        ld = r.get("load_s", {})
        out.append(f"| {t} | {fmt(r.get('vad_frame_ms', {}).get('p50'), 3)} / p95 {fmt(r.get('vad_frame_ms', {}).get('p95'), 3)} | {heads} | "
                   f"{fmt(r.get('selector_ms', {}).get('p50'), 3)} | {ld.get('sensevoice')} / {ld.get('text')} / {ld.get('vad')} / {ld.get('heads')} / {ld.get('selector')} |")
    out += ["", "Model files on disk: " + ", ".join(f"`{k}` {fmt(v / 2**20, 1)} MB" if v else f"`{k}` missing" for k, v in b.get("model_files", {}).items())
            + "." + ((" Heads: " + ", ".join(f"`{k}` {v / 1024:.0f} KB" for k, v in b["head_files"].items()) + ".") if b.get("head_files") else ""), ""]
    if m:
        cpu = m.get("cpu", {})
        flags = ", ".join(f"{f} {'yes' if v else 'no'}" for f, v in cpu.get("flags", {}).items())
        mem = m.get("mem_mb", {})
        out += [(f"Machine recorded by the bench: {cpu.get('model_name')}, nproc {cpu.get('nproc')}"
                 + (f", MemTotal {mem.get('MemTotal')} MB" if mem.get("MemTotal") else "")
                 + f", python {m.get('python')}, onnxruntime {m.get('onnxruntime')}, numpy {m.get('numpy')}."
                 + (f" Flags of interest: {flags}." if flags else "")), ""]
    return out


def pf(v) -> str:
    return "n/a" if v is None else ("pass" if v else "**FAIL**")


def replay_section(r: dict, path: Path) -> list[str]:
    c, res, comp, mem = r["config"], r["results"], r["compute"], r["memory"]
    sent = r.get("sent") or {}
    modes = Counter(u.get("fused_mode") for u in r.get("utterances", []) if u.get("fused_mode"))
    out = [f"### Replay bench (`{path.name}`, written by `tools/replay_bench.py`)", "",
           f"Generated {r.get('generated_at')} on `{r.get('host')}` by `{r.get('command')}`."
           + (f" The clip attribution was recomputed from the stored utterances and clip spans at {r['reattributed']['at']} by "
              f"`{r['reattributed']['command']}`. The latencies were not touched." if r.get("reattributed") else ""), "",
           (f"- Clips: {c['n_clips']} from `{c['clip_source']}` (duration p50 {fmt(c['clip_duration_s'].get('p50'), 2)} s, "
            f"max {fmt(c['clip_duration_s'].get('max'), 2)} s), streamed {'at real-time pace' if c['realtime'] else 'as fast as possible'} "
            f"on one stream with {c['tail_s']} s of silence after each clip. "
            + ("" if not c["warmup_clips"] else
               f"Before the measured run, the first {'clip was' if c['warmup_clips'] == 1 else str(c['warmup_clips']) + ' clips were'} "
               "streamed once on a separate stream as a warm-up.")
            + (f" Streaming the {sent['audio_s']} s of audio took {sent['wall_s']} s." if sent.get("audio_s") is not None else "")),
           (f"- Service: {c['threads']} SenseVoice threads, {c['text_threads']} RoBERTa threads, endpoint silence {c['endpoint_silence_ms']} ms, "
            f"provisionals at {c['provisional_hz']} Hz over a {c['provisional_window_s']} s window, "
            f"fusion mode `{c.get('fusion_mode_requested', c['fusion_mode'])}` resolved to `{c['fusion_mode']}`, "
            f"head files in `heads/`: {', '.join(c['heads']) or 'none'}."
            + (f" The finals report `fused.mode` {', '.join(f'`{k}` ({v})' for k, v in modes.items())}." if modes else "")),
           (f"- Models loaded in {r['load_s']} s. The load average was {r['loadavg_start']} at the start and {r['loadavg_end']} at the end."),
           (f"- Messages: {res['n_finals']} finals, {res['n_responses']} responses, {res['n_provisionals']} provisionals, "
            f"errors: {res['errors'] or 'none'}" + ("" if r.get("finished_cleanly", True) else ". The run did not drain cleanly") + "."), "",
           "#### Final state after the endpoint, by utterance length (targets of design 1.10, at 4 threads)", "",
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
    out += ["#### Other targets of design 1.10", "", "| target | measured | pass |", "|---|---|---|",
            f"| reply text within {rp['target_p95_ms']} ms of the final state (p95) | p50 {fmt(rp.get('p50'), 3)}, p95 {fmt(rp.get('p95'), 3)}, max {fmt(rp.get('max'), 3)} ms over {rp.get('n', 0)} replies | {pf(rp['pass'])} |",
            (f"| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 {fmt(cad['full_window_interval_s'].get('p50'), 3)} s (n {cad['full_window_interval_s'].get('n', 0)}), "
            f"tail-window gap p50 {fmt(cad['tail_window_interval_s'].get('p50'), 3)} s (n {cad['tail_window_interval_s'].get('n', 0)}), "
            f"full-to-tail switch gap p50 {fmt(cad.get('full_to_tail_switch_interval_s', {}).get('p50'), 3)} s, first provisional "
            f"{fmt(cad.get('first_provisional_offset_s', {}).get('p50'), 3)} s after speech start, {cad['n_full']} full + {cad['n_tail']} tail provisionals, "
            f"{cad['utterances_over_1p5s_with_a_provisional']} utterances over 1.5 s got one, dropped {cad['dropped_provisionals']}, provisional SenseVoice p50 {fmt(cad['sensevoice_ms'].get('p50'))} ms | {pf(cad['pass'])} |"),
            (f"| sustained compute below {comp['target_max']} s per second of audio | worker {fmt(comp['worker_compute_s_per_audio_s'], 3)} s per audio s "
            f"({fmt(comp['worker_compute_s_per_speech_s'], 3)} per speech s), whole process {fmt(comp['process_cpu_s_per_audio_s'], 3)} cpu s per audio s, "
            f"cpu % sampled mean {fmt(comp['process_cpu_percent_sampled'].get('mean'))} max {fmt(comp['process_cpu_percent_sampled'].get('max'))} | {pf(comp['pass'])} |"),
            f"| memory | rss {mem['rss_mb_after_load']} MB after load, peak {mem['peak_rss_mb']} MB during the run (service + client + monitor in one process) | - |",
            (f"| worker lag | max health queue_s {fmt(res.get('max_queue_s'), 3)} s, degraded finals {res['finals_degraded']}, empty transcripts {res['finals_empty_text']} | - |"),
            (f"| clip attribution ({res.get('clip_attribution', 'by t_media_end')}) | {counted(res['clips_without_final'], 'clips without a final')}. "
             f"{counted(res['clips_with_several_finals'], 'clips split into several finals')}. "
             f"{counted(res.get('finals_unattributed') or [], 'finals outside every clip span')}."
             + ("" if "finals_ending_after_clip" not in res else
                " No final ends after the end of its clip." if not res["finals_ending_after_clip"] else
                f" {res['finals_ending_after_clip']} of the {res['n_finals']} finals end after the end of their clip, by up to "
                f"{fmt(res.get('max_end_after_clip_s'), 3)} s.") + " | - |"), ""]
    out += ["Definitions:", "",
            "- Service-side latency runs from the arrival of the endpoint frame (`t_service_recv_ns`) to the emission of the final state.",
            "- Client-side latency is the monitor's arrival time minus the capture time of the endpoint frame. "
            "It is valid because client and monitor run on one host.",
            "- Endpoint to response runs from the arrival of the endpoint frame to the emission of the response.",
            "- The service, the replay client and the monitor run in one process, so RSS and CPU include all three.",
            (f"- A final belongs to the clip whose span holds its `t_media_end`, the time the speech stopped "
             f"(rule: `{res.get('clip_attribution', 't_media_end within the clip span')}`). "
             + ("A span runs from the start of a clip to the start of the next one, so it includes the silence streamed after the clip. "
                if "next clip" in res.get("clip_attribution", "") else "")
             + "The segment start is not used because it carries the VAD pre-roll."),
            clip_note(str(c.get("clip_source", ""))), ""]
    return out


def laptop_section() -> list[str]:
    out = [("Every cell below reads `[laptop: not yet measured]` until the benches run on the laptop. "
            "When their JSON files are in `results/`, this section shows them in place of the table. "
            "docs/runbook_laptop.md gives the steps. Nothing here is typed from memory."), "", "Commands:", "", "```"] + LAPTOP_COMMANDS + ["```", "",
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


def _by_host(paths: list[str], container_host: str) -> tuple[list[tuple[Path, dict | None]], list[tuple[Path, dict | None]]]:
    """(container files, other hosts' files); an unreadable file goes with the container."""
    mine, other = [], []
    for p in paths:
        d = load_json(Path(p))
        (other if d is not None and d.get("host", container_host) != container_host else mine).append((Path(p), d))
    return mine, other


def _sections(files: list[tuple[Path, dict | None]], fn) -> list[str]:
    out = []
    for p, d in files:
        out += fn(d, p) if d else [f"[`{p.name}` unreadable]", ""]
    return out


def render(root: Path, results: Path, container_host: str = CONTAINER_HOST) -> str:
    out = ["# Measurements", "",
           ("Rendered by `python tools/render_measurements.py --out results/measurements.md` from the JSON files under `results/`. "
            "The first block describes the machine that rendered this file, from `lscpu`, `nproc`, `free -m`, the running interpreter and `.git`. "
            "Each bench section names the host that ran it and the machine it recorded. "
            "The targets are the real-time definition of design 1.10 (docs/design.md). "
            "Every measured number was written by the script named in its section."), ""]
    out += header(root)
    benches, laptop_benches = _by_host(sorted(glob.glob(str(results / "bench_components_*.json"))), container_host)
    replays, laptop_replays = _by_host(sorted(glob.glob(str(results / "replay_bench_*.json"))), container_host)
    out += [f"## Container (host `{container_host}`)", ""]
    out += params_section(load_json(results / "params.json"))
    if benches:
        out += _sections(benches, bench_section)
    else:
        out += ["### Component bench", "", "[not yet measured: run `python tools/bench_components.py --threads 1,2,4 --clips bench_clips/`]", ""]
    if replays:
        out += _sections(replays, replay_section)
    else:
        out += ["### Replay bench", "", "[not yet measured: run `python tools/replay_bench.py --clips \"proxy/*.wav\" --n 20 --realtime`]", ""]
    out += ["## Laptop (user-run)", ""]
    if laptop_benches or laptop_replays:
        out += _sections(laptop_benches, bench_section) + _sections(laptop_replays, replay_section)
    else:
        out += laptop_section()
    notes = results / LAPTOP_NOTES
    if notes.exists():
        out += [f"### Notes from the laptop (`results/{LAPTOP_NOTES}`, typed by the user)", "", notes.read_text(encoding="utf-8").strip(), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="render results/measurements.md from the bench json files")
    ap.add_argument("--out", type=Path, default=ROOT / "results/measurements.md")
    ap.add_argument("--results", type=Path, default=None, help="directory of the json files (default: the --out directory)")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--container-host", default=CONTAINER_HOST,
                    help="host name the build container recorded in its bench json files; any other host is the laptop")
    a = ap.parse_args(argv)
    results = a.results or a.out.parent
    md = render(a.root, results, a.container_host)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    print(f"wrote {a.out} ({len(md.splitlines())} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
