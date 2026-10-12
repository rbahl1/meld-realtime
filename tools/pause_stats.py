#!/usr/bin/env python
"""internal pauses and endpoint splits (design 1.3, decisions "end-pointing").

runs the service's silero vad (meld_rt.experts.vad) over 16 khz wavs, fed the
way the replay client feeds the service: 512-sample pcm16 frames, the last one
zero-padded, then --tail-s of silence.

- pauses: one pass per clip with a segmenter that never closes, so the rnn
  state is never reset. the segment opens where the start rule fires (8 frames
  above 0.5). after that, a pause is a run of frames with p below the end
  threshold (vad.SILENCE_P, 0.35) that a frame at or above it ends, i.e. a run
  the endpoint counter sees and then resets. the silence after the last speech
  is not a pause.
- splits: for each candidate --endpoint-silence-ms the clip goes through a
  segmenter built like the service's (ServiceConfig values). a clip is split
  when it opens two or more segments. "split by a pause" counts only segments
  that open after a silence endpoint, not after the 15 s cap.
- added latency: the tail, i.e. the setting rounded up to whole 32 ms frames.
  with --bench, the tail plus the service-side p50 / p95 of the <= 5 s bucket
  of a replay bench file, which is measured from the endpoint frame and so
  leaves the tail out.

writes results/pause_stats_<label>.json and prints a markdown table.

usage:
  uv run python tools/pause_stats.py --wavs "models/sensevoice/test_wavs/*.wav" "proxy/*.wav" --label available --bench results/replay_bench_vm.json
  uv run python tools/pause_stats.py --wavs "audio/dev/*.wav" --label meld_dev --bench results/replay_bench_vm.json
"""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt import protocol as P  # noqa: E402
from meld_rt.clients.replay import load_wav  # noqa: E402
from meld_rt.experts.vad import FRAME_S, SILENCE_P, VadSegmenter  # noqa: E402
from meld_rt.service import ServiceConfig  # noqa: E402

SETTINGS_MS = (300, 400, 500, 600, 800)
FRAME_MS = FRAME_S * 1000


def collect_wavs(patterns: list[str]) -> dict[str, list[Path]]:
    """sorted files of each comma- or space-separated glob or path, keyed by the pattern; a file
    matched by two patterns stays with the first."""
    groups: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for pat in patterns:
        for item in pat.split(","):
            item = item.strip()
            if not item:
                continue
            hits = sorted(glob.glob(item)) or ([item] if Path(item).exists() else [])
            new = [Path(h) for h in hits if Path(h) not in seen]
            seen.update(new)
            if new:
                groups.setdefault(item, []).extend(new)
    return groups


def clip_frames(pcm16: np.ndarray, tail_s: float) -> list[np.ndarray]:
    """the frames the replay client sends for a clip: the clip, then tail_s of silence."""
    return P.pcm16_frames(pcm16) + P.pcm16_frames(np.zeros(int(tail_s * P.SR), np.int16))


def service_segmenter(model: Path, endpoint_silence_ms: int) -> VadSegmenter:
    c = ServiceConfig()
    return VadSegmenter(model, endpoint_silence_ms=endpoint_silence_ms, min_speech_ms=c.min_speech_ms,
                        preroll_ms=c.preroll_ms, max_segment_s=c.max_segment_s, threads=1)


def open_segmenter(model: Path) -> VadSegmenter:
    """a segmenter that never closes, so the probabilities come from one unbroken rnn pass."""
    c = ServiceConfig()
    return VadSegmenter(model, endpoint_silence_ms=10**9, min_speech_ms=c.min_speech_ms,
                        preroll_ms=c.preroll_ms, max_segment_s=10**6, threads=1)


def frame_probs(vad: VadSegmenter, frames: list[np.ndarray]) -> np.ndarray:
    vad.reset(keep_clock=False)
    out = np.empty(len(frames), np.float32)
    for i, f in enumerate(frames):
        vad.push(f)
        out[i] = vad.last_prob
    return out


def extract_pauses(probs, start_frames: int, threshold: float = 0.5, silence_p: float = SILENCE_P) -> list[tuple[int, int]]:
    """(first frame, length in frames) of every internal pause. the segment opens on the frame where
    start_frames consecutive frames above threshold end (VadSegmenter.push); from the next frame on,
    a pause is a maximal run with p < silence_p that a frame with p >= silence_p ends. a run still
    open at the last frame is the trailing silence and is left out."""
    p = np.asarray(probs, dtype=np.float64)
    run, opened = 0, None
    for i, x in enumerate(p):
        run = run + 1 if x > threshold else 0
        if run >= start_frames:
            opened = i
            break
    if opened is None:
        return []
    pauses, start = [], None
    for i in range(opened + 1, len(p)):
        if p[i] < silence_p:
            if start is None:
                start = i
        elif start is not None:
            pauses.append((start, i - start))
            start = None
    return pauses


def count_segments(vad: VadSegmenter, frames: list[np.ndarray]) -> dict:
    """segments opened, forced endpoints, and segments opened after a silence endpoint."""
    vad.reset(keep_clock=False)
    segments = forced = after_pause = 0
    last_forced = None
    for f in frames:
        for ev in vad.push(f):
            if ev.kind == "speech_start":
                segments += 1
                after_pause += last_forced is False
            else:
                forced += ev.forced
                last_forced = ev.forced
    return {"segments": segments, "forced": forced, "after_pause": after_pause}


def pct(xs, q: float) -> float | None:
    return round(float(np.percentile(xs, q)), 2) if len(xs) else None


def dist(xs) -> dict:
    """numpy linear-interpolation percentiles."""
    return {"n": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "p95": pct(xs, 95),
            "max": round(float(max(xs)), 2) if len(xs) else None}


def bench_le5s(path: Path) -> dict:
    """service-side latency of the <= 5 s bucket of a replay bench file."""
    d = json.loads(path.read_text(encoding="utf-8"))
    b = d["results"]["by_bucket"]["le5s"]["service_total_ms"]
    if not b.get("n"):
        raise SystemExit(f"{path} has no utterances up to 5 s; pass the proxy-clip bench")
    return {"file": rel(path), "endpoint_silence_ms": d["config"]["endpoint_silence_ms"], "bucket": "le5s",
            "n": b["n"], "service_total_p50_ms": b["p50"], "service_total_p95_ms": b["p95"]}


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def summarize(clips: list[dict], segs: dict[int, VadSegmenter], lat: dict | None) -> dict:
    """pause distribution and split counts per setting over the clips in which the vad opened a segment."""
    speech = [c for c in clips if c["by_setting"][str(next(iter(segs)))]["segments"] > 0]
    n = len(speech)
    pauses_all = [x for c in speech for x in c["pauses_ms"]]
    clip_max = [max(c["pauses_ms"], default=0.0) for c in speech]
    rows = []
    for ms, v in segs.items():
        tail = round(v.silence_frames * FRAME_MS, 1)
        split = sum(c["by_setting"][str(ms)]["segments"] >= 2 for c in speech)
        by_pause = sum(c["by_setting"][str(ms)]["after_pause"] > 0 for c in speech)
        row = {"endpoint_silence_ms": ms, "tail_frames": v.silence_frames, "added_latency_ms": tail,
               "clips_split": split, "pct_split": round(100 * split / n, 1) if n else None,
               "clips_split_by_pause": by_pause, "pct_split_by_pause": round(100 * by_pause / n, 1) if n else None,
               "clips_with_pause_ge_tail": sum(m >= tail for m in clip_max),
               "pauses_ge_tail": sum(x >= tail for x in pauses_all),
               "forced_endpoints": sum(c["by_setting"][str(ms)]["forced"] for c in speech)}
        if lat:
            row["wait_p50_ms"] = round(tail + lat["service_total_p50_ms"], 2)
            row["wait_p95_ms"] = round(tail + lat["service_total_p95_ms"], 2)
        rows.append(row)
    return {"n_clips": len(clips), "n_with_speech": n, "audio_s": round(sum(c["duration_s"] for c in clips), 3),
            "pauses_ms": dist(pauses_all), "clip_max_pause_ms": dist(clip_max), "settings": rows}


def analyze(files: list[Path] | dict[str, list[Path]], model: Path, settings=SETTINGS_MS, tail_s: float = 1.0,
            bench: Path | None = None) -> dict:
    """files, or files keyed by group (the --wavs pattern); the top level pools every clip."""
    groups = files if isinstance(files, dict) else {"all": list(files)}
    probe = open_segmenter(model)
    segs = {ms: service_segmenter(model, ms) for ms in settings}
    clips = []
    for group, paths in groups.items():
        for f in paths:
            pcm = load_wav(f)
            frames = clip_frames(pcm, tail_s)
            pauses = extract_pauses(frame_probs(probe, frames), probe.start_frames, probe.threshold)
            row = {"file": rel(f), "group": group, "duration_s": round(len(pcm) / P.SR, 3),
                   "pauses_ms": [round(n * FRAME_MS, 1) for _, n in pauses],
                   "pause_starts_s": [round(s * FRAME_S, 3) for s, _ in pauses]}
            row["by_setting"] = {str(ms): count_segments(v, frames) for ms, v in segs.items()}
            clips.append(row)
    lat = bench_le5s(bench) if bench else None
    return {
        **summarize(clips, segs, lat),
        "groups": {g: summarize([c for c in clips if c["group"] == g], segs, lat) for g in groups} if len(groups) > 1 else {},
        "vad": {"model": rel(model), "start_frames": probe.start_frames, "start_threshold": probe.threshold,
                "end_threshold": SILENCE_P, "frame_ms": FRAME_MS, "max_segment_s": ServiceConfig().max_segment_s,
                "tail_s_after_clip": tail_s},
        "service_latency": lat,
        "notes": ["a pause is a run of frames with p < end_threshold after the segment opened, ended by a frame with "
                  "p >= end_threshold, in whole 32 ms frames",
                  "percentiles use numpy linear interpolation, and clip_max_pause_ms counts a clip without a pause as 0",
                  "shares are over the clips in which the vad opened a segment",
                  "a clip is split when it opens two or more segments, and split by a pause when a segment opens after a "
                  "silence endpoint rather than the forced one",
                  "added latency is the tail: the setting rounded up to whole frames, before the endpoint frame arrives",
                  "wait = added latency + service-side latency of the bench's <= 5 s bucket (endpoint frame arrival to "
                  "final emission), from the end of speech to the final state",
                  "groups repeat the summary for the clips of each --wavs pattern"],
        "clips": clips,
    }


def markdown(res: dict, label: str) -> str:
    p, m = res["pauses_ms"], res["clip_max_pause_ms"]
    lines = [f"pause stats `{label}`: {res['n_clips']} clips, {res['n_with_speech']} with speech, {p['n']} internal pauses",
             "",
             f"internal pause ms: p50 {p['p50']}, p90 {p['p90']}, p95 {p['p95']}, max {p['max']}",
             f"longest pause per clip ms: p50 {m['p50']}, p90 {m['p90']}, p95 {m['p95']}, max {m['max']}",
             ""]
    wait = res["service_latency"] is not None
    head = "| endpoint ms | tail frames | added latency ms | clips split | % split | split by a pause | % by a pause |"
    head += " wait p50 / p95 ms |" if wait else ""
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for r in res["settings"]:
        line = (f"| {r['endpoint_silence_ms']} | {r['tail_frames']} | {r['added_latency_ms']} | {r['clips_split']} | "
                f"{r['pct_split']} | {r['clips_split_by_pause']} | {r['pct_split_by_pause']} |")
        line += f" {r['wait_p50_ms']} / {r['wait_p95_ms']} |" if wait else ""
        lines.append(line)
    if wait:
        s = res["service_latency"]
        lines += ["", f"wait = added latency + service-side p50 / p95 of `{s['file']}` (<= 5 s bucket, n {s['n']}, "
                      f"run at {s['endpoint_silence_ms']} ms)"]
    if res.get("groups"):
        lines += ["", "| group | clips with speech | pauses | pause p50 / p95 / max ms | "
                      + " / ".join(str(r["endpoint_silence_ms"]) for r in res["settings"]) + " ms: clips split |",
                  "|---|---|---|---|---|"]
        for g, r in res["groups"].items():
            p = r["pauses_ms"]
            lines.append(f"| `{g}` | {r['n_with_speech']} | {p['n']} | {p['p50']} / {p['p95']} / {p['max']} | "
                         + " / ".join(str(x["clips_split"]) for x in r["settings"]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="internal pauses and endpoint splits under the service's silero vad")
    ap.add_argument("--wavs", nargs="+", required=True, help="globs or files, comma- or space-separated")
    ap.add_argument("--label", required=True, help="writes results/pause_stats_<label>.json")
    ap.add_argument("--settings", default=",".join(map(str, SETTINGS_MS)), help="endpoint silence settings in ms")
    ap.add_argument("--tail-s", type=float, default=1.0, help="silence after each clip, as the replay client sends")
    ap.add_argument("--bench", type=Path, default=None, help="replay bench json for the service-side latency")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    files = collect_wavs(a.wavs)
    if not files:
        ap.error(f"no wavs match {a.wavs}")
    settings = tuple(int(s) for s in a.settings.split(",") if s.strip())
    res = analyze(files, a.models / ServiceConfig().vad, settings, a.tail_s, a.bench)
    out = {"host": socket.gethostname(), "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python tools/pause_stats.py " + shlex.join(sys.argv[1:] if argv is None else argv),
           "label": a.label, "wavs": a.wavs, **res}
    path = a.out or ROOT / f"results/pause_stats_{a.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(markdown(res, a.label))
    print(f"\nwrote {rel(path)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
