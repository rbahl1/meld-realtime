#!/usr/bin/env python
"""replay bench (design 1.10, 1.12): the real service in-process, a monitor
subscriber in the same process, and --n clips streamed back to back on one
stream at real-time pace (--realtime) with --tail-s of silence after each, so
the endpoint fires the way it does live. measures, from the messages the
service publishes: endpoint-to-final (timing_ms.service_total, arrival of the
endpoint frame to the final's emission) and endpoint-to-response p50/p95 per
utterance-length bucket (<= 5 s, 5-8 s, over 8 s up to the 15 s cap), the reply
delay after the final, the provisional cadence and dropped provisionals, the
worker compute per second of audio, and peak rss and cpu % of the process
through psutil. the real-time targets of design 1.10 are checked per bucket.

proxy clips are piper tts of meld dev text: their latencies are real, their
emotion labels are not a result (sensevoice labels affectless tts neutral or
unknown).

--video (extension spec 3.3, 6): the clips are mp4 files. each clip's audio
(ffmpeg, the options of tools/extract_audio.py) and its frames (--video-fps,
640 px wide, jpeg quality 80) go out on one media-time schedule, the service
runs with --vision (and --reply, the commit policy), and the result gets a
"vision" section: vision wait and aggregation per final, face coverage,
degraded counts, video frames processed / dropped / skipped, the vision
thread's time per frame, provisional text-expert time, commits, and the
streamed reply's first-delta time. the frames of the next clips are decoded by
a prefetch thread while a clip streams, so the client holds a few clips of
jpegs at a time; its decoding and encoding run in this process and count in
its cpu and rss. without --video the bench and its json are what they were.

usage: python tools/replay_bench.py --clips "proxy/*.wav" --n 20 --realtime --out results/replay_bench_$(hostname).json
       python tools/replay_bench.py --video --clips "/home/user/meld-data/mp4/test/*.mp4" --n 200 --realtime \
           --reply llm --out results/replay_bench_$(hostname)_tav.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import socket
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt import protocol as P  # noqa: E402
from meld_rt.clients import connect_push, connect_sub  # noqa: E402
from meld_rt.clients.replay import (  # noqa: E402
    VIDEO_FPS,
    decode_audio,
    decode_frames,
    load_wav,
    send_av,
    send_frames,
)
from meld_rt.service import Service, ServiceConfig  # noqa: E402

# design 1.10, at 4 threads on the build container; ms after the endpoint frame arrived
TARGETS = {
    "final_ms": {"le5s": {"p50": 250.0, "p95": 400.0}, "le8s": {"p50": 350.0, "p95": 550.0}, "le15s": {"p50": 700.0, "p95": 1000.0}},
    "reply_after_final_ms": {"p95": 5.0},
    "provisional_full_interval_s": {"target": 1.0, "tolerance": 0.1},
    "worker_compute_s_per_audio_s": {"max": 0.3},
}
BUCKETS = (("le5s", 0.0, 5.0, "up to 5 s (83 % of MELD)"), ("le8s", 5.0, 8.0, "5 to 8 s (95 % cumulative)"),
           ("le15s", 8.0, float("inf"), "8 s to the 15 s cap (99.7 % cumulative)"))
STAGES = ("frontend", "sensevoice", "prosody", "text_encoder", "heads")
# a final belongs to the clip whose span holds the time its speech stopped (t_media_end). a clip's
# span runs from its start to the next clip's start, so it takes in the silence streamed after it:
# only that clip can have produced speech there. meld clips are cut at the end of the subtitle and
# their speech often runs to the last frame, so the vad's hangover puts t_media_end past the clip
# end (up to 0.576 s on the meld bench). the segment start is never used, because the 256 ms
# pre-roll puts it inside the previous clip's silent tail.
ATTRIBUTION_RULE = "t_media_end within [media_start, next clip's media_start)"
ATTRIBUTION_NOTE = ("a final is attributed to the clip whose span holds its t_media_end (the time the speech stopped). "
                    "a span runs from the clip's start to the next clip's start, so it includes the silence after the clip. "
                    "the segment start carries the vad pre-roll and is not used")
STREAM = "bench"
WARMUP_STREAM = "warmup"


def pct(xs, q: float) -> float | None:
    return round(float(np.percentile(np.asarray(xs, dtype=np.float64), q)), 2) if len(xs) else None


def summary(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "max": round(float(max(xs)), 2), "mean": round(float(np.mean(xs)), 2)}


def bucket_of(duration_s: float) -> str:
    for name, lo, hi, _ in BUCKETS:
        if lo < duration_s <= hi or (lo == 0.0 and duration_s <= hi):
            return name
    return BUCKETS[-1][0]


def collect_clips(patterns: list[str], n: int) -> list[Path]:
    """first n files, sorted, of the comma-separated globs or paths."""
    files: list[Path] = []
    for pat in patterns:
        for item in pat.split(","):
            item = item.strip()
            if not item:
                continue
            hits = sorted(glob.glob(item)) or ([item] if Path(item).exists() else [])
            files += [Path(h) for h in hits]
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out[:n] if n > 0 else out


class Collector(threading.Thread):
    """subscriber that keeps every message with its arrival time (the monitor role)."""

    def __init__(self, addr: str):
        super().__init__(name="collector", daemon=True)
        self.sub = connect_sub(addr)
        self.msgs: list[tuple[int, str, dict]] = []
        self.last_non_health = time.monotonic()
        self.n_health = 0
        self._halt = threading.Event()

    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        while not self._halt.is_set():
            if poller.poll(50):
                topic, m = P.parse_out(self.sub.recv_multipart())
                self.msgs.append((P.now_ns(), topic, m))
                if topic == "health":
                    self.n_health += 1
                else:
                    self.last_non_health = time.monotonic()

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)
        self.sub.close()


class Sampler(threading.Thread):
    """rss every 100 ms and process cpu % every 500 ms."""

    def __init__(self):
        super().__init__(name="sampler", daemon=True)
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.rss: list[int] = []
        self.cpu: list[float] = []
        self._halt = threading.Event()

    def run(self) -> None:
        i = 0
        while not self._halt.is_set():
            self.rss.append(self.proc.memory_info().rss)
            if i % 5 == 4:
                self.cpu.append(self.proc.cpu_percent(None))
            i += 1
            self._halt.wait(0.1)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2)


def wait_idle(col: Collector, svc: Service, idle_s: float, timeout_s: float) -> bool:
    """true once nothing but health has arrived for idle_s, the worker is idle and (--reply llm)
    no reply is waiting or streaming: a generation can be silent for longer than idle_s (prefill
    under load, the first-token timeout), and its deltas, response and cpu belong to the run."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if time.monotonic() - col.last_non_health >= idle_s and svc.queue_s() == 0.0 and svc.reply_idle():
            return True
        time.sleep(0.05)
    return False


def stream_clips(push, clips: list[dict], stream_id: str, realtime: bool, tail_s: float) -> dict:
    """ctl start, every clip followed by tail_s of silence, ctl stop; one media clock."""
    frames: list[np.ndarray] = []
    tail = P.pcm16_frames(np.zeros(int(tail_s * P.SR), np.int16))
    for c in clips:
        f = P.pcm16_frames(c["pcm"])
        c["media_start"] = round(len(frames) * P.FRAME_S, 3)
        c["media_end"] = round((len(frames) + len(f)) * P.FRAME_S, 3)
        frames += f + tail
    push.send_multipart(P.ctl_msg(stream_id, "start"))
    t0 = time.monotonic()
    seq = send_frames(push, frames, stream_id, seq0=0, realtime=realtime, t0=t0)
    wall = time.monotonic() - t0
    push.send_multipart(P.ctl_msg(stream_id, "stop"))
    return {"frames": seq, "audio_s": round(seq * P.FRAME_S, 3), "wall_s": round(wall, 3)}


def clip_of(t_media_end: float, clips: list[dict]) -> int | None:
    """index of the clip whose span holds the time the speech stopped: the last clip that
    starts at or before it (see ATTRIBUTION_RULE). None before the first clip."""
    idx = None
    for i, c in enumerate(clips):
        if c["media_start"] <= t_media_end:
            idx = i
        else:
            break
    return idx


def attribution(rows: list[dict], clips: list[dict]) -> dict:
    """per streamed instance (a name repeats under --repeat), reported as name#index. exact by
    construction: the spans tile the stream, so a clip without a final produced none and a clip
    with several was split (a forced endpoint at the 15 s cap, or a pause longer than the
    endpoint silence)"""
    hits = [r["clip_index"] for r in rows if r["clip_index"] is not None]
    past = [round(end_of(r) - clips[r["clip_index"]]["media_end"], 3) for r in rows
            if r["clip_index"] is not None and end_of(r) > clips[r["clip_index"]]["media_end"]]
    return {"clip_attribution": ATTRIBUTION_RULE,
            "clips_without_final": [f"{c['name']}#{i}" for i, c in enumerate(clips) if i not in hits],
            "clips_with_several_finals": [f"{clips[i]['name']}#{i}" for i in sorted(set(hits)) if hits.count(i) > 1],
            "finals_unattributed": [r["utt_id"] for r in rows if r["clip_index"] is None],
            "finals_ending_after_clip": len(past), "max_end_after_clip_s": max(past, default=None)}


def end_of(row: dict) -> float:
    """t_media_end of an utterance row. rows written before it was stored have segment_s,
    which is t_media_end - t_media_start rounded to 1 ms."""
    return row["t_media_end"] if "t_media_end" in row else round(row["t_media_start"] + row["segment_s"], 3)


def analyze(msgs: list[tuple[int, str, dict]], clips: list[dict], stream_id: str) -> dict:
    finals = [(t, m) for t, topic, m in msgs if topic == "state.final" and m.get("stream_id") == stream_id]
    provs = [m for _, topic, m in msgs if topic == "state.provisional" and m.get("stream_id") == stream_id]
    resps = {m["utt_id"]: m for _, topic, m in msgs if topic == "response" and m.get("stream_id") == stream_id}
    healths = [m for _, topic, m in msgs if topic == "health" and m.get("stream_id") == stream_id]
    errors = [m for _, topic, m in msgs if topic == "error" and m.get("stream_id") == stream_id]

    rows = []
    for t_recv, f in finals:
        tm = f["timing_ms"]
        r = resps.get(f["utt_id"])
        dur = round(f["t_media_end"] - f["t_media_start"], 3)
        ps = sorted((p for p in provs if p["utt_id"] == f["utt_id"]), key=lambda p: p["t_media_end"])
        ci = clip_of(f["t_media_end"], clips)
        rows.append({
            "utt_id": f["utt_id"], "clip": clips[ci]["name"] if ci is not None else None, "clip_index": ci,
            "clip_duration_s": clips[ci]["duration_s"] if ci is not None else None,
            "t_media_start": f["t_media_start"], "t_media_end": f["t_media_end"], "segment_s": dur, "bucket": bucket_of(dur),
            "service_total_ms": tm.get("service_total"),
            "client_ms": round((t_recv - int(f["t_speech_end_capture_ns"])) / 1e6, 2) if f.get("t_speech_end_capture_ns") else None,
            "response_ms": round((int(r["t_emitted_ns"]) - int(f["t_service_recv_ns"])) / 1e6, 2) if r else None,
            "reply_after_final_ms": round((int(r["t_emitted_ns"]) - int(f["t_emitted_ns"])) / 1e6, 3) if r else None,
            "stages_ms": {k: tm.get(k, 0.0) for k in STAGES}, "vad_tail_ms": tm.get("vad_tail"), "provisional_in_flight_ms": tm.get("provisional_in_flight", 0.0),
            "n_provisionals": len(ps), "n_full": sum(p["window"] == "full" for p in ps), "n_tail": sum(p["window"] == "tail" for p in ps),
            "prov_media_ends": [p["t_media_end"] for p in ps], "prov_windows": [p["window"] for p in ps],
            "prov_sensevoice_ms": [p["timing_ms"].get("sensevoice") for p in ps],
            "asr_text": f["asr"]["text"], "asr_conf": f["asr"]["conf"], "degraded": f["degraded"],
            "fused_label": f["fused"]["label"], "fused_mode": f["fused"]["mode"], "continued_from": f.get("continued_from"),
        })

    by_bucket = {}
    for name, lo, hi, desc in BUCKETS:
        rs = [r for r in rows if r["bucket"] == name]
        b = {"description": desc, "n": len(rs),
             "service_total_ms": summary([r["service_total_ms"] for r in rs]),
             "client_ms": summary([r["client_ms"] for r in rs]),
             "response_ms": summary([r["response_ms"] for r in rs]),
             "reply_after_final_ms": summary([r["reply_after_final_ms"] for r in rs]),
             "stages_p50_ms": {k: pct([r["stages_ms"][k] for r in rs], 50) for k in STAGES} if rs else {},
             "provisional_in_flight_p50_ms": pct([r["provisional_in_flight_ms"] for r in rs], 50) if rs else None,
             "segment_s": summary([r["segment_s"] for r in rs])}
        tgt = TARGETS["final_ms"][name]
        st = b["service_total_ms"]
        b["target_ms"] = tgt
        b["pass"] = (st["p50"] <= tgt["p50"] and st["p95"] <= tgt["p95"]) if rs else None
        by_bucket[name] = b

    # provisional cadence: media-time gaps between consecutive provisionals of one utterance, by window
    full_gaps, tail_gaps, switch_gaps, first_offsets = [], [], [], []
    for r in rows:
        ends, wins = r["prov_media_ends"], r["prov_windows"]
        if ends:
            first_offsets.append(round(ends[0] - r["t_media_start"], 3))
        for a, b_, wa, wb in zip(ends, ends[1:], wins, wins[1:]):
            gap = round(b_ - a, 3)
            if wa == wb == "full":
                full_gaps.append(gap)
            elif wa == wb == "tail":
                tail_gaps.append(gap)
            else:
                switch_gaps.append(gap)
    long_utts = [r for r in rows if r["segment_s"] >= 1.5]
    cad = {"n_provisionals": len(provs), "n_full": sum(p["window"] == "full" for p in provs), "n_tail": sum(p["window"] == "tail" for p in provs),
           "full_window_interval_s": summary(full_gaps), "tail_window_interval_s": summary(tail_gaps),
           "full_to_tail_switch_interval_s": summary(switch_gaps),
           "per_utterance": summary([r["n_provisionals"] for r in rows]),
           "first_provisional_offset_s": summary(first_offsets),
           "utterances_over_1p5s_with_a_provisional": f"{sum(r['n_provisionals'] > 0 for r in long_utts)}/{len(long_utts)}",
           "sensevoice_ms": summary([p["timing_ms"].get("sensevoice") for p in provs]),
           "stability": summary([p.get("stability") for p in provs]),
           "dropped_provisionals": healths[-1]["dropped_provisionals"] if healths else None}
    tol = TARGETS["provisional_full_interval_s"]
    cad["pass"] = (abs(cad["full_window_interval_s"]["p50"] - tol["target"]) <= tol["tolerance"]) if full_gaps else None
    cad["rule"] = f"median media-time gap between consecutive full-window provisionals within {tol['target']} +- {tol['tolerance']} s"

    reply = summary([r["reply_after_final_ms"] for r in rows])
    reply["target_p95_ms"] = TARGETS["reply_after_final_ms"]["p95"]
    reply["pass"] = (reply["p95"] <= reply["target_p95_ms"]) if reply["n"] else None
    worker_ms = sum(sum(r["stages_ms"].values()) for r in rows) + sum(
        sum(p["timing_ms"].get(k, 0.0) or 0.0 for k in STAGES) for p in provs)
    codes: dict[str, int] = {}
    for e in errors:
        codes[e["code"]] = codes.get(e["code"], 0) + 1
    return {"n_finals": len(rows), "n_responses": len(resps), "n_provisionals": len(provs), "n_health": len(healths),
            "errors": codes, "max_queue_s": max((h["queue_s"] for h in healths), default=None),
            "health_rss_mb_last": healths[-1]["rss_mb"] if healths else None,
            **attribution(rows, clips),
            "finals_degraded": sum(bool(r["degraded"]) for r in rows), "finals_empty_text": sum(not r["asr_text"].strip() for r in rows),
            "all": {"service_total_ms": summary([r["service_total_ms"] for r in rows]), "client_ms": summary([r["client_ms"] for r in rows]),
                    "response_ms": summary([r["response_ms"] for r in rows]), "segment_s": summary([r["segment_s"] for r in rows]),
                    "stages_p50_ms": {k: pct([r["stages_ms"][k] for r in rows], 50) for k in STAGES} if rows else {}},
            "by_bucket": by_bucket, "reply_after_final_ms": reply, "provisionals": cad,
            "worker_compute_s": round(worker_ms / 1000, 3), "utterances": rows}


# -- video (--video) ---------------------------------------------------------------

class Prefetcher(threading.Thread):
    """decodes the frames of the clips in order and jpeg-encodes them, at most `ahead`
    clips before the one streaming; get(i) blocks until clip i is ready."""

    def __init__(self, clips: list[dict], fps: float, ahead: int = 2, quality: int = P.JPEG_QUALITY):
        super().__init__(name="prefetch", daemon=True)
        self.clips, self.fps, self.quality = clips, float(fps), int(quality)
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(ahead)))
        self.decode_s: list[float] = []
        self._halt = threading.Event()

    def run(self) -> None:
        for i, c in enumerate(self.clips):
            if self._halt.is_set():
                return
            t = time.perf_counter()
            try:
                frames = decode_frames(Path(c["path"]), self.fps)
                jpegs = [P.encode_jpeg(f, self.quality) for f in frames]
                w, h = (int(frames.shape[2]), int(frames.shape[1])) if len(frames) else (0, 0)
                item = {"jpegs": jpegs, "w": w, "h": h, "error": None}
            except Exception as e:  # noqa: BLE001 - the clip goes out without video
                item = {"jpegs": [], "w": 0, "h": 0, "error": f"{type(e).__name__}: {e}"}
            self.decode_s.append(time.perf_counter() - t)
            while not self._halt.is_set():
                try:
                    self._q.put((i, item), timeout=0.2)
                    break
                except queue.Full:
                    continue

    def get(self, i: int, timeout: float = 120.0) -> dict:
        j, item = self._q.get(timeout=timeout)
        if j != i:
            raise RuntimeError(f"prefetch order: wanted clip {i}, got {j}")
        return item

    def stop(self) -> None:
        self._halt.set()


def stream_clips_av(push, clips: list[dict], stream_id: str, realtime: bool, tail_s: float, fps: float,
                    ahead: int = 2) -> dict:
    """ctl start, every clip's audio and video on one media clock (each clip followed by
    tail_s of silence without video), ctl stop. the media span of each clip is fixed up
    front from its audio, as stream_clips does."""
    tail = P.pcm16_frames(np.zeros(int(tail_s * P.SR), np.int16))
    n = 0
    for c in clips:
        c["_frames"] = P.pcm16_frames(c["pcm"])
        c["media_start"] = round(n * P.FRAME_S, 3)
        c["media_end"] = round((n + len(c["_frames"])) * P.FRAME_S, 3)
        n += len(c["_frames"]) + len(tail)
    pre = Prefetcher(clips, fps, ahead)
    pre.start()
    first = pre.get(0) if clips else None    # the clock starts once the first clip's frames are ready
    push.send_multipart(P.ctl_msg(stream_id, "start"))
    t0 = time.monotonic()
    seq = vseq = late = 0
    errors = {}
    try:
        for i, c in enumerate(clips):
            v = first if i == 0 else pre.get(i)
            if v["error"]:
                errors[c["name"]] = v["error"]
            c["n_video_frames"] = len(v["jpegs"])
            start = seq * P.FRAME_S
            if realtime and time.monotonic() > t0 + start + 0.05:
                late += 1
            seq, vseq = send_av(push, c.pop("_frames") + tail, v["jpegs"], v["w"], v["h"], fps, stream_id, realtime,
                                seq0=seq, vseq0=vseq, t0_media=start, t0=t0 + start)
    finally:
        pre.stop()
    wall = time.monotonic() - t0
    push.send_multipart(P.ctl_msg(stream_id, "stop"))
    return {"frames": seq, "video_frames": vseq, "audio_s": round(seq * P.FRAME_S, 3), "wall_s": round(wall, 3),
            "clips_started_late": late, "video_decode_errors": errors,
            "video_decode_s": summary(pre.decode_s)}


def analyze_vision(msgs: list[tuple[int, str, dict]], stream_id: str) -> dict:
    """the vision-mode numbers of a --video run (see the module docstring)."""
    finals = [m for _, t, m in msgs if t == "state.final" and m.get("stream_id") == stream_id]
    provs = [m for _, t, m in msgs if t == "state.provisional" and m.get("stream_id") == stream_id]
    commits = [m for _, t, m in msgs if t == "state.commit" and m.get("stream_id") == stream_id]
    deltas = [m for _, t, m in msgs if t == "response.delta" and m.get("stream_id") == stream_id]
    resps = [m for _, t, m in msgs if t == "response" and m.get("stream_id") == stream_id]
    healths = [m for _, t, m in msgs if t == "health" and m.get("stream_id") == stream_id]
    vfin = [f for f in finals if f.get("vision")]
    degraded: dict[str, int] = {}
    for f in finals:
        for d in f.get("degraded", []):
            degraded[d] = degraded.get(d, 0) + 1
    present = [f for f in vfin if f["vision"]["present"]]
    full = [p for p in provs if p.get("window") == "full"]
    cblocks = [f["commit"] for f in finals if f.get("commit")]
    early = [c for c in cblocks if c.get("source") == "policy"]
    by_utt = {f["utt_id"]: f for f in finals}
    gain = [round(by_utt[c["utt_id"]]["t_media_end"] - by_utt[c["utt_id"]]["t_media_start"]
                  + by_utt[c["utt_id"]]["timing_ms"].get("vad_tail", 0.0) / 1000 - c["t_commit_s"], 3)
            for c in commits if c["utt_id"] in by_utt]
    fallbacks: dict[str, int] = {}
    for r in resps:
        if r.get("fallback"):
            k = str(r["fallback"]).split(":")[0]
            fallbacks[k] = fallbacks.get(k, 0) + 1
    h = healths[-1] if healths else {}
    return {
        "n_finals": len(finals), "finals_with_vision_block": len(vfin),
        "face_present_finals": len(present), "face_present_rate": round(len(present) / len(vfin), 4) if vfin else None,
        "presence_when_present": summary([f["vision"]["presence"] for f in present]),
        "n_tracks_when_present": summary([f["vision"]["n_tracks"] for f in present]),
        "vision_wait_ms": summary([f["timing_ms"].get("vision_wait") for f in vfin]),
        "vision_aggregate_ms": summary([f["timing_ms"].get("vision_aggregate") for f in vfin]),
        "degraded": degraded,
        "fused_modes": {m: sum(f["fused"]["mode"] == m for f in finals) for m in sorted({f["fused"]["mode"] for f in finals})},
        "face_flag_finals": sum(bool(f["discrepancy"].get("face_flag")) for f in finals),
        "provisionals_full_stacked_tav": f"{sum(p['fused']['mode'] == 'stacked_tav' for p in full)}/{len(full)}",
        "provisional_text_encoder_ms": summary([p["timing_ms"].get("text_encoder") for p in full if p["timing_ms"].get("text_encoder")]),
        "provisional_vision_aggregate_ms": summary([p["timing_ms"].get("vision_aggregate") for p in provs]),
        "commits": {"state_commit_messages": len(commits), "finals_with_commit_block": len(cblocks),
                    "committed_before_endpoint": len(early),
                    "rate_before_endpoint": round(len(early) / len(cblocks), 4) if cblocks else None,
                    "matches_final_rate": round(sum(c["matches_final"] for c in cblocks) / len(cblocks), 4) if cblocks else None,
                    "t_commit_s": summary([c["t_commit_s"] for c in commits]),
                    "seconds_before_endpoint_commit": summary(gain)},
        "reply": {"n_responses": len(resps), "n_deltas": len(deltas),
                  "sources": {s: sum(r.get("source", "bank") == s for r in resps) for s in ("llm", "bank")},
                  "fallbacks": fallbacks,
                  "first_delta_ms": summary([r.get("first_delta_ms") for r in resps]),
                  "gen_ms": summary([r.get("gen_ms") for r in resps]),
                  "n_tokens": summary([r.get("n_tokens") for r in resps if r.get("source") == "llm"])},
        "video": {k: h.get(k) for k in ("video_frames", "dropped_video_frames", "skipped_video_frames", "bad_video_frames",
                                        "vision_lag_finals", "vision_frame_ms_mean", "vision_error")},
    }


def reattribute(path: Path, command: str) -> dict:
    """recompute the clip attribution of a written bench file under ATTRIBUTION_RULE, from the
    utterances and clip spans it stores, and rewrite it. the latencies are not touched."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = d["config"]["clips"]
    res = d["results"]
    previous = res.get("clip_attribution")
    for u in d["utterances"]:
        ci = clip_of(end_of(u), clips)
        u.update({"clip": clips[ci]["name"] if ci is not None else None, "clip_index": ci,
                  "clip_duration_s": clips[ci]["duration_s"] if ci is not None else None})
    att = attribution(d["utterances"], clips)
    res.update(att)
    d["notes"] = [ATTRIBUTION_NOTE if n.startswith("a final is attributed to the clip") else n for n in d.get("notes", [])]
    if previous != ATTRIBUTION_RULE:
        d["reattributed"] = {"command": command, "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "previous_rule": previous}
    Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")
    return att


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="stream clips through the real service and measure the real-time contract")
    ap.add_argument("--clips", nargs="+", help="globs or files, comma- or space-separated")
    ap.add_argument("--reattribute", nargs="+", type=Path, metavar="JSON",
                    help="recompute the clip attribution of written bench files from their stored utterances and clip spans, then exit")
    ap.add_argument("--n", type=int, default=200, help="first n clips in sorted order (0: all)")
    ap.add_argument("--repeat", type=int, default=1, help="stream the clip list this many times")
    ap.add_argument("--realtime", action="store_true", help="pace frames at 32 ms; without it frames go as fast as possible")
    ap.add_argument("--out", type=Path, default=ROOT / f"results/replay_bench_{socket.gethostname()}.json")
    ap.add_argument("--threads", type=int, default=4, help="sensevoice intra-op threads")
    ap.add_argument("--text-threads", type=int, default=2)
    ap.add_argument("--endpoint-silence-ms", type=int, default=ServiceConfig().endpoint_silence_ms, help="default: the service default")
    ap.add_argument("--provisional-hz", type=float, default=1.0)
    ap.add_argument("--provisional-window-s", type=float, default=4.0)
    ap.add_argument("--provisional-budget-ms", type=float, default=0.0)
    ap.add_argument("--fusion-mode", default="auto")
    ap.add_argument("--tail-s", type=float, default=1.0, help="silence after each clip so the endpoint fires")
    ap.add_argument("--warmup", type=int, default=1, help="clips streamed first on a throwaway stream (not measured)")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads")
    ap.add_argument("--jsonl", type=Path, default=None, help="also write every message as json lines")
    g = ap.add_argument_group("--video: mp4 clips through the vision path (extension spec 3.3)")
    g.add_argument("--video", action="store_true", help="the clips are mp4 files: audio and frames, service with --vision")
    g.add_argument("--video-fps", type=float, default=VIDEO_FPS, help="frames per second sent")
    g.add_argument("--video-fps-max", type=float, default=ServiceConfig().video_fps_max)
    g.add_argument("--vision-lag-ms", type=float, default=ServiceConfig().vision_lag_ms)
    g.add_argument("--reply", choices=("bank", "llm"), default="bank")
    g.add_argument("--llm-dir", type=Path, default=None)
    g.add_argument("--llm-threads", type=int, default=ServiceConfig().llm_threads)
    g.add_argument("--commit-policy", type=Path, default=None)
    g.add_argument("--no-commit", action="store_true")
    g.add_argument("--prefetch", type=int, default=2, help="clips decoded ahead of the one streaming")
    a = ap.parse_args(argv)
    if a.reattribute:
        cmd = "python tools/replay_bench.py " + " ".join(sys.argv[1:] if argv is None else argv)
        for path in a.reattribute:
            r = reattribute(path, cmd)
            print(f"{path}: {len(r['clips_without_final'])} clips without a final, {len(r['clips_with_several_finals'])} with several, "
                  f"{len(r['finals_unattributed'])} finals outside every span")
        return 0
    if not a.clips:
        ap.error("--clips is required")

    files = collect_clips(a.clips, a.n)
    if not files:
        ap.error(f"no clips match {a.clips}")
    clips = [{"name": f.name, "path": str(f), "pcm": decode_audio(f) if a.video else load_wav(f)} for f in files] * max(1, a.repeat)
    clips = [dict(c) for c in clips]
    for c in clips:
        c["duration_s"] = round(len(c["pcm"]) / P.SR, 3)

    def log(s: str) -> None:
        print(s, file=sys.stderr, flush=True)

    log(f"{len(clips)} clips, {sum(c['duration_s'] for c in clips):.1f} s of speech, tail {a.tail_s} s each, realtime={a.realtime}")

    ext = {}
    if a.video:
        ext = {"vision": True, "video_fps_max": a.video_fps_max, "vision_lag_ms": a.vision_lag_ms, "reply": a.reply,
               "llm_dir": a.llm_dir, "llm_threads": a.llm_threads, "commit_policy": a.commit_policy, "commit": not a.no_commit}
    cfg = ServiceConfig(models=a.models, heads=a.heads, in_addr="tcp://127.0.0.1:*", out_addr="tcp://127.0.0.1:*",
                        threads=a.threads, text_threads=a.text_threads, endpoint_silence_ms=a.endpoint_silence_ms,
                        provisional_hz=a.provisional_hz, provisional_window_s=a.provisional_window_s,
                        provisional_budget_ms=a.provisional_budget_ms, fusion_mode=a.fusion_mode, preload=True, **ext)

    def stream(clip_list: list[dict], sid: str) -> dict:
        if a.video:
            return stream_clips_av(push, clip_list, sid, a.realtime, a.tail_s, a.video_fps, a.prefetch)
        return stream_clips(push, clip_list, sid, a.realtime, a.tail_s)
    svc = Service(cfg)
    rss_before = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    svc.start()
    if not svc.wait_ready(180):
        log("service did not load its models")
        svc.stop()
        return 2
    load_s = round(time.perf_counter() - t0, 2)
    rss_loaded = psutil.Process().memory_info().rss
    col = Collector(svc.out_addr)
    col.start()
    push = connect_push(svc.in_addr)
    deadline = time.monotonic() + 10
    while col.n_health == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    if col.n_health == 0:
        log("no health message; the subscriber never joined")
        svc.stop()
        return 2

    try:
        if a.warmup > 0:
            w = [dict(c) for c in clips[:a.warmup]]
            stream(w, WARMUP_STREAM)
            wait_idle(col, svc, 2.0, 60)
            n_before = len(col.msgs)
        else:
            n_before = 0
        sampler = Sampler()
        sampler.start()
        proc = psutil.Process()
        cpu0 = proc.cpu_times()
        load0 = [round(x, 2) for x in os.getloadavg()]
        t_stream0 = time.monotonic()
        sent = stream(clips, STREAM)
        finished = wait_idle(col, svc, 2.0, 120)
        wall_total = time.monotonic() - t_stream0
        cpu1 = proc.cpu_times()
        load1 = [round(x, 2) for x in os.getloadavg()]
        sampler.stop()
    finally:
        push.close()
        col.stop()
        svc.stop()

    msgs = col.msgs[n_before:]
    if a.jsonl:
        a.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(a.jsonl, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps({"topic": topic, "t_recv_ns": t, **m}, ensure_ascii=False) + "\n" for t, topic, m in msgs)
    res = analyze(msgs, clips, STREAM)
    if a.video:
        res["vision"] = analyze_vision(msgs, STREAM)
    cpu_s = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
    audio_s = sent["audio_s"]
    speech_s = round(sum(c["duration_s"] for c in clips), 3)
    comp = {"worker_compute_s": res["worker_compute_s"], "audio_s_streamed": audio_s, "speech_s": speech_s,
            "worker_compute_s_per_audio_s": round(res["worker_compute_s"] / audio_s, 4) if audio_s else None,
            "worker_compute_s_per_speech_s": round(res["worker_compute_s"] / speech_s, 4) if speech_s else None,
            "process_cpu_s": round(cpu_s, 3), "process_cpu_s_per_audio_s": round(cpu_s / audio_s, 4) if audio_s else None,
            "process_cpu_percent_sampled": summary(sampler.cpu), "wall_s": round(wall_total, 3),
            "target_max": TARGETS["worker_compute_s_per_audio_s"]["max"],
            "rule": "sum of the per-stage worker timings of every final and provisional, divided by the seconds of audio streamed (speech plus tails)"}
    comp["pass"] = (comp["worker_compute_s_per_audio_s"] <= comp["target_max"]) if comp["worker_compute_s_per_audio_s"] is not None else None
    out = {"host": socket.gethostname(), "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python tools/replay_bench.py " + " ".join(sys.argv[1:] if argv is None else argv),
           "config": {"threads": a.threads, "text_threads": a.text_threads, "endpoint_silence_ms": a.endpoint_silence_ms,
                      "provisional_hz": a.provisional_hz, "provisional_window_s": a.provisional_window_s,
                      "provisional_budget_ms": a.provisional_budget_ms, "fusion_mode_requested": a.fusion_mode, "fusion_mode": svc.mode,
                      "heads": svc.model_versions.get("heads", []), "model_versions": svc.model_versions,
                      "realtime": a.realtime, "tail_s": a.tail_s, "warmup_clips": a.warmup, "n_clips": len(clips), "repeat": a.repeat,
                      "clip_source": str(Path(files[0]).parent), "clips": [{"name": c["name"], "duration_s": c["duration_s"],
                                                                            "media_start": c["media_start"], "media_end": c["media_end"],
                                                                            **({"n_video_frames": c.get("n_video_frames")} if a.video else {})}
                                                                           for c in clips],
                      "clip_duration_s": summary([c["duration_s"] for c in clips]),
                      **({"video": True, "video_fps": a.video_fps, "video_fps_max": a.video_fps_max,
                          "vision_lag_ms": a.vision_lag_ms, "reply": a.reply, "generator": svc.generator_status(),
                          "commit_policy": svc.commit_policy_name, "vision_rule": svc.vision_rule,
                          "vision_heads": [k for k in ("vision_cal", "fusion3", "sentiment3") if svc.heads.get(k) is not None]}
                         if a.video else {})},
           "load_s": load_s, "sent": sent, "finished_cleanly": finished, "loadavg_start": load0, "loadavg_end": load1,
           "memory": {"rss_mb_before_load": round(rss_before / 2**20, 1), "rss_mb_after_load": round(rss_loaded / 2**20, 1),
                      "peak_rss_mb": round(max(sampler.rss) / 2**20, 1) if sampler.rss else None, "health_rss_mb_last": res["health_rss_mb_last"]},
           "compute": comp, "targets": TARGETS,
           "notes": (["video: the client decodes the mp4 frames and jpeg-encodes them in this process, so cpu and rss include that work",
                      ("video: worker_compute_s keeps its rule (the worker's stage timings); the vision thread's time is "
                       "vision.video.vision_frame_ms_mean per processed frame")] if a.video else []) +
                    ["service, replay client and monitor run in one process on this host; rss and cpu include all three",
                     "service_total_ms is the service-side latency: endpoint frame arrival (t_service_recv_ns) to the final's emission",
                     "client_ms is the monitor's arrival time minus the capture time of the endpoint frame, valid because both run here",
                     "response_ms is from the endpoint frame arrival to the response's emission",
                     "proxy clips are piper tts: latencies are real, emotion labels are not a result",
                     ATTRIBUTION_NOTE],
           "results": {k: v for k, v in res.items() if k != "utterances"}, "utterances": res["utterances"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")

    r = res
    print(f"replay bench on {out['host']}: {len(clips)} clips, {r['n_finals']} finals, {r['n_provisionals']} provisionals "
          f"({r['provisionals']['n_full']} full, {r['provisionals']['n_tail']} tail), dropped {r['provisionals']['dropped_provisionals']}, "
          f"errors {r['errors'] or 'none'}, threads {a.threads}, mode {svc.mode}, loadavg {load0} -> {load1}")
    print("bucket | n | final p50 / p95 ms (target) | response p50 / p95 ms | client p50 ms | pass")
    for name, b in r["by_bucket"].items():
        if b["n"]:
            s, q, c = b["service_total_ms"], b["response_ms"], b["client_ms"]
            print(f"{name} | {b['n']} | {s['p50']} / {s['p95']} ({b['target_ms']['p50']:.0f} / {b['target_ms']['p95']:.0f}) | "
                  f"{q['p50']} / {q['p95']} | {c['p50']} | {'pass' if b['pass'] else 'FAIL'}")
        else:
            print(f"{name} | 0 | no utterances in this bucket | | | n/a")
    rp = r["reply_after_final_ms"]
    print(f"reply after final: p50 {rp.get('p50')} p95 {rp.get('p95')} max {rp.get('max')} ms (target p95 <= {rp['target_p95_ms']}): "
          f"{'pass' if rp['pass'] else 'FAIL' if rp['pass'] is not None else 'n/a'}")
    cad = r["provisionals"]
    print(f"provisional cadence: full-window gap p50 {cad['full_window_interval_s'].get('p50')} s, tail gap p50 "
          f"{cad['tail_window_interval_s'].get('p50')} s, per utterance p50 {cad['per_utterance'].get('p50')}, "
          f"{cad['utterances_over_1p5s_with_a_provisional']} utterances over 1.5 s got one: {'pass' if cad['pass'] else 'FAIL' if cad['pass'] is not None else 'n/a'}")
    if a.video:
        v = r["vision"]
        print(f"vision: face in {v['face_present_finals']}/{v['finals_with_vision_block']} finals, vision wait p50 "
              f"{v['vision_wait_ms'].get('p50')} p95 {v['vision_wait_ms'].get('p95')} ms, aggregate p50 {v['vision_aggregate_ms'].get('p50')} ms, "
              f"degraded {v['degraded'] or 'none'}, video {v['video']}")
        print(f"commit: {v['commits']['committed_before_endpoint']}/{v['commits']['finals_with_commit_block']} before the endpoint; "
              f"reply: {v['reply']['sources']} first delta p50 {v['reply']['first_delta_ms'].get('p50')} "
              f"p95 {v['reply']['first_delta_ms'].get('p95')} ms, fallbacks {v['reply']['fallbacks'] or 'none'}")
    print(f"compute: worker {comp['worker_compute_s_per_audio_s']} s per audio s (target <= {comp['target_max']}): "
          f"{'pass' if comp['pass'] else 'FAIL'}; process cpu {comp['process_cpu_s_per_audio_s']} s per audio s, "
          f"cpu% sampled mean {comp['process_cpu_percent_sampled'].get('mean')} max {comp['process_cpu_percent_sampled'].get('max')}; "
          f"peak rss {out['memory']['peak_rss_mb']} MB; load {load_s} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
