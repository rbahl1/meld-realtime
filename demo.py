#!/usr/bin/env python
"""demo.sh and demo_tav.sh for a shell without bash (ex. windows powershell): the same
steps, environment variables, defaults and printed lines. the service and the monitor
run in the background, the replay client in the foreground, and the summary is the
python block of the bash script itself, cut out as tools/render_demo_docs.py cuts it.
the background processes are stopped on exit, as the bash trap does.

usage: uv run python demo.py [clip.wav]          (as ./demo.sh)
       uv run python demo.py --tav [clip.mp4]    (as ./demo_tav.sh)
env: as the scripts: THREADS, ENDPOINT_MS, N_FINALS, DEMO_LOG, and with --tav DEMO_MP4,
     VIDEO_FPS, REPLY, LLM_DIR
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))

from render_demo_docs import summary_code, utf8_flags  # noqa: E402

DEFAULT_WAV = "proxy/dev_0001.wav"
FALLBACK_WAV = "models/sensevoice/test_wavs/en.wav"


def env(name: str, default: str) -> str:
    """${NAME:-default}: the default when the variable is unset or empty."""
    return os.environ.get(name) or default


def python(*args: str) -> list[str]:
    """this interpreter (the project's environment under uv run) on one module or script."""
    return [sys.executable, *utf8_flags(), *args]


def plan(tav: bool, clip: str | None) -> dict:
    """the clip, the log and the three command lines of demo.sh (or demo_tav.sh), with their echo lines.
    raises SystemExit(2) where demo_tav.sh exits 2."""
    threads, endpoint, n_finals = env("THREADS", "4"), env("ENDPOINT_MS", "500"), env("N_FINALS", "1")
    if not tav:
        clip = clip or DEFAULT_WAV
        notes = []
        if not Path(clip).is_file():
            notes.append(f"no clip at {clip}; using {FALLBACK_WAV}")
            clip = FALLBACK_WAV
        log = env("DEMO_LOG", "results/demo_log.jsonl")
        return {"clip": clip, "log": log, "notes": notes, "steps": [
            (f"== service: threads={threads} endpoint={endpoint}ms",
             python("-m", "meld_rt.service", "--threads", threads, "--endpoint-silence-ms", endpoint, "--preload",
                    "--log-level", "warning")),
            (f"== monitor: state and response lines below, everything in {log}",
             python("-m", "meld_rt.clients.monitor", "--jsonl", log, "--topics", "state.provisional", "state.final",
                    "response", "error", "--exit-after-final", n_finals, "--idle-s", "20", "--max-s", "180")),
            (f"== replay: {clip} at real-time pace (waits for the models to load)",
             python("-m", "meld_rt.clients.replay", "--wav", clip, "--realtime", "--stream-id", "demo", "--wait-ready", "180")),
        ]}
    clip = clip or os.environ.get("DEMO_MP4") or ""
    if not clip or not Path(clip).is_file():
        print(f"no mp4 at '{clip}': give one as the first argument or set DEMO_MP4", file=sys.stderr)
        raise SystemExit(2)
    fps, reply = env("VIDEO_FPS", "10"), env("REPLY", "llm")
    log = env("DEMO_LOG", "results/demo_tav_log.jsonl")
    llm = ["--llm-dir", os.environ["LLM_DIR"]] if os.environ.get("LLM_DIR") else []
    return {"clip": clip, "log": log, "notes": [], "steps": [
        (f"== service: --vision --reply {reply} threads={threads} endpoint={endpoint}ms",
         python("-m", "meld_rt.service", "--vision", "--reply", reply, *llm, "--threads", threads,
                "--endpoint-silence-ms", endpoint, "--preload", "--log-level", "warning")),
        (f"== monitor: state, commit and reply lines below, everything in {log}",
         python("-m", "meld_rt.clients.monitor", "--jsonl", log, "--topics", "state.provisional", "state.final",
                "state.commit", "response.delta", "response", "error", "--exit-after-final", n_finals,
                "--idle-s", "30", "--max-s", "300")),
        (f"== replay: {clip} (audio and {fps} fps video) at real-time pace (waits for the models to load)",
         python("-m", "meld_rt.clients.replay", "--video", clip, "--video-fps", fps, "--realtime", "--stream-id", "demo",
                "--wait-ready", "300")),
    ]}


def stop(procs: list[subprocess.Popen]) -> None:
    """the bash trap: end the background processes and wait for them."""
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="demo.sh (or demo_tav.sh with --tav) without bash")
    ap.add_argument("clip", nargs="?", help="a wav (default proxy/dev_0001.wav), or an mp4 with --tav (default $DEMO_MP4)")
    ap.add_argument("--tav", action="store_true", help="the three-input demo of demo_tav.sh")
    a = ap.parse_args(argv)
    os.chdir(ROOT)   # cd "$(dirname "$0")": a relative clip path is taken from the repository root, as in bash
    try:
        p = plan(a.tav, a.clip)
    except SystemExit as e:
        return int(e.code or 0)
    for line in p["notes"]:
        print(line, flush=True)
    log = Path(p["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(b"")
    (svc_echo, svc_cmd), (mon_echo, mon_cmd), (rep_echo, rep_cmd) = p["steps"]
    procs: list[subprocess.Popen] = []
    try:
        print(svc_echo, flush=True)
        procs.append(subprocess.Popen(svc_cmd))
        print(mon_echo, flush=True)
        mon = subprocess.Popen(mon_cmd)
        procs.append(mon)
        print(rep_echo, flush=True)
        rc = subprocess.call(rep_cmd)
        if rc:
            return rc            # set -e
        mon.wait()               # wait "$MON" || true
        print("== summary", flush=True)
        return subprocess.call(python("-c", summary_code(ROOT, "demo_tav.sh" if a.tav else "demo.sh"), str(log)))
    except KeyboardInterrupt:
        return 130
    finally:
        stop(procs)


if __name__ == "__main__":
    sys.exit(main())
