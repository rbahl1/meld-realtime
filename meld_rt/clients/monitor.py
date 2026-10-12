"""monitor: subscribe to every topic, print one line per message with its
timings, append jsonl. client-side latency is this process's now minus
t_speech_end_capture_ns, which is valid when the client ran on this host; the
service-side number is timing_ms.service_total. a robot body would subscribe
the same way.

usage: python -m meld_rt.clients.monitor [--jsonl results/demo_log.jsonl] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import zmq

from meld_rt import protocol as P
from meld_rt.clients import connect_sub


def _top(d: dict) -> str:
    if not d:
        return "-"
    k = max(d, key=d.get)
    return f"{k}@{d[k]:.2f}"


def fmt_line(topic: str, m: dict, latency_ms: float | None) -> str:
    sid = m.get("stream_id", "")
    if topic in ("state.final", "state.provisional"):
        tm = m.get("timing_ms", {})
        head = "final " if topic == "state.final" else "prov  "
        parts = [f"{head} {sid} utt={m.get('utt_id')} {m.get('window')} media={m.get('t_media_start')}-{m.get('t_media_end')}"]
        if topic == "state.provisional":
            parts.append(f"partial={m.get('text', {}).get('partial')!r} stab={m.get('stability')}")
        else:
            parts.append(f"asr={m.get('asr', {}).get('text')!r} conf={m.get('asr', {}).get('conf')}")
        parts.append(f"audio={_top(m.get('audio', {}).get('emotion', {}))} text={_top(m.get('text', {}).get('emotion', {}))}")
        f = m.get("fused", {})
        parts.append(f"fused={f.get('label')}@{f.get('confidence')} {f.get('mode')} sent={f.get('sentiment')}")
        d = m.get("discrepancy", {})
        parts.append(f"jsd={d.get('jsd')} {d.get('direction')}")
        parts.append(f"service_total={tm.get('service_total')}ms" + (f" client={latency_ms:.1f}ms" if latency_ms is not None else ""))
        parts.append(f"[sv={tm.get('sensevoice')} fe={tm.get('frontend')} pros={tm.get('prosody')} txt={tm.get('text_encoder')} "
                     f"heads={tm.get('heads')} vad_tail={tm.get('vad_tail')} prov_in_flight={tm.get('provisional_in_flight')}]")
        if m.get("degraded"):
            parts.append(f"degraded={m['degraded']}")
        return " ".join(parts)
    if topic == "response":
        return (f"reply  {sid} utt={m.get('utt_id')} {m.get('text')!r} [{m.get('template_id')} {m.get('style')}] "
                f"grounding={m.get('grounding')}")
    if topic == "health":
        return (f"health {sid or '-'} queue={m.get('queue_s')}s dropped={m.get('dropped_provisionals')} mode={m.get('fusion_mode')} "
                f"rss={m.get('rss_mb')}MB loaded={m.get('models_loaded')}")
    if topic == "error":
        return f"error  {sid or '-'} {m.get('code')}: {m.get('msg')}"
    return f"{topic} {json.dumps(m)[:200]}"


class Monitor:
    """collects messages; prints and logs them; knows when to stop."""

    def __init__(self, addr: str = P.OUT_ADDR, jsonl: Path | None = None, quiet: bool = False,
                 topics: tuple[str, ...] = (), out=None):
        self.sub = connect_sub(addr, topics)
        self.jsonl = open(jsonl, "a", encoding="utf-8") if jsonl else None  # noqa: SIM115 - closed in close()
        self.quiet = quiet
        self.out = out or sys.stdout
        self.counts: dict[str, int] = {}
        self.last_final_emit_ns: dict[tuple[str, int], int] = {}

    def close(self) -> None:
        self.sub.close()
        if self.jsonl:
            self.jsonl.close()

    def handle(self, topic: str, m: dict) -> None:
        t_recv = P.now_ns()
        self.counts[topic] = self.counts.get(topic, 0) + 1
        latency = None
        if topic == "state.final" and m.get("t_speech_end_capture_ns"):
            latency = (t_recv - int(m["t_speech_end_capture_ns"])) / 1e6
            self.last_final_emit_ns[(m.get("stream_id"), m.get("utt_id"))] = int(m.get("t_emitted_ns", 0))
        extra = {}
        if topic == "response":
            t_final = self.last_final_emit_ns.get((m.get("stream_id"), m.get("utt_id")))
            if t_final:
                extra["after_final_ms"] = round((int(m.get("t_emitted_ns", 0)) - t_final) / 1e6, 3)
        if self.jsonl:
            row = {"topic": topic, "t_recv_ns": t_recv, **({"latency_client_ms": round(latency, 3)} if latency is not None else {}),
                   **extra, **m}
            self.jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.jsonl.flush()
        if not self.quiet or topic == "error":
            line = fmt_line(topic, m, latency)
            if extra:
                line += f" after_final={extra['after_final_ms']}ms"
            print(line, file=self.out, flush=True)

    def run(self, exit_after_final: int = 0, idle_s: float = 0.0, max_s: float = 0.0) -> dict:
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        t_start = last = time.monotonic()
        while True:
            now = time.monotonic()
            if max_s and now - t_start >= max_s:
                break
            if idle_s and now - last >= idle_s:
                break
            if exit_after_final and self.counts.get("response", 0) >= exit_after_final:
                break
            if poller.poll(100):
                topic, m = P.parse_out(self.sub.recv_multipart())
                if topic != "health":
                    last = time.monotonic()
                self.handle(topic, m)
        return dict(self.counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="print and log every message from the meld_rt service")
    ap.add_argument("--addr", default=P.OUT_ADDR)
    ap.add_argument("--jsonl", type=Path, help="append every message as one json line")
    ap.add_argument("--quiet", action="store_true", help="print errors only (jsonl still written)")
    ap.add_argument("--topics", nargs="*", default=(), help="subscribe to these topics only")
    ap.add_argument("--exit-after-final", type=int, default=0, metavar="N", help="exit once N replies were seen")
    ap.add_argument("--idle-s", type=float, default=0.0, help="exit after this long without a non-health message")
    ap.add_argument("--max-s", type=float, default=0.0, help="exit after this many seconds")
    a = ap.parse_args(argv)
    if a.jsonl:
        a.jsonl.parent.mkdir(parents=True, exist_ok=True)
    mon = Monitor(a.addr, a.jsonl, a.quiet, tuple(a.topics))
    try:
        counts = mon.run(a.exit_after_final, a.idle_s, a.max_s)
    except KeyboardInterrupt:
        counts = dict(mon.counts)
    finally:
        mon.close()
    print("monitor counts " + json.dumps(counts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
