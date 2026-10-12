"""end to end: the service in-process on free ports, a real-time replay of
en.wav (and the first long enough proxy clip when proxy/ exists), a typed
text message, and a sequence gap. 2 intra-op threads."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zmq

from meld_rt import protocol as P
from meld_rt.clients import connect_push, connect_sub
from meld_rt.clients.replay import load_wav, replay, send_frames, send_text
from meld_rt.service import Service, ServiceConfig
from meld_rt.state import State

ROOT = Path(__file__).resolve().parents[1]
EN_WAV = ROOT / "models/sensevoice/test_wavs/en.wav"
MODELS = [ROOT / "models/silero_vad.onnx", ROOT / "models/sensevoice/model.int8.onnx", ROOT / "models/roberta-base-11.onnx"]
THREADS = 2
ENDPOINT_MS = 500          # the default. en.wav has a 448 ms pause at 3.392 s, 500 ms keeps it one utterance (results/pause_stats_available.json)
MAX_SERVICE_TOTAL_MS = 1500
REPLY_WITHIN_MS = 5.0

pytestmark = pytest.mark.skipif(not all(p.exists() for p in MODELS + [EN_WAV]), reason="models not fetched")


def proxy_clip() -> Path | None:
    """first proxy clip of at least 2.5 s, when tools/make_proxy_clips.py has run."""
    man = ROOT / "proxy/manifest.json"
    if not man.exists():
        return None
    for row in json.loads(man.read_text()):
        p = ROOT / "proxy" / row["file"]
        if row.get("duration_s", 0) >= 2.5 and p.exists():
            return p
    return None


CLIPS = [EN_WAV] + ([proxy_clip()] if proxy_clip() else [])


@pytest.fixture(scope="module")
def service():
    cfg = ServiceConfig(in_addr="tcp://127.0.0.1:*", out_addr="tcp://127.0.0.1:*", threads=THREADS,
                        text_threads=THREADS, preload=True, endpoint_silence_ms=ENDPOINT_MS)
    svc = Service(cfg)
    svc.start()
    assert svc.wait_ready(120), "service did not load its models"
    yield svc
    svc.stop()


class Client:
    def __init__(self, svc: Service):
        self.sub = connect_sub(svc.out_addr)
        self.push = connect_push(svc.in_addr)
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        self.msgs: list[tuple[str, dict]] = []
        # pub/sub slow joiner: the first health message proves the subscription is live
        assert self.collect(lambda ms: any(t == "health" for t, _ in ms), 10.0), "no health message"

    def close(self):
        self.sub.close()
        self.push.close()

    def collect(self, until, timeout_s: float, grace_s: float = 0.0) -> bool:
        deadline = time.monotonic() + timeout_s
        done_at = None
        while time.monotonic() < deadline:
            if self.poller.poll(50):
                self.msgs.append(P.parse_out(self.sub.recv_multipart()))
            if done_at is None and until(self.msgs):
                if grace_s <= 0:
                    return True
                done_at = time.monotonic() + grace_s
            if done_at is not None and time.monotonic() >= done_at:
                return True
        return until(self.msgs)

    def of(self, topic: str, stream_id: str) -> list[dict]:
        return [m for t, m in self.msgs if t == topic and m.get("stream_id") == stream_id]


@pytest.fixture
def client(service):
    c = Client(service)
    yield c
    c.close()


def has_response(stream_id: str):
    return lambda ms: any(t == "response" and m.get("stream_id") == stream_id for t, m in ms)


@pytest.mark.parametrize("wav", CLIPS, ids=[p.name for p in CLIPS])
def test_realtime_replay_gives_provisionals_one_final_and_one_reply(service, client, wav):
    sid = f"e2e-{wav.stem}"
    pcm = load_wav(wav)
    sender = threading.Thread(target=replay, args=(client.push, pcm, sid, True), kwargs={"tail_s": 1.0}, daemon=True)
    sender.start()
    assert client.collect(has_response(sid), timeout_s=len(pcm) / P.SR + 30, grace_s=0.5), "no reply arrived"
    sender.join(timeout=5)

    provisionals = client.of("state.provisional", sid)
    finals = client.of("state.final", sid)
    responses = client.of("response", sid)
    healths = client.of("health", sid)
    errors = client.of("error", sid)
    assert len(provisionals) >= 1, "no provisional state"
    assert len(finals) == 1, [f["asr"]["text"] for f in finals]
    assert len(responses) == 1
    assert healths, "no health message for the stream"
    assert not errors, errors

    f = finals[0]
    assert f["kind"] == "final" and f["asr"]["text"].strip()
    assert f["timing_ms"]["service_total"] < MAX_SERVICE_TOTAL_MS, f["timing_ms"]
    assert f["t_media_end"] > f["t_media_start"] >= 0 and f["seq_range"][1] > f["seq_range"][0] >= 0
    assert f["timing_ms"]["vad_tail"] == pytest.approx(ENDPOINT_MS, abs=P.FRAME_S * 1000)
    assert f["fused"]["mode"] in ("poe", "poe_uncalibrated", "stacked") and f["fused"]["label"] in f["fused"]["emotion"]
    assert abs(sum(f["fused"]["emotion"].values()) - 1) < 1e-3 and abs(sum(f["audio"]["emotion"].values()) - 1) < 1e-3
    assert f["t_service_recv_ns"] <= f["t_emitted_ns"] and f["t_speech_end_capture_ns"] > 0
    assert "asr_failed" not in f["degraded"] and "low_asr_conf" not in f["degraded"]
    State.from_dict(f)  # schema round trip

    r = responses[0]
    assert r["utt_id"] == f["utt_id"] and r["text"].strip() and r["grounding"] and r["template_id"] and r["style"]
    assert 0 <= (r["t_emitted_ns"] - f["t_emitted_ns"]) / 1e6 < REPLY_WITHIN_MS

    # uncalibrated fusion: the label is the product's argmax and the confidence the text head's probability of it,
    # and the selector never asserts (bucket capped at mid, so ask or hold_back on a fresh stream)
    fu = f["fused"]
    if fu["mode"] == "poe_uncalibrated":
        assert fu["calibrated"] is False
        assert fu["confidence"] == pytest.approx(f["text"]["emotion"][fu["label"]], abs=1e-4)
        assert fu["label"] == max(fu["emotion"], key=fu["emotion"].get)
    if not fu["calibrated"]:
        assert r["style"] in ("ask", "hold_back"), r
        assert (fu["confidence"] >= service.high_conf) == ("rule=uncalibrated_cap" in r["grounding"]), (fu, r, service.high_conf)

    for p in provisionals:
        assert p["kind"] == "provisional" and p["utt_id"] == f["utt_id"] and p["window"] in ("full", "tail")
        assert 0.0 <= p["stability"] <= 1.0 and p["t_media_end"] <= f["t_media_end"] + ENDPOINT_MS / 1000
        if p["window"] == "tail":
            assert p["text"]["partial"] is None
    # audio-path provisionals and the final are in media order
    ends = [p["t_media_end"] for p in provisionals]
    assert ends == sorted(ends)
    h = healths[-1]
    assert h["models_loaded"] and h["fusion_mode"] in ("poe", "stacked") and h["rss_mb"] > 0
    print(f"\n{wav.name}: service_total {f['timing_ms']['service_total']} ms, provisionals {len(provisionals)}, "
          f"timing {f['timing_ms']}, asr {f['asr']['text']!r}, fused {f['fused']['label']}@{f['fused']['confidence']} "
          f"{f['fused']['mode']}, reply {r['text']!r}")


def test_text_message_is_text_only(client):
    sid = "e2e-text"
    send_text(client.push, sid, "well that went great, the GR-6 system failed again")
    assert client.collect(has_response(sid), timeout_s=20, grace_s=0.2)
    finals = client.of("state.final", sid)
    assert len(finals) == 1
    f = finals[0]
    assert f["fused"]["mode"] == "text_only" and "no_audio" in f["degraded"]
    assert f["asr"]["text"] == "well that went great, the GR-6 system failed again"
    assert f["discrepancy"]["flag"] is False and f["discrepancy"]["direction"] == "none"
    assert f["text"]["keyphrase"] and f["text"]["keyphrase"] in f["asr"]["text"]
    assert f["timing_ms"]["sensevoice"] == 0 and f["timing_ms"]["service_total"] < MAX_SERVICE_TOTAL_MS
    r = client.of("response", sid)[0]
    assert r["grounding"] and r["utt_id"] == f["utt_id"]
    assert not client.of("state.provisional", sid)


def test_sequence_gap_marks_audio_gap(client):
    sid = "e2e-gap"
    frames = P.pcm16_frames(load_wav(EN_WAV))
    tail = P.pcm16_frames(np.zeros(P.SR, np.int16))
    cut = int(2.0 / P.FRAME_S)          # 2.0 s in: speech started at 1.09 s, so a segment is open
    skip = 40                           # 1.28 s of missing frames, more than the 500 ms rule

    def send():
        client.push.send_multipart(P.ctl_msg(sid, "start"))
        t0 = time.monotonic()
        seq = send_frames(client.push, frames[:cut], sid, seq0=0, realtime=True, t0=t0)
        send_frames(client.push, frames[cut:] + tail, sid, seq0=seq + skip, realtime=True, t0=time.monotonic())
        client.push.send_multipart(P.ctl_msg(sid, "stop"))

    sender = threading.Thread(target=send, daemon=True)
    sender.start()

    def gap_final(ms):
        return any(t == "state.final" and m.get("stream_id") == sid and "audio_gap" in m["degraded"] for t, m in ms)

    assert client.collect(gap_final, timeout_s=40, grace_s=1.0), [m for t, m in client.msgs if t != "health"]
    sender.join(timeout=10)
    finals = client.of("state.final", sid)
    assert any("audio_gap" in f["degraded"] for f in finals)
    assert any(e["code"] == "audio_gap" for e in client.of("error", sid))
    # the segment cut by the gap and the one after it are different utterances
    assert len({f["utt_id"] for f in finals}) == len(finals)
    assert all(f["seq_range"][1] >= f["seq_range"][0] for f in finals)
