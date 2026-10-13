"""vision mode, commit ticks and the streamed reply of meld_rt.service (spec 3.2,
3.4, 3.5). packages A (meld_rt.vision_features, the vision heads), C (the
generator) and D (meld_rt.commit) are replaced by small fakes, so these tests
do not depend on their files: a fake face expert that reads the frame colour,
a fake vision_features module, tiny npz heads, a commit policy that emits at
the second tick and a generator that streams five words. the end-to-end part
runs the real vad, sensevoice and roberta on en.wav at real-time pace with
synthetic frames; one test uses the real yunet / fer files when they exist,
and one the real vision_features module when it exists."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import zmq

from meld_rt import protocol as P
from meld_rt.clients import connect_push, connect_sub
from meld_rt.clients.replay import load_wav, replay, replay_video, send_text
from meld_rt.discrepancy import audio7, jsd
from meld_rt.experts.vad import VadEvent
from meld_rt.experts.vision import Face, posterior_meld
from meld_rt.heads import entropy_bits, head_params, lr_params, save_head
from meld_rt.labels import EMOTIONS, SENTIMENTS, SV_EMO_TOKENS, SV_EVENT_TOKENS
from meld_rt.service import (
    VIDEO_QUEUE_MAX,
    FaceFrame,
    Job,
    ReplyJob,
    Service,
    ServiceConfig,
    Stream,
    VideoItem,
    VideoQueue,
    VisionResult,
    face_discrepancy,
    face_frame,
    skip_media,
    vision_rule_from,
)
from meld_rt.state import FACE_DISCREPANCY_KEYS, VISION_TIMING_KEYS, State

ROOT = Path(__file__).resolve().parents[1]
EN_WAV = ROOT / "models/sensevoice/test_wavs/en.wav"
MODELS = [ROOT / "models/silero_vad.onnx", ROOT / "models/sensevoice/model.int8.onnx", ROOT / "models/roberta-base-11.onnx"]
YUNET = ROOT / "models/vision/face_detection_yunet_2023mar.onnx"
FER = ROOT / "models/vision/facial_expression_recognition_mobilefacenet_2022july.onnx"
CORE_HEADS = ("text_ctx.npz", "audio_cal.npz", "fusion.npz", "sentiment.npz")
THREADS = 2
UNIFORM = np.full(7, 1 / 7)
cv2 = pytest.importorskip("cv2")
have_audio_models = pytest.mark.skipif(not all(p.exists() for p in MODELS + [EN_WAV]), reason="models not fetched")


# -- fakes ---------------------------------------------------------------------

FER_HAPPY, FER_ANGRY = 3, 0      # zoo order: angry, disgust, fearful, happy, neutral, sad, surprised


class FakeExpert:
    """one face per frame, fer logits from the frame colour: blue frames look happy,
    red ones angry, black ones have no face."""

    def __init__(self):
        self.resets = 0
        self.n = 0

    def reset(self):
        self.resets += 1
        self.n = 0

    def process(self, frame):
        b, g, r = (float(frame[..., i].mean()) for i in range(3))
        if max(b, g, r) < 20:
            return []
        logits = np.zeros(7, np.float32)
        logits[FER_HAPPY if b >= r else FER_ANGRY] = 4.0
        self.n += 1
        mm = float("nan") if self.n == 1 else 0.05
        h, w = frame.shape[:2]
        box = np.array([w * 0.3, h * 0.2, w * 0.25, h * 0.4])
        return [Face(0, box, np.zeros((5, 2)), 0.9, logits, np.full(512, 0.01, np.float32), mm, 0.01)]


def make_fake_vf():
    """the interface of spec 2.1, with the largest-area rule only."""
    m = types.ModuleType("meld_rt.vision_features")
    m.calls = []

    @dataclass
    class FaceRows:
        t: np.ndarray
        track: np.ndarray
        area: np.ndarray
        score: np.ndarray
        logits: np.ndarray
        mouth: np.ndarray
        face: np.ndarray
        emb: np.ndarray
        frame_t: np.ndarray | None = None

    Ev = namedtuple("FaceEvidence", "present track p_face n_frames presence log_area n_tracks asd_corr score mouth emb")

    def frame_log_rms(frame):
        x = np.asarray(frame, np.float64) / 32768.0
        return float(np.log(max(float(np.sqrt(np.mean(x * x))), 1e-5)))

    def face_evidence(rows, env_t, env_v, rule, t0, t1):
        m.calls.append({"rule": rule, "t0": t0, "t1": t1, "n_rows": len(rows.t), "n_env": len(env_t),
                        "n_frames": None if rows.frame_t is None else len(rows.frame_t)})
        sel = (rows.t >= t0) & (rows.t < t1)
        if not sel.any():
            return Ev(False, None, UNIFORM.copy(), 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, np.zeros(512, np.float32))
        tracks = np.unique(rows.track[sel])
        tr = max(tracks, key=lambda k: rows.area[sel & (rows.track == k)].mean())
        k = sel & (rows.track == tr)
        n_win = len(np.unique(rows.frame_t)) if rows.frame_t is not None else len(np.unique(rows.t[sel]))
        p = posterior_meld(rows.logits[k]).mean(axis=0)
        return Ev(True, int(tr), p, int(k.sum()), float(k.sum() / max(n_win, k.sum())), float(np.log(rows.area[k].mean())),
                  len(tracks), 0.0, float(rows.score[k].mean()), 0.0, rows.emb[k].mean(axis=0))

    def vision_features_A(ev):
        lp = np.log(np.clip(ev.p_face, 1e-8, 1))
        tail = [float(np.max(ev.p_face)), entropy_bits(lp), ev.presence, ev.log_area, ev.n_tracks / 4, ev.asd_corr, ev.score,
                ev.mouth * 10, float(ev.present)]
        return np.concatenate([lp, tail]).astype(np.float32)

    def vision_features_B(ev):
        return np.concatenate([vision_features_A(ev), ev.emb]).astype(np.float32)

    def build_fusion3_features(lt, la, lf, asr_conf, p_laughter, p_unknown, face_present, presence):
        tail = [entropy_bits(lt), entropy_bits(la), entropy_bits(lf), asr_conf, p_laughter, p_unknown, float(face_present), presence]
        return np.concatenate([lt, la, lf, tail]).astype(np.float32)

    m.FaceRows, m.frame_log_rms, m.face_evidence = FaceRows, frame_log_rms, face_evidence
    m.vision_features_A, m.vision_features_B, m.build_fusion3_features = vision_features_A, vision_features_B, build_fusion3_features
    return m


class FakePolicy:
    """emits at the second tick."""
    name = "fake_policy"

    def __init__(self):
        self.ticks = []

    @classmethod
    def load(cls, path):
        return cls()

    def features(self, tick):
        self.ticks.append(tick)
        return np.array([tick["t_s"], tick["k"], float(np.max(tick["fused"])), tick["same_label_run"]])

    def decide(self, x):
        return bool(x[1] >= 2), 0.7, 0.3


GenResult = namedtuple("GenResult", "text n_tokens first_token_ms total_ms stop_reason masked_steps")
WORDS = ["You", " sound", " pleased", " about", " that."]


class FakeGenerator:
    first_word_delay_s = 0.0
    stall_after_first_s = 0.0           # > 0: sleeps this long after the first word (a stalled generation)
    fail = False

    def __init__(self, model_dir, threads=2, max_new_tokens=40):
        self.model_dir, self.threads = Path(model_dir), threads
        self.calls = 0
        self.deadlines = []

    def generate(self, messages, allowed, on_delta, deadline_s=None):
        self.calls += 1
        self.deadlines.append(deadline_s)
        if self.fail:
            raise RuntimeError("boom")
        time.sleep(self.first_word_delay_s)
        for i, w in enumerate(WORDS):
            on_delta(w)
            time.sleep(self.stall_after_first_s if i == 0 and self.stall_after_first_s else 0.005)
        return GenResult("".join(WORDS), 9, 12.0, 60.0, "eos", 0)


def make_fake_gen_module():
    m = types.ModuleType("meld_rt.response.generator")
    m.ReplyGenerator = FakeGenerator
    m.calls = []

    def build_messages(state, low_conf, high_conf=None):
        m.calls.append({"low_conf": low_conf, "high_conf": high_conf})
        f = state["fused"]
        msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": json.dumps(f["label"])}]
        return msgs, [f"fused.label={f['label']}@{f['confidence']:.2f}"], [f["label"]] if f["confidence"] >= low_conf else []

    def check_reply(text, allowed):
        return {"n_words": len(text.split()), "exclamation": "!" in text, "forbidden": [], "empty": not text.strip()}

    m.build_messages, m.check_reply = build_messages, check_reply
    return m


def make_fake_commit_module():
    m = types.ModuleType("meld_rt.commit")
    m.CommitPolicy = FakePolicy
    return m


def write_heads(d: Path, core: bool = True) -> Path:
    """core heads (symlinks to heads/) plus tiny vision heads: the calibrator sharpens the
    raw face posterior, fusion3 adds the three log-posteriors with the face counted twice."""
    d.mkdir(parents=True, exist_ok=True)
    if core:
        for f in CORE_HEADS:
            if (ROOT / "heads" / f).exists():
                (d / f).symlink_to(ROOT / "heads" / f)
    man = json.loads((ROOT / "heads/manifest.json").read_text()) if (ROOT / "heads/manifest.json").exists() else {}
    man["vision"] = {"rule": "largest", "thresholds": {"low_conf": 0.2, "high_conf": 0.9}, "jsd_face_flag": 0.25}
    (d / "manifest.json").write_text(json.dumps(man))
    eye = np.eye(7)
    w_cal = np.zeros((16, 7))
    w_cal[:7] = eye
    save_head(d / "vision_cal.npz", head_params(lr_params(w_cal.T, np.zeros(7)), np.zeros(16), np.ones(16), EMOTIONS, 0.5,
                                                 variant="A", rule="largest"))
    w3 = np.zeros((29, 7))
    w3[:7], w3[7:14], w3[14:21] = eye, eye, 2 * eye
    save_head(d / "fusion3.npz", head_params(lr_params(w3.T, np.zeros(7)), np.zeros(29), np.ones(29), EMOTIONS, 1.0))
    ws = np.zeros((29, 3))
    ws[14 + EMOTIONS.index("joy"), SENTIMENTS.index("positive")] = 1.0
    save_head(d / "sentiment3.npz", head_params(lr_params(ws.T, np.zeros(3)), np.zeros(29), np.ones(29), SENTIMENTS, 1.0))
    np.savez(d / "commit_policy.npz", c=np.array(0.05))
    return d


def jpeg(color, w=320, h=180) -> bytes:
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = color
    return P.encode_jpeg(f)


BLUE, RED, BLACK = (255, 40, 40), (40, 40, 255), (0, 0, 0)


def bare_service(tmp_path: Path, **kw) -> Service:
    """a vision-mode service that is never started: its methods run on the test thread."""
    cfg = ServiceConfig(heads=write_heads(tmp_path / "heads", core=False), vision=True, **kw)
    svc = Service(cfg)
    svc.vf = make_fake_vf()
    svc.new_vision_expert = FakeExpert
    svc.heads = svc._load_heads()
    svc.vision_rule = vision_rule_from(svc.manifest)
    return svc


def item(st: Stream, t: float, color=BLUE, seq: int = 0) -> VideoItem:
    with st.v_cv:
        st.v_pending += 1
    return VideoItem(st, seq, t, 0, 0, 320, 180, jpeg(color), st.v_epoch)


def run_frame(svc: Service, it: VideoItem) -> None:
    try:
        svc._vision_frame(it)
    finally:
        with it.st.v_cv:
            it.st.v_pending -= 1


def drain(svc: Service) -> list[tuple[str, dict]]:
    out = []
    while not svc._out.empty():
        out.append(svc._out.get_nowait())
    return out


# -- rules without models -----------------------------------------------------

def test_face_direction_uses_the_words_tone_rule():
    joy, anger = np.eye(7)[EMOTIONS.index("joy")] * 0.9 + 0.1 / 7, np.eye(7)[EMOTIONS.index("anger")] * 0.9 + 0.1 / 7
    d, flag, direction = face_discrepancy(joy, anger, 0.2)
    assert flag and direction == "face_more_negative" and d > 0.5
    assert face_discrepancy(anger, joy, 0.2)[2] == "face_more_positive"
    surprise = np.eye(7)[EMOTIONS.index("surprise")] * 0.9 + 0.1 / 7
    assert face_discrepancy(joy, surprise, 0.2)[2] == "face_surprised"
    assert face_discrepancy(joy, joy, 0.2) == (0.0, False, "none")


def test_vision_rule_from_manifest():
    assert vision_rule_from({"vision": {"rule": "asd"}}) == "asd"
    assert vision_rule_from({"vision": {"rule": {"chosen": "mouth"}}}) == "mouth"
    assert vision_rule_from({"vision": {"speaker_rule": "present"}}) == "present"
    assert vision_rule_from({"vision": {"rule": "nonsense"}}) == "largest"
    assert vision_rule_from({}) == "largest" and vision_rule_from(None) == "largest"


def test_video_queue_drops_the_oldest_frame():
    q = VideoQueue(VIDEO_QUEUE_MAX)
    st = Stream("s", vad=None)
    dropped = [q.put(VideoItem(st, i, i / 10, 0, 0, 1, 1, b"", 0)) for i in range(VIDEO_QUEUE_MAX + 3)]
    assert dropped[:VIDEO_QUEUE_MAX] == [None] * VIDEO_QUEUE_MAX
    assert [d.seq for d in dropped[VIDEO_QUEUE_MAX:]] == [0, 1, 2]
    assert [q.get(0.01).seq for _ in range(VIDEO_QUEUE_MAX)] == [3, 4, 5, 6] and q.get(0.01) is None


def test_ingest_never_decodes_and_counts_drops(tmp_path, monkeypatch):
    svc = bare_service(tmp_path)
    st = Stream("s", vad=None)
    payload = jpeg(BLUE)
    decodes = []
    real_imdecode = cv2.imdecode

    def counting_imdecode(*a, **kw):
        decodes.append(threading.current_thread().name)
        return real_imdecode(*a, **kw)

    monkeypatch.setattr(cv2, "imdecode", counting_imdecode)
    for k in range(10):
        svc._on_video(st, {"seq": k, "t_media": k / 10, "t_capture_ns": 0, "w": 320, "h": 180}, payload, 0)
    assert decodes == []
    assert len(svc._video_q) == VIDEO_QUEUE_MAX and st.dropped_video_frames == 10 - VIDEO_QUEUE_MAX
    assert st.v_pending == VIDEO_QUEUE_MAX
    # the vision thread's side decodes (the counter sees decoding when it happens)
    it = svc._video_q.get(0.1)
    svc._vision_frame(it)
    assert len(decodes) == 1 and st.video_frames == 1


def test_core_service_refuses_video_without_a_stream_or_a_model_load(tmp_path):
    """without --vision a video message gets the core's bad_message (once per stream id, not per
    frame), opens no stream and does not start loading the models."""
    svc = Service(ServiceConfig(heads=tmp_path))
    frame = np.zeros((48, 64, 3), np.uint8)
    for k in range(3):
        svc._handle(P.frame_msg("camonly", k, 1, k / 10, frame), P.now_ns())
    svc._handle(P.frame_msg("other", 0, 1, 0.0, frame), P.now_ns())
    assert not svc._streams and not svc._load_requested.is_set() and len(svc._video_q) == 0
    msgs = drain(svc)
    assert [(t, m["stream_id"], m["code"]) for t, m in msgs] == [("error", "camonly", "bad_message"),
                                                                  ("error", "other", "bad_message")]
    assert msgs[0][1]["msg"].startswith("unknown topic 'video'")


def test_frame_gap_epoch_and_ring(tmp_path):
    svc = bare_service(tmp_path, video_fps_max=10.0)
    st = Stream("s", vad=None)
    for t in (0.0, 0.05, 0.1, 0.18, 0.3):      # 0.05 is closer than 0.75 / 10 s to 0.0
        run_frame(svc, item(st, t))
    assert [f.t for f in st.v_frames] == [0.0, 0.1, 0.18, 0.3] and st.skipped_video_frames == 1
    assert st.v_last_t == 0.3 and st.video_frames == 4 and st.v_pending == 0
    f = st.v_frames[0]
    assert f.logits.shape == (1, 7) and f.emb.shape == (1, 512) and 0 < f.area[0] < 1 and np.isnan(f.mouth[0])
    ex = svc._experts["s"][1]
    # a frame queued before ctl start belongs to the old media clock
    stale = item(st, 0.5)
    with st.v_cv:
        st.v_epoch += 1
        st.v_frames.clear()
        st.v_last_t = float("-inf")
    run_frame(svc, stale)
    assert not st.v_frames and st.skipped_video_frames == 2
    run_frame(svc, item(st, 0.0))
    assert ex.resets == 1 and len(st.v_frames) == 1
    # the ring keeps the last 60 s of media time
    for t in np.arange(1.0, 62.0, 1.0):
        run_frame(svc, item(st, float(t)))
    assert st.v_frames[0].t >= 61.0 - 60.0 and st.v_frames[-1].t == 61.0
    # an undecodable payload is counted, not raised, by the thread loop
    bad = VideoItem(st, 0, 70.0, 0, 0, 320, 180, b"\xff\xd8 broken", st.v_epoch)
    with pytest.raises(ValueError):
        svc._vision_frame(bad)


def test_window_evidence_calibration_and_no_face(tmp_path):
    svc = bare_service(tmp_path)
    st = Stream("s", vad=None)
    for k in range(10):
        run_frame(svc, item(st, k / 10, BLUE if k < 8 else BLACK))
    with st.v_cv:
        st.env.extend((i * P.FRAME_S + P.FRAME_S / 2, -3.0) for i in range(40))
    tm, deg = {}, []
    res = svc._vision_window(st, 0.0, 1.0, tm, deg, wait=True)
    assert res.present and res.calibrated and res.n_frames == 8 and res.presence == pytest.approx(0.8)
    joy = EMOTIONS.index("joy")
    assert int(np.argmax(res.raw)) == joy and int(np.argmax(res.p_face)) == joy and res.p_face[joy] > res.raw[joy]
    assert deg == [] and tm["vision_wait"] < 20 and "vision_aggregate" in tm
    call = svc.vf.calls[-1]
    assert call == {"rule": "largest", "t0": 0.0, "t1": 1.0, "n_rows": 8, "n_env": 40, "n_frames": 10}
    b = res.block(svc.vision_rule)
    assert b.present and b.rule == "largest" and b.track == 0 and abs(sum(b.emotion.values()) - 1) < 1e-9
    # frames without a face, and a window without frames, give no_face and a uniform posterior
    for t0, t1 in ((0.8, 1.0), (5.0, 6.0)):
        deg = []
        res = svc._vision_window(st, t0, t1, {}, deg, wait=True)
        assert not res.present and np.allclose(res.p_face, UNIFORM) and deg == ["no_face"]
    # without package A the state says so and fusion3 is skipped
    svc.vf = None
    deg = []
    res = svc._vision_window(st, 0.0, 1.0, {}, deg, wait=False)
    assert not res.present and deg == ["vision_unavailable"]
    assert svc._fusion3_features(UNIFORM, UNIFORM, res, 0.9, None, deg) is None


def test_final_waits_for_the_vision_thread_at_most_vision_lag(tmp_path):
    svc = bare_service(tmp_path, vision_lag_ms=120.0)
    st = Stream("s", vad=None)
    run_frame(svc, item(st, 0.0))
    # a frame is still in processing and none reached t_media_end - 0.1 s: the wait times out
    with st.v_cv:
        st.v_pending = 1
    tm, deg = {}, []
    svc._vision_window(st, 0.0, 2.0, tm, deg, wait=True)
    assert "vision_lag" in deg and 110 <= tm["vision_wait"] < 400 and st.vision_lag_finals == 1

    # the vision thread catches up after 30 ms: the wait ends then, without vision_lag
    def catch_up():
        time.sleep(0.03)
        run_frame(svc, item(st, 1.95))
        with st.v_cv:
            st.v_pending -= 2
            st.v_cv.notify_all()

    with st.v_cv:
        st.v_pending = 1
    th = threading.Thread(target=catch_up)
    th.start()
    tm, deg = {}, []
    res = svc._vision_window(st, 0.0, 2.0, tm, deg, wait=True)
    th.join()
    assert "vision_lag" not in deg and 20 <= tm["vision_wait"] < 110 and res.present
    # nothing queued: no wait at all, even when the last frame is old (the video ended)
    tm, deg = {}, []
    svc._vision_window(st, 0.0, 9.0, tm, deg, wait=True)
    assert tm["vision_wait"] < 20 and "vision_lag" not in deg


def test_commit_ticks_first_emit_and_final_block(tmp_path):
    svc = bare_service(tmp_path)
    svc.commit_policy, svc.commit_policy_name = FakePolicy(), "fake_policy"
    st = Stream("s", vad=None)
    vis = VisionResult(True, np.eye(7)[4] * 0.9 + 0.1 / 7, UNIFORM, track=0, n_frames=10, presence=1.0)

    def job(t_end, utt=1, window="full"):
        return Job("provisional", "s", utt, None, None, window, [0, 1], 0.5, t_end, 0, 123, 0.0, None, [])

    fused = np.eye(7)[4] * 0.8 + 0.2 / 7
    assert svc._commit_tick(st, job(1.524), fused, UNIFORM, UNIFORM, vis, 0.9, 2) is None
    c = svc._commit_tick(st, job(2.556), fused, UNIFORM, UNIFORM, vis, 0.9, 5)
    assert c["label"] == "joy" and c["t_commit_s"] == 2.056 and c["q_emit"] == 0.7 and c["policy"] == "fake_policy"
    assert c["t_service_recv_ns"] == 123 and c["utt_id"] == 1
    ticks = svc.commit_policy.ticks
    assert [t["k"] for t in ticks] == [1, 2] and [t["same_label_run"] for t in ticks] == [1, 2]
    assert ticks[0]["prev_fused"] is None and np.allclose(ticks[1]["prev_fused"], fused)
    assert ticks[1]["face_present"] is True and ticks[1]["n_words"] == 5 and ticks[1]["asr_conf"] == 0.9
    # after the first emit the policy is not asked again for this utterance; tail windows never
    assert svc._commit_tick(st, job(3.6), fused, UNIFORM, UNIFORM, vis, 0.9, 7) is None and len(ticks) == 2
    assert svc._commit_tick(st, job(6.0, utt=2, window="tail"), fused, UNIFORM, UNIFORM, vis, 0.9, 7) is None

    final = State.empty("s", 1, "final", vision_mode=True)
    final.fused.label, final.fused.confidence = "neutral", 0.55
    fj = Job("final", "s", 1, None, None, "full", [0, 1], 0.5, 4.0, 0, 0, 512.0, None, [])
    cb = svc._final_commit(st, fj, final)
    assert cb.source == "policy" and cb.label == "joy" and cb.t_commit_s == 2.056 and cb.matches_final is False
    assert 1 not in st.ticks and 1 not in st.commits
    # no emit: the final is the commit, at the segment length plus the vad tail
    fj2 = Job("final", "s", 2, None, None, "full", [0, 1], 10.0, 12.5, 0, 0, 512.0, None, [])
    final.utt_id = 2
    cb = svc._final_commit(st, fj2, final)
    assert cb.source == "endpoint" and cb.label == "neutral" and cb.t_commit_s == 3.012 and cb.matches_final
    svc.commit_policy = None
    assert svc._final_commit(st, fj2, final) is None


# -- ingest: the provisional schedule and the media clock ----------------------

class FakeVad:
    """the VadSegmenter surface the ingest path reads. media time is pushed samples / 16 khz
    (no skip method, like VadSegmenter); speech starts when frame `start` is pushed, with
    `pre` frames of segment before it, and never ends."""
    silence_frames = 16

    def __init__(self, start: int = 34, pre: int = 8):
        self._n_samples, self.start, self.pre, self.seg = 0, start, pre, 0

    @property
    def t_media(self) -> float:
        return self._n_samples / P.SR

    def push(self, frame):
        self._n_samples += P.FRAME_SAMPLES
        if self._n_samples // P.FRAME_SAMPLES == self.start:
            self.seg = self.pre
            return [VadEvent("speech_start", self.t_media - self.pre * P.FRAME_S, None, False)]
        if self.seg:
            self.seg += 1
        return []

    def speech_pcm(self):
        return np.zeros(self.seg * P.FRAME_SAMPLES, np.float32)

    def reset(self, keep_clock: bool = True):
        self.seg = 0
        if not keep_clock:
            self._n_samples = 0


def push_audio(svc: Service, st: Stream, seqs) -> list[Job]:
    """ingest audio frames with these sequence numbers; the jobs they queued (the worker stays idle)."""
    jobs = []
    payload = np.zeros(P.FRAME_SAMPLES, np.int16).tobytes()
    for q in seqs:
        svc._on_audio(st, {"seq": q, "n_samples": P.FRAME_SAMPLES, "t_capture_ns": 0}, payload, 0)
        while (j := svc._jobs.get(0)) is not None:
            jobs.append(j)
    return jobs


def test_vision_provisionals_tick_on_the_segment_grid(tmp_path):
    """vision mode: full windows at about 1, 2, 3 and 4 s of segment (package D's ticks), the 4th
    still a full window; the core schedule is unchanged (a frame of drift per tick, 4.096 s is a tail)."""
    def windows(svc):
        st = Stream("s", vad=FakeVad())
        jobs = push_audio(svc, st, range(34 + 300))
        seg0 = jobs[0].t_media_start
        return [(j.window, round(j.t_media_end - seg0, 3), round(len(j.pcm) / P.SR, 3)) for j in jobs]

    vis = windows(bare_service(tmp_path / "v"))
    full = [w for w in vis if w[0] == "full"]
    assert [w[1] for w in full] == [1.024, 2.016, 3.008, 4.0] and all(w[1] == w[2] for w in full)
    assert [round(w[1] * 1.0) for w in full] == [1, 2, 3, 4]       # _commit_tick's k
    tails = [w for w in vis if w[0] == "tail"]
    assert tails[0][1] == 5.024 and all(w[2] == 4.0 for w in tails) and [w[1] for w in tails[:3]] == [5.024, 7.04, 9.056]
    core = windows(Service(ServiceConfig(heads=tmp_path)))
    assert [w[1] for w in core if w[0] == "full"] == [1.024, 2.048, 3.072]
    assert [w[1] for w in core if w[0] == "tail"][:2] == [4.096, 6.112]


def test_vision_mode_keeps_audio_on_the_video_media_clock_over_a_gap(tmp_path):
    """a sequence gap (long: audio_gap, or short) advances audio media time over the missing frames in
    vision mode, so the envelope and the face windows stay on the video's clock; the core counts
    received samples only, as before."""
    svc = bare_service(tmp_path / "v")
    st = Stream("s", vad=FakeVad(start=10_000))
    push_audio(svc, st, list(range(10)) + [30])           # 20 frames (640 ms) missing: an audio_gap
    assert st.vad.t_media == pytest.approx(31 * P.FRAME_S)
    assert st.env[-1][0] == pytest.approx(30 * P.FRAME_S + P.FRAME_S / 2)
    assert [m["code"] for t, m in drain(svc) if t == "error"] == ["audio_gap"]
    push_audio(svc, st, [33])                              # 2 frames missing: no audio_gap, same clock
    assert st.vad.t_media == pytest.approx(34 * P.FRAME_S) and not drain(svc)
    core = Service(ServiceConfig(heads=tmp_path))
    st = Stream("c", vad=FakeVad(start=10_000))
    push_audio(core, st, list(range(10)) + [30, 33])
    assert st.vad.t_media == pytest.approx(12 * P.FRAME_S)

    class WithSkip:
        def __init__(self):
            self.skipped = []

        def skip(self, n):
            self.skipped.append(n)

    v = WithSkip()
    skip_media(v, 1024)
    skip_media(v, 0)
    assert v.skipped == [1024]
    if (ROOT / "models/silero_vad.onnx").exists():
        from meld_rt.experts.vad import VadSegmenter

        real = VadSegmenter(ROOT / "models/silero_vad.onnx")
        real.push(np.zeros(P.FRAME_SAMPLES, np.int16))
        skip_media(real, 3 * P.FRAME_SAMPLES)
        assert real.t_media == pytest.approx(4 * P.FRAME_S)


class FakeSV:
    """SenseVoiceExpert.run's result, one scripted 9-way emotion posterior per call."""

    def __init__(self, emos):
        self.emos = list(emos)

    def run(self, pcm):
        n_ev = len(SV_EVENT_TOKENS)
        return types.SimpleNamespace(text="that was great", asr_conf=0.9, lang="en", tokens_per_s=3.0,
                                     emo_probs=np.asarray(self.emos.pop(0), np.float32),
                                     event_probs=np.full(n_ev, 1 / n_ev, np.float32), enc_pooled=None,
                                     timing_ms={"frontend": 1.0, "model": 2.0, "decode": 0.5, "total": 3.5})


def sv_emo(top: str, p: float = 0.6) -> np.ndarray:
    e = np.full(len(SV_EMO_TOKENS), (1 - p) / (len(SV_EMO_TOKENS) - 1))
    e[SV_EMO_TOKENS.index(top)] = p
    return e


def test_provisional_stability_compares_posteriors_of_one_kind(tmp_path):
    """vision mode with fusion3: full windows compare fusion3 with the previous fusion3 posterior; the
    first tail window compares its audio posterior with the previous window's audio posterior, not
    with fusion3's (which would measure the switch of model at 4 s)."""
    svc = bare_service(tmp_path)
    svc.mode = "poe"
    emos = [sv_emo("ANGRY"), sv_emo("ANGRY", 0.5), sv_emo("SAD"), sv_emo("ANGRY", 0.7), sv_emo("ANGRY", 0.65)]
    svc.sv = FakeSV(emos)
    st = svc._streams["s"] = Stream("s", vad=None)
    for k in range(60):
        run_frame(svc, item(st, k / 10))

    def prov(t_end, window="full"):
        t0 = 0.0 if window == "full" else t_end - 4.0
        svc._run_provisional(Job("provisional", "s", 1, np.zeros(16000, np.float32), None, window, [0, 1], t0, t_end, 0, 0,
                                 0.0, None, []))
        ((_, m),) = [x for x in drain(svc) if x[0] == "state.provisional"]
        return m

    ps = [prov(t) for t in (1.024, 2.016, 3.008, 4.0)] + [prov(5.024, "tail")]
    assert [p["fused"]["mode"] for p in ps] == ["stacked_tav"] * 4 + ["poe_uncalibrated"]
    fused = [np.array(list(p["fused"]["emotion"].values())) for p in ps]
    assert ps[0]["stability"] == 1.0
    assert ps[1]["stability"] == pytest.approx(1 - jsd(fused[0], fused[1]), abs=1e-4)
    a4, a5 = audio7(emos[3]), audio7(emos[4])
    assert ps[4]["stability"] == pytest.approx(1 - jsd(a4, a5), abs=1e-4)
    assert abs((1 - jsd(fused[3], fused[4])) - (1 - jsd(a4, a5))) > 0.01   # the cross-kind value differs


# -- the reply thread ---------------------------------------------------------

def reply_service(tmp_path, **kw) -> Service:
    svc = Service(ServiceConfig(heads=tmp_path, reply="llm", llm_dir=tmp_path / "fake-llm", **kw))
    svc._gen_mod = make_fake_gen_module()
    svc.generator = FakeGenerator(tmp_path / "fake-llm")
    svc.llm_model = "fake-llm"
    svc.low_conf, svc.high_conf = 0.4, 0.9
    return svc


def reply_job(conf=0.7) -> ReplyJob:
    from meld_rt.response.selector import Response

    s = State.empty("s", 3, "final", vision_mode=True)
    s.fused.label, s.fused.confidence = "joy", conf
    bank = Response("How was it for you?", "joy_none_02", "ask", ["fused.label=joy@0.70", "discrepancy.direction=none"])
    return ReplyJob("s", 3, s.to_dict(), bank, P.now_ns())


def test_llm_reply_streams_words_then_response(tmp_path):
    svc = reply_service(tmp_path)
    svc._reply_llm(reply_job())
    msgs = drain(svc)
    deltas = [m for t, m in msgs if t == "response.delta"]
    (resp,) = [m for t, m in msgs if t == "response"]
    assert [t for t, _ in msgs] == ["response.delta"] * len(WORDS) + ["response"]
    assert [d["seq"] for d in deltas] == list(range(len(WORDS))) and "".join(d["text"] for d in deltas) == "".join(WORDS)
    assert resp["text"] == "".join(WORDS) and resp["source"] == "llm" and resp["fallback"] is None
    assert resp["template_id"] == "llm" and resp["style"] == "generated" and resp["model"] == "fake-llm"
    assert resp["grounding"] == ["fused.label=joy@0.70"] and resp["checks"]["n_words"] == 5 and resp["n_tokens"] == 9
    assert resp["first_delta_ms"] >= 0 and resp["gen_ms"] >= resp["first_delta_ms"] - 1 and resp["stop_reason"] == "eos"
    # the prompt gets the selector's thresholds (its confidence word matches the selector's bucket), and the
    # generator the reply deadline
    assert svc._gen_mod.calls == [{"low_conf": 0.4, "high_conf": 0.9}]
    assert svc.generator.deadlines == [svc.cfg.llm_reply_timeout_s]


@pytest.mark.parametrize("case", ["missing", "raises", "slow"])
def test_llm_reply_falls_back_to_the_bank_line(tmp_path, case):
    svc = reply_service(tmp_path, llm_first_token_timeout_s=0.1)
    if case == "missing":
        svc.generator, svc.gen_error = None, "generator_unavailable: FileNotFoundError"
    elif case == "raises":
        svc.generator.fail = True
    else:
        svc.generator.first_word_delay_s = 0.4
    t0 = time.perf_counter()
    svc._reply_llm(reply_job())
    waited = time.perf_counter() - t0
    msgs = drain(svc)
    assert [t for t, _ in msgs] == ["response.delta", "response"]
    d, r = msgs[0][1], msgs[1][1]
    assert d["seq"] == 0 and d["text"] == r["text"] == "How was it for you?"
    assert r["source"] == "bank" and r["template_id"] == "joy_none_02" and r["style"] == "ask"
    assert r["grounding"] == ["fused.label=joy@0.70", "discrepancy.direction=none"]
    want = {"missing": "generator_unavailable: FileNotFoundError", "raises": "generator_error: RuntimeError",
            "slow": "first_token_timeout"}[case]
    assert r["fallback"] == want
    if case == "slow":
        # the late generation is stopped at its first word, and the reply thread waits for it
        assert svc.generator.calls == 1 and 0.35 < waited < 1.5
    if case != "missing":
        assert r["checks"] is not None


def test_a_reply_that_stalls_after_its_first_word_ends_at_the_deadline(tmp_path):
    """the first word comes, then nothing: the response goes out at the deadline with the words
    streamed so far, and a generation that does not end is marked stuck (health then says so)."""
    svc = reply_service(tmp_path, llm_reply_timeout_s=0.3)
    svc.generator.stall_after_first_s = 2.0
    sent = []
    svc._send = lambda topic, payload: sent.append((topic, payload))
    t0 = time.perf_counter()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("meld_rt.service.STUCK_JOIN_S", 0.2)
        svc._reply_llm(reply_job())
    waited = time.perf_counter() - t0
    msgs = drain(svc)
    assert [t for t, _ in msgs] == ["response.delta", "response"]
    r = msgs[1][1]
    assert r["text"] == "You" and r["source"] == "llm" and r["fallback"] == "reply_timeout_after_1_words"
    assert r["checks"]["n_words"] == 1 and r["first_delta_ms"] >= 0
    assert 1.2 < waited < 2.5            # deadline 0.3 s + grace 1.0 s + 0.2 s join, not the 2 s stall
    assert svc.generator is None and svc.gen_error == "generator_stuck"
    svc._health()
    (h,) = [m for t, m in sent if t == "health"]
    assert h["generator"] == "generator_stuck"
    # later replies are bank lines
    svc._reply_llm(reply_job())
    assert drain(svc)[1][1]["fallback"] == "generator_stuck"


def test_generator_status_in_health():
    svc = Service(ServiceConfig(reply="llm"))
    assert svc.generator_status() is None            # not loaded yet
    svc.gen_error = "no_generator_module"
    assert svc.generator_status() == "no_generator_module"
    svc.generator, svc.llm_model, svc.gen_error = object(), "qwen", None
    assert svc.generator_status() == "qwen"
    svc.generator, svc.gen_error = None, "generator_stuck"
    assert svc.generator_status() == "generator_stuck"
    assert Service(ServiceConfig(vision=True)).generator_status() is None


def test_reply_queue_counts_a_taken_job_until_done():
    from meld_rt.service import ReplyQueue

    q = ReplyQueue()
    assert q.idle()
    q.put(reply_job())
    assert not q.idle()
    job = q.get(0.1)
    assert job is not None and len(q) == 0 and not q.idle()
    q.done()
    assert q.idle()


def test_superseded_reply_takes_the_bank_line(tmp_path):
    svc = reply_service(tmp_path)
    svc._reply_q.put(reply_job())
    svc._reply_q.put(reply_job())
    first = svc._reply_q.get(0.1)
    assert svc._reply_q.has_newer(first.stream_id)
    svc._reply_bank(first, "superseded")
    assert drain(svc)[1][1]["fallback"] == "superseded"


# -- real files where they exist ------------------------------------------------

@pytest.mark.skipif(not (YUNET.exists() and FER.exists()), reason="vision models not fetched")
def test_real_vision_expert_in_the_vision_path(tmp_path):
    """yunet and fer on synthetic frames (no face in them): the frames are processed and the
    window has no face."""
    svc = bare_service(tmp_path)
    svc.new_vision_expert = Service.new_vision_expert.__get__(svc)
    st = Stream("s", vad=None)
    rng = np.random.default_rng(0)
    for k in range(3):
        f = rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)
        it = VideoItem(st, k, k / 10, 0, 0, 640, 360, P.encode_jpeg(f), st.v_epoch)
        with st.v_cv:
            st.v_pending += 1
        run_frame(svc, it)
    assert st.video_frames == 3 and all(len(f.track) == 0 for f in st.v_frames)
    deg = []
    res = svc._vision_window(st, 0.0, 1.0, {}, deg, wait=True)
    assert not res.present and deg == ["no_face"]


def test_real_vision_features_module_takes_the_service_rows(tmp_path):
    """package A's module, when it exists, reads the rows the service builds (frame_t included)."""
    vf = pytest.importorskip("meld_rt.vision_features")
    svc = bare_service(tmp_path)
    svc.vf = vf
    st = Stream("s", vad=None)
    for k in range(10):
        run_frame(svc, item(st, k / 10, BLUE if k % 2 == 0 else BLACK))
    with st.v_cv:
        st.env.extend((i * P.FRAME_S + P.FRAME_S / 2, float(np.sin(i))) for i in range(40))
    deg = []
    res = svc._vision_window(st, 0.0, 1.0, {}, deg, wait=False)
    assert res.present and res.n_frames == 5 and res.presence == pytest.approx(0.5) and deg == []
    assert int(np.argmax(res.raw)) == EMOTIONS.index("joy")
    x = svc._fusion3_features(UNIFORM, UNIFORM, res, 0.9, None, deg)
    assert x.shape == (29,) and x[-2] == 1.0 and x[-1] == pytest.approx(0.5)
    assert svc._log_rms(np.full(512, 3277, np.int16)) == pytest.approx(vf.frame_log_rms(np.full(512, 3277, np.int16)))


class ScriptedExpert:
    """VisionExpert stand-in that returns given Face lists, one list per processed frame."""

    def __init__(self, frames):
        self.frames, self.n = frames, 0

    def reset(self):
        self.n = 0

    def process(self, frame):
        self.n += 1
        return self.frames[self.n - 1]


class RecordingVF:
    """the real vision_features module, with every face_evidence result kept."""

    def __init__(self, vf):
        self._vf, self.evidence = vf, []

    def __getattr__(self, name):
        return getattr(self._vf, name)

    def face_evidence(self, *a, **k):
        ev = self._vf.face_evidence(*a, **k)
        self.evidence.append(ev)
        return ev


def test_live_rows_and_the_video_cache_give_the_same_features(tmp_path, monkeypatch):
    """the same Face lists through the service's live row builder (_vision_frame -> face_frame into the
    stream's ring, _face_evidence -> face_rows over the window) and through a written and read video
    cache (vision_features.clip_rows, what training and evaluation read) give the same vision_features_B
    under every speaker rule, for the whole clip and for a part of it."""
    vf = pytest.importorskip("meld_rt.vision_features")
    monkeypatch.syspath_prepend(str(ROOT / "tests"))
    from test_vision_features import FPS, H, W, synthetic_faces, write_video_cache

    faces = synthetic_faces(15, 2)
    for fs in faces:   # values at the cache's storage precision (float16 embeddings, float32 scalars)
        for i, f in enumerate(fs):
            fs[i] = Face(f.track, f.box, f.landmarks, float(np.float32(f.score)), f.logits,
                         None if f.embed is None else f.embed.astype(np.float16).astype(np.float32),
                         float(np.float32(f.mouth_motion)), float(np.float32(f.face_motion)))
    frames = faces + [[], [], []]                                   # three trailing frames without a face
    write_video_cache(tmp_path / "video_test.npz", {"1_1": faces}, {"1_1": 3})
    off = vf.clip_rows(vf.load_video_cache("test", tmp_path), "1_1")
    pcm16 = (np.random.default_rng(1).normal(0, 3000, 16000 * 2) * np.repeat(np.sin(np.arange(20) / 3) ** 2, 1600)).astype(np.int16)
    env_off = vf.audio_envelope(pcm16.astype(np.float32) / 32768.0)

    svc = bare_service(tmp_path)
    svc.vf = rec = RecordingVF(vf)
    svc.new_vision_expert = lambda: ScriptedExpert(frames)
    st = Stream("s", vad=None)
    for k in range(len(frames)):
        with st.v_cv:
            st.v_pending += 1
        run_frame(svc, VideoItem(st, k, k / FPS, 0, 0, W, H, jpeg(BLACK, W, H), st.v_epoch))
    assert len(st.v_frames) == len(frames)
    with st.v_cv:   # the envelope as the ingest thread keeps it: one value per received frame, at its centre
        st.env.extend((i * P.FRAME_S + P.FRAME_S / 2, vf.frame_log_rms(fr)) for i, fr in enumerate(P.pcm16_frames(pcm16)))
    tracks = set()
    for rule in vf.RULES:
        svc.vision_rule = rule
        for (t0, t1), (o0, o1) in (((0.0, len(frames) / FPS), (-np.inf, np.inf)), ((0.3, 1.2), (0.3, 1.2))):
            n = len(rec.evidence)
            svc._face_evidence(st, t0, t1, [])
            assert len(rec.evidence) == n + 1
            live, ref = rec.evidence[-1], vf.face_evidence(off, *env_off, rule, o0, o1)
            assert (live.present, live.track, live.n_frames, live.n_tracks, live.n_window_frames) == (
                ref.present, ref.track, ref.n_frames, ref.n_tracks, ref.n_window_frames), (rule, t0)
            assert np.array_equal(vf.vision_features_B(live), vf.vision_features_B(ref)), (rule, t0)
            tracks.add((t0, live.track))
    assert len({tr for t0, tr in tracks if t0 == 0.0}) > 1     # the rules do not all pick the same track


# -- end to end ---------------------------------------------------------------

@pytest.fixture(scope="module")
def vservice(tmp_path_factory):
    heads = write_heads(tmp_path_factory.mktemp("vheads") / "heads")
    cfg = ServiceConfig(in_addr="tcp://127.0.0.1:*", out_addr="tcp://127.0.0.1:*", threads=THREADS, text_threads=THREADS,
                        preload=True, heads=heads, vision=True, reply="llm", llm_dir=tmp_path_factory.mktemp("fake-llm"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "meld_rt.vision_features", make_fake_vf())
        mp.setitem(sys.modules, "meld_rt.commit", make_fake_commit_module())
        mp.setitem(sys.modules, "meld_rt.response.generator", make_fake_gen_module())
        svc = Service(cfg)
        svc.new_vision_expert = FakeExpert
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

    def of(self, topic: str, sid: str) -> list[dict]:
        return [m for t, m in self.msgs if t == topic and m.get("stream_id") == sid]


@pytest.fixture
def vclient(vservice):
    c = Client(vservice)
    yield c
    c.close()


def has(topic: str, sid: str):
    return lambda ms: any(t == topic and m.get("stream_id") == sid for t, m in ms)


@have_audio_models
def test_audio_and_video_give_face_states_a_commit_and_a_streamed_reply(vservice, vclient):
    sid = "tav-en"
    pcm = load_wav(EN_WAV)
    n_video = int(len(pcm) / P.SR * 10)
    clip = {"pcm": pcm, "jpegs": [jpeg(BLUE)] * n_video, "w": 320, "h": 180, "fps": 10.0}
    sender = threading.Thread(target=replay_video, args=(vclient.push, clip, sid, True), kwargs={"tail_s": 1.0}, daemon=True)
    sender.start()
    assert vclient.collect(has("response", sid), timeout_s=len(pcm) / P.SR + 30, grace_s=1.2), "no reply"
    sender.join(timeout=5)
    provs, finals = vclient.of("state.provisional", sid), vclient.of("state.final", sid)
    commits, deltas, resps = vclient.of("state.commit", sid), vclient.of("response.delta", sid), vclient.of("response", sid)
    assert not vclient.of("error", sid)
    assert len(finals) == 1 and len(resps) == 1 and len(commits) == 1 and provs
    f, r, c = finals[0], resps[0], commits[0]

    full = [p for p in provs if p["window"] == "full"]
    assert full and all(p["vision"]["present"] and p["fused"]["mode"] == "stacked_tav" for p in full)
    assert all(p["vision"]["rule"] == "largest" and "commit" not in p for p in provs)
    assert any(p["text"]["emotion"] != {e: 1 / 7 for e in EMOTIONS} for p in full), "no text posterior on a partial"
    # the fake policy emits at the second tick (about 2 s in), or later when a provisional was dropped under load
    assert c["utt_id"] == f["utt_id"] and c["policy"] == "fake_policy" and 1.5 < c["t_commit_s"] < 4.2
    assert c["label"] in EMOTIONS and c["q_emit"] == 0.7 and c["t_emitted_ns"] < f["t_emitted_ns"]

    v = f["vision"]
    assert v["present"] and v["rule"] == "largest" and v["track"] == 0 and v["presence"] > 0.8 and v["n_frames"] >= 40
    assert max(v["raw"], key=v["raw"].get) == "joy" and max(v["emotion"], key=v["emotion"].get) == "joy"
    fu = f["fused"]
    assert fu["mode"] == "stacked_tav" and fu["calibrated"] is True and fu["sentiment"] in SENTIMENTS
    assert abs(sum(fu["emotion"].values()) - 1) < 1e-6
    assert f["discrepancy"]["face_jsd"] is not None and f["discrepancy"]["face_direction"] in (
        "none", "face_more_negative", "face_more_positive", "face_surprised", "mixed")
    assert f["commit"] == {"label": c["label"], "confidence": c["confidence"], "t_commit_s": c["t_commit_s"],
                           "matches_final": c["label"] == fu["label"], "source": "policy"}
    assert "vision_lag" not in f["degraded"] and "no_face" not in f["degraded"]
    assert f["timing_ms"]["vision_wait"] < vservice.cfg.vision_lag_ms and f["timing_ms"]["vision_aggregate"] < 50
    State.from_dict(f)

    assert [d["seq"] for d in deltas] == list(range(len(WORDS))) and "".join(d["text"] for d in deltas) == r["text"]
    assert r["source"] == "llm" and r["fallback"] is None and r["utt_id"] == f["utt_id"] and r["first_delta_ms"] > 0
    assert r["t_emitted_ns"] > deltas[-1]["t_emitted_ns"] > f["t_emitted_ns"]
    h = vclient.of("health", sid)[-1]
    assert h["video_frames"] >= n_video - 2 and h["dropped_video_frames"] == 0 and h["commit_policy"] == "fake_policy"
    print(f"\ntiming {f['timing_ms']}, vision_wait {f['timing_ms']['vision_wait']} ms, "
          f"fused {fu['label']}@{fu['confidence']}, commit {c['label']} at {c['t_commit_s']} s, reply {r['text']!r} "
          f"first delta {r['first_delta_ms']} ms")


@have_audio_models
def test_audio_without_video_has_no_face_and_does_not_wait(vservice, vclient):
    sid = "tav-novideo"
    replay(vclient.push, load_wav(EN_WAV), sid, realtime=False, tail_s=1.0)
    assert vclient.collect(has("response", sid), timeout_s=40, grace_s=0.3)
    (f,) = vclient.of("state.final", sid)
    assert f["vision"]["present"] is False and "no_face" in f["degraded"] and "vision_lag" not in f["degraded"]
    assert f["timing_ms"]["vision_wait"] < 20 and f["fused"]["mode"] == "stacked_tav"
    assert f["discrepancy"]["face_jsd"] is None and f["discrepancy"]["face_flag"] is False
    assert vclient.of("response", sid)[0]["source"] == "llm"


@have_audio_models
def test_typed_text_in_vision_mode_with_bank_reply(vservice, vclient):
    """a text message has no media window (vision and commit null); with --reply bank the
    line goes out as one delta and the response keeps its core fields."""
    sid = "tav-text"
    vservice.cfg.reply = "bank"
    try:
        send_text(vclient.push, sid, "well that went great")
        assert vclient.collect(has("response", sid), timeout_s=20, grace_s=0.2)
    finally:
        vservice.cfg.reply = "llm"
    (f,), (d,), (r,) = vclient.of("state.final", sid), vclient.of("response.delta", sid), vclient.of("response", sid)
    assert f["vision"] is None and f["commit"] is None and f["fused"]["mode"] == "text_only"
    assert d["seq"] == 0 and d["text"] == r["text"] and "source" not in r and r["template_id"]


def test_health_reports_the_extension(vservice):
    c = Client(vservice)
    try:
        h = next(m for t, m in c.msgs if t == "health")
    finally:
        c.close()
    assert h["vision"] is True and h["reply"] == "llm" and h["generator"] and "video_queue" in h


class NeverEmit(FakePolicy):
    name = "never"

    def decide(self, x):
        return False, 0.2, 0.8


@have_audio_models
def test_a_long_utterance_gets_commit_ticks_1_2_3_4(vservice, vclient):
    """en.wav is one 5.6 s utterance: four full-window ticks near 1, 2, 3 and 4 s of segment, the ticks
    of package D's traces. a policy that never emits sees all four; the final is the commit."""
    sid = "tav-ticks"
    pol = NeverEmit()
    saved = vservice.commit_policy, vservice.commit_policy_name
    vservice.commit_policy, vservice.commit_policy_name = pol, pol.name
    try:
        pcm = load_wav(EN_WAV)
        n_video = int(len(pcm) / P.SR * 10)
        clip = {"pcm": pcm, "jpegs": [jpeg(BLUE)] * n_video, "w": 320, "h": 180, "fps": 10.0}
        sender = threading.Thread(target=replay_video, args=(vclient.push, clip, sid, True), kwargs={"tail_s": 1.0},
                                  daemon=True)
        sender.start()
        assert vclient.collect(has("response", sid), timeout_s=len(pcm) / P.SR + 30, grace_s=0.3), "no reply"
        sender.join(timeout=5)
    finally:
        vservice.commit_policy, vservice.commit_policy_name = saved
    (f,) = vclient.of("state.final", sid)
    assert f["commit"]["source"] == "endpoint" and not vclient.of("state.commit", sid)
    full = [p for p in vclient.of("state.provisional", sid) if p["window"] == "full"]
    spans = [round(p["t_media_end"] - p["t_media_start"], 3) for p in full]
    dropped = vservice._streams[sid].dropped_provisionals
    print(f"\nfull-window ticks at {spans} s, k {[t['k'] for t in pol.ticks]}, dropped provisionals {dropped}")
    assert [t["t_s"] for t in pol.ticks] == spans
    if dropped == 0:
        assert [t["k"] for t in pol.ticks] == [1, 2, 3, 4] and spans == [1.024, 2.016, 3.008, 4.0]
    else:   # a provisional dropped under load is not a tick; the others stay on the grid
        assert set(spans) <= {1.024, 2.016, 3.008, 4.0} and [t["k"] for t in pol.ticks] == [round(x) for x in spans]


@pytest.fixture(scope="module")
def core_service():
    cfg = ServiceConfig(in_addr="tcp://127.0.0.1:*", out_addr="tcp://127.0.0.1:*", threads=THREADS, text_threads=THREADS,
                        preload=True)
    svc = Service(cfg)
    svc.start()
    assert svc.wait_ready(120), "service did not load its models"
    yield svc
    svc.stop()


CORE_HEALTH_KEYS = {"stream_id", "dropped_provisionals", "queue_s", "queue_audio_s", "fusion_mode", "model_versions",
                    "rss_mb", "models_loaded", "uptime_s", "t_emitted_ns"}
CORE_RESPONSE_KEYS = {"stream_id", "utt_id", "text", "template_id", "style", "grounding", "t_emitted_ns"}


@have_audio_models
def test_core_service_publishes_the_core_topics_and_keys_only(core_service):
    """no --vision, --reply bank: no response.delta, no state.commit, no vision, commit, face or
    vision timing keys, no extension health keys; video gets one bad_message and opens no stream."""
    c = Client(core_service)
    try:
        frame = np.zeros((48, 64, 3), np.uint8)
        for k in range(3):
            c.push.send_multipart(P.frame_msg("core-cam", k, P.now_ns(), k / 10, frame))
        replay(c.push, load_wav(EN_WAV), "core-en", realtime=False, tail_s=1.0)
        assert c.collect(has("response", "core-en"), timeout_s=40, grace_s=1.2)
    finally:
        c.close()
    assert {t for t, _ in c.msgs} <= {"health", "state.provisional", "state.final", "response", "error"}
    errors = [m for t, m in c.msgs if t == "error"]
    assert [(e["stream_id"], e["code"]) for e in errors] == [("core-cam", "bad_message")]
    assert "core-cam" not in core_service._streams
    (f,) = c.of("state.final", "core-en")
    core = State.empty("x", 0, "final").to_dict()
    assert list(f) == list(core) and list(f["discrepancy"]) == list(core["discrepancy"])
    assert not set(FACE_DISCREPANCY_KEYS) & set(f["discrepancy"]) and not set(VISION_TIMING_KEYS) & set(f["timing_ms"])
    assert f["fused"]["mode"] != "stacked_tav"
    for p in c.of("state.provisional", "core-en"):
        assert "vision" not in p and "commit" not in p and p["fused"]["mode"] != "stacked_tav"
    (r,) = c.of("response", "core-en")
    assert set(r) == CORE_RESPONSE_KEYS
    healths = [m for t, m in c.msgs if t == "health"]
    assert healths and all(set(h) == CORE_HEALTH_KEYS for h in healths)


def test_face_frame_without_faces():
    ff = face_frame(0.5, (180, 320, 3), [])
    assert isinstance(ff, FaceFrame) and ff.t == 0.5 and ff.emb.shape == (0, 512) and ff.logits.shape == (0, 7)
