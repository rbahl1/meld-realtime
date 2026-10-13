"""vision features (extension spec 2.1) and the vision heads: audio envelope, speaker rules,
face evidence and its feature layouts, the fusion3 layout, the words-face discrepancy,
the offline cache reader against the live ring, and the new head classes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from meld_rt import heads, labels  # noqa: E402
from meld_rt import protocol as P  # noqa: E402
from meld_rt import vision_features as vf  # noqa: E402
from meld_rt.experts.vision import FER_CLASSES, Face, posterior_meld  # noqa: E402

FPS = 10.0


def env_for(t_end: float = 2.0, freq: float = 1.5):
    """an envelope sampled like audio_envelope: 32 ms frames, centres."""
    t = np.arange(0, t_end, vf.HOP_S) + vf.HOP_S / 2
    return t, np.sin(2 * np.pi * freq * t)


def rows_of(spec: list[dict], frame_t=None) -> vf.FaceRows:
    """spec: one dict per track with frames, area, mouth (callable of t or value), face, logits, score."""
    t, tr, area, score, lg, mouth, face = [], [], [], [], [], [], []
    for s in spec:
        for j, k in enumerate(s["frames"]):
            tk = k / FPS
            t.append(tk)
            tr.append(s["track"])
            area.append(s["area"])
            score.append(s.get("score", 0.9))
            lg.append(s.get("logits", np.zeros(7)))
            m = s["mouth"](tk) if callable(s["mouth"]) else s["mouth"]
            mouth.append(np.nan if j == 0 else m)
            face.append(np.nan if j == 0 else s.get("face", 0.0))
    return vf.FaceRows(np.array(t), np.array(tr), np.array(area), np.array(score), np.array(lg).reshape(-1, 7),
                       np.array(mouth), np.array(face), None, frame_t)


def three_tracks():
    """track 0 biggest but short and still; track 1 seen in every frame with a mouth that follows the
    envelope; track 2 smallest with the highest mouth-minus-face motion."""
    et, ev = env_for()
    rng = np.random.default_rng(3)
    noise = {k: rng.normal(0, 0.01) for k in range(20)}
    spec = [
        {"track": 0, "frames": range(10), "area": 0.05, "mouth": lambda t: 0.2 + noise[int(round(t * FPS))], "face": 0.4},
        {"track": 1, "frames": range(20), "area": 0.02, "mouth": lambda t: 0.5 + 0.2 * np.interp(t, et, ev), "face": 0.1},
        {"track": 2, "frames": range(5, 20), "area": 0.01, "mouth": lambda t: 0.9 + noise[int(round(t * FPS))], "face": 0.0},
    ]
    return rows_of(spec, np.arange(20) / FPS), et, ev


# ---------------------------------------------------------------- envelope

def test_audio_envelope_frames_and_values():
    x = np.full(1000, 0.25, np.float32)
    t, v = vf.audio_envelope(x)
    assert np.allclose(t, [0.016, 0.048])
    assert np.isclose(v[0], np.log(0.25))
    assert np.isclose(v[1], np.log(0.25 * np.sqrt(488 / 512)))   # the last frame is zero-padded as the client pads it
    t0, v0 = vf.audio_envelope(np.zeros(0))
    assert t0.size == v0.size == 0
    assert np.isclose(vf.audio_envelope(np.zeros(512))[1][0], np.log(vf.LOG_RMS_FLOOR))


def test_live_envelope_equals_offline():
    rng = np.random.default_rng(0)
    pcm16 = (rng.normal(0, 3000, 16000 * 2)).astype(np.int16)
    t_off, v_off = vf.audio_envelope(pcm16.astype(np.float32) / 32768.0)
    ring = vf.EnvelopeRing()
    for i, fr in enumerate(P.pcm16_frames(pcm16)):
        ring.add_frame(i * P.FRAME_S, fr)
    t_live, v_live = ring.arrays()
    assert np.allclose(t_live, t_off) and np.allclose(v_live, v_off, atol=1e-6)
    assert np.isclose(vf.frame_log_rms(pcm16[:512]), vf.frame_log_rms(P.pcm16_to_f32(pcm16[:512].tobytes())))


def test_envelope_ring_keeps_max_s():
    ring = vf.EnvelopeRing(max_s=1.0)
    for i in range(100):
        ring.add_frame(i * P.FRAME_S, np.ones(512, np.int16))
    t, _ = ring.arrays()
    assert t[-1] - t[0] <= 1.0 + 1e-9 and len(t) < 100


# ---------------------------------------------------------------- speaker rules

def test_each_rule_picks_its_track():
    rows, et, ev = three_tracks()
    chosen = {r: vf.select_speaker(rows, et, ev, r) for r in vf.RULES}
    assert {r: c[0] for r, c in chosen.items()} == {"largest": 0, "present": 1, "asd": 1, "mouth": 2}
    assert chosen["asd"][1]["asd_corr"] > 0.99 and not chosen["asd"][1]["fallback"]
    assert all(c[1]["n_tracks"] == 3 for c in chosen.values())
    # the chosen track's asd correlation is reported under every rule
    assert np.isfinite(chosen["largest"][1]["asd_corr"]) and chosen["largest"][1]["asd_corr"] < 0.9


def test_present_tie_goes_to_larger_area_then_lower_id():
    spec = [{"track": 0, "frames": range(5), "area": 0.01, "mouth": 0.1},
            {"track": 1, "frames": range(5), "area": 0.03, "mouth": 0.1},
            {"track": 2, "frames": range(5), "area": 0.03, "mouth": 0.1}]
    assert vf.select_speaker(rows_of(spec), None, None, "present")[0] == 1
    assert vf.select_speaker(rows_of(spec), None, None, "largest")[0] == 1


def test_asd_and_mouth_fall_back_to_largest():
    et, ev = env_for()
    # three frames per track: two motion samples, too few for asd (4) and mouth (3)
    spec = [{"track": 0, "frames": range(3), "area": 0.01, "mouth": lambda t: np.interp(t, et, ev)},
            {"track": 1, "frames": range(3), "area": 0.04, "mouth": 0.3}]
    rows = rows_of(spec)
    for rule in ("asd", "mouth"):
        tr, info = vf.select_speaker(rows, et, ev, rule)
        assert tr == 1 and info["fallback"] and np.isnan(info["asd_corr"]), rule
    # no audio envelope: asd has nothing to correlate with
    rows, _, _ = three_tracks()
    tr, info = vf.select_speaker(rows, np.zeros(0), np.zeros(0), "asd")
    assert tr == 0 and info["fallback"]


def test_single_track_same_under_every_rule():
    et, ev = env_for()
    for frames in (range(1), range(2), range(12)):
        rows = rows_of([{"track": 7, "frames": frames, "area": 0.02, "mouth": lambda t: np.interp(t, et, ev)}])
        assert {vf.select_speaker(rows, et, ev, r)[0] for r in vf.RULES} == {7}


def test_no_rows_no_speaker_and_bad_rule():
    assert vf.select_speaker(vf.FaceRows.empty(), None, None, "asd")[0] is None
    with pytest.raises(ValueError):
        vf.select_speaker(vf.FaceRows.empty(), None, None, "loudest")


# ---------------------------------------------------------------- evidence and features

def test_face_evidence_fields_and_window():
    rows, et, ev = three_tracks()
    rng = np.random.default_rng(1)
    rows.logits = rng.normal(0, 1, rows.logits.shape)
    emb = rng.normal(0, 1, (len(rows), 512)).astype(np.float32)
    emb[::2] = np.nan
    rows.emb = emb
    e = vf.face_evidence(rows, et, ev, "asd", 0.5, 1.5)        # frames 5..14
    m = (rows.track == 1) & (rows.t >= 0.5) & (rows.t < 1.5)
    assert e.present and e.track == 1 and e.n_frames == 10 and e.n_window_frames == 10
    assert np.isclose(e.presence, 1.0) and e.n_tracks == 3        # track 0 has frames 5..9 in the window
    assert np.allclose(e.p_face, posterior_meld(rows.logits[m]).mean(axis=0))
    assert np.isclose(e.log_area, np.log(0.02)) and np.isclose(e.score, 0.9)
    assert np.isclose(e.mouth, np.nanmean(rows.mouth[m]))
    keep = m & np.isfinite(emb).all(axis=1)
    assert np.allclose(e.emb, emb[keep].mean(axis=0), atol=1e-6)
    assert 0.9 < e.asd_corr <= 1.0
    # presence counts every processed frame of the window, faces or not
    e2 = vf.face_evidence(rows, et, ev, "largest", 0.0, 2.0)
    assert e2.track == 0 and e2.n_window_frames == 20 and np.isclose(e2.presence, 0.5)
    # without frame_t, only the frames with a face count
    rows.frame_t = None
    assert vf.face_evidence(rows, et, ev, "largest", 0.0, 2.0).n_window_frames == 20
    # the window is half open: t1 itself is out
    assert vf.face_evidence(rows, et, ev, "asd", 0.5, 0.6).n_frames == 1


def test_absent_evidence_and_its_features():
    e = vf.face_evidence(vf.FaceRows.empty(np.arange(5) / FPS), None, None, "asd")
    assert not e.present and e.track is None and e.n_window_frames == 5
    assert np.allclose(e.p_face, 1 / 7) and e.presence == 0 and e.log_area == 0 and not e.emb.any()
    a = vf.vision_features_A(e)
    assert a.shape == (16,) and a.dtype == np.float32
    assert np.allclose(a[:7], np.log(1 / 7)) and np.isclose(a[7], 1 / 7) and np.isclose(a[8], np.log2(7), atol=1e-5)
    assert not a[9:].any()
    b = vf.vision_features_B(e)
    assert b.shape == (528,) and np.array_equal(b[:16], a) and not b[16:].any()


def test_feature_layout_A_and_B():
    rows, et, ev = three_tracks()
    rows.emb = np.ones((len(rows), 512), np.float32) * np.arange(len(rows))[:, None]
    e = vf.face_evidence(rows, et, ev, "asd")
    a = vf.vision_features_A(e)
    p = np.clip(e.p_face, heads.EPS, 1)
    named = dict(zip(vf.FEATURES_A, a))
    assert np.allclose(a[:7], np.log(p), atol=1e-6)
    assert np.isclose(named["max_p"], p.max()) and np.isclose(named["entropy_bits"], heads.entropy_bits(np.log(p)), atol=1e-5)
    assert np.isclose(named["presence"], e.presence) and np.isclose(named["log_area"], e.log_area)
    assert np.isclose(named["n_tracks_div4"], 3 / 4) and np.isclose(named["asd_corr"], e.asd_corr)
    assert np.isclose(named["mouth_x10"], 10 * e.mouth, atol=1e-5) and named["present"] == 1.0
    b = vf.vision_features(e, "B")
    assert np.allclose(b[16:], e.emb) and np.array_equal(vf.vision_features(e, "A"), a)
    with pytest.raises(ValueError):
        vf.vision_features(e, "C")


def test_fusion3_layout_and_absent_face():
    rng = np.random.default_rng(0)
    lt, la, lf = (heads.log_softmax(rng.normal(size=7)) for _ in range(3))
    x = vf.build_fusion3_features(lt, la, lf, 0.8, 0.1, 0.2, True, 0.75)
    assert x.shape == (29,) and x.dtype == np.float32 and len(vf.FEATURES_FUSION3) == 29 == heads.FUSION3_DIM
    assert np.allclose(x[:7], lt) and np.allclose(x[7:14], la) and np.allclose(x[14:21], lf)
    assert np.allclose(x[21:24], [heads.entropy_bits(lt), heads.entropy_bits(la), heads.entropy_bits(lf)], atol=1e-5)
    assert np.allclose(x[24:], [0.8, 0.1, 0.2, 1.0, 0.75])
    # the two-way layout is the same text, audio and tail values
    two = heads.build_fusion_features(lt, la, 0.8, 0.1, 0.2)
    assert np.allclose(x[:14], two[:14]) and np.allclose(x[[21, 22, 24, 25, 26]], two[14:])
    y = vf.build_fusion3_features(lt, la, lf, 0.8, 0.1, 0.2, False, 0.75)
    assert np.allclose(y[14:21], np.log(1 / 7)) and np.isclose(y[23], np.log2(7), atol=1e-5) and y[27] == 0 and y[28] == 0
    with pytest.raises(ValueError):
        vf.build_fusion3_features(lt[:6], la, lf, 0.8, 0.1, 0.2, True, 0.75)


def onehot(name: str, p: float = 0.9) -> np.ndarray:
    v = np.full(7, (1 - p) / 6)
    v[labels.EMO_INDEX[name]] = p
    return v


def test_face_discrepancy_direction_and_flag():
    d, flag, direction = vf.face_discrepancy(onehot("joy"), onehot("anger"), 0.3)
    assert flag and direction == "face_more_negative" and d > 0.5
    assert vf.face_discrepancy(onehot("sadness"), onehot("joy"), 0.3)[2] == "face_more_positive"
    assert vf.face_discrepancy(onehot("neutral"), onehot("surprise", 0.6), 0.1)[2] == "face_surprised"
    assert vf.face_discrepancy(onehot("joy"), onehot("joy"), 0.3) == (pytest.approx(0.0, abs=1e-9), False, "none")
    assert vf.face_discrepancy(onehot("joy"), onehot("anger"), 0.3, present=False) == (0.0, False, "none")
    assert set(vf.FACE_DIRECTIONS) >= {vf.face_discrepancy(onehot(a), onehot(b), 0.0)[2] for a in labels.EMOTIONS for b in labels.EMOTIONS}


# ---------------------------------------------------------------- offline cache against the live ring

W, H = 640, 360


def synthetic_faces(n_frames: int, seed: int = 0) -> list[list[Face]]:
    """per frame, the Face objects of two tracks (track 1 leaves after frame 6); embeddings on even frames."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        fs = []
        for tr in (0, 1) if k <= 6 else (0,):
            box = np.array([100.0 + 3 * k + 200 * tr, 50.0, 80.0 + 10 * tr, 100.0])
            first = k == 0
            fs.append(Face(tr, box, rng.uniform(0, 300, (5, 2)), float(rng.uniform(0.6, 1.0)), rng.normal(0, 2, 7).astype(np.float32),
                           rng.normal(0, 1, 512).astype(np.float32) if k % 2 == 0 else None,
                           float("nan") if first else float(rng.uniform(0, 0.1)), float("nan") if first else float(rng.uniform(0, 0.1))))
        frames.append(fs)
    return frames


def write_video_cache(path: Path, clips: dict[str, list[list[Face]]], extra_frames: dict[str, int] | None = None) -> None:
    """cache/video_{split}.npz in the layout of tools/extract_video_features.py merge(), rows of the
    clips interleaved so the reader must sort them."""
    keys = list(clips)
    rows = []
    for ci, k in enumerate(keys):
        for fi, faces in enumerate(clips[k]):
            for f in faces:
                rows.append((ci, fi, f))
    order = np.random.default_rng(5).permutation(len(rows))
    rows = [rows[i] for i in order]
    e_row, e_vec = [], []
    for i, (_, fi, f) in enumerate(rows):
        if f.embed is not None:
            e_row.append(i)
            e_vec.append(f.embed.astype(np.float16))
    n = len(rows)
    nf = [len(clips[k]) + (extra_frames or {}).get(k, 0) for k in keys]
    np.savez(path, keys=np.array(keys), n_frames=np.array(nf, np.int32), frame_w=np.full(len(keys), 1280, np.int32),
             frame_h=np.full(len(keys), 720, np.int32),
             f_clip=np.array([r[0] for r in rows], np.int32), f_frame=np.array([r[1] for r in rows], np.int32),
             f_track=np.array([r[2].track for r in rows], np.int32),
             f_box=np.array([r[2].box / [W, H, W, H] for r in rows], np.float32).reshape(n, 4),
             f_lmk=np.array([(r[2].landmarks / [W, H]).reshape(-1) for r in rows], np.float32).reshape(n, 10),
             f_score=np.array([r[2].score for r in rows], np.float32), f_mouth=np.array([r[2].mouth_motion for r in rows], np.float32),
             f_face=np.array([r[2].face_motion for r in rows], np.float32),
             f_logits=np.array([r[2].logits for r in rows], np.float32).reshape(n, 7), f_has_emb=np.array([r[2].embed is not None for r in rows]),
             e_row=np.array(e_row, np.int64), e_vec=np.array(e_vec, np.float16).reshape(len(e_vec), 512),
             fps=np.float32(FPS), meta=np.array(json.dumps({"errors": {}})))


def test_clip_rows_reads_the_cache_layout(tmp_path):
    clips = {"3_1": synthetic_faces(12, 0), "0_0": synthetic_faces(9, 1), "5_2": [[] for _ in range(4)]}
    write_video_cache(tmp_path / "video_dev.npz", clips)
    c = vf.load_video_cache("dev", tmp_path)
    assert c["fps_value"] == FPS and set(c["index"]) == set(clips)
    r = vf.clip_rows(c, "3_1")
    faces = [(k, f) for k, fs in enumerate(clips["3_1"]) for f in fs]
    assert len(r) == len(faces) and np.allclose(r.frame_t, np.arange(12) / FPS)
    assert np.all(np.diff(r.t) >= 0)                    # sorted by frame, then track
    for i, (k, f) in enumerate(faces):
        assert np.isclose(r.t[i], k / FPS) and r.track[i] == f.track
        assert np.isclose(r.area[i], f.box[2] * f.box[3] / (W * H), rtol=1e-5)
        assert np.allclose(r.logits[i], f.logits, atol=1e-6)
        if f.embed is None:
            assert np.isnan(r.emb[i]).all()
        else:
            assert np.allclose(r.emb[i], f.embed.astype(np.float16).astype(np.float32))
    empty = vf.clip_rows(c, "5_2")
    assert len(empty) == 0 and len(empty.frame_t) == 4
    assert not vf.face_evidence(empty, None, None, "asd").present
    missing = vf.clip_rows(c, "9_9")
    assert len(missing) == 0 and len(missing.frame_t) == 0


def test_offline_rows_give_the_live_evidence(tmp_path):
    """the same faces through the live ring (pixel boxes) and through the cache (normalized boxes,
    float32 / float16 storage) give the same evidence and features for every rule."""
    frames = synthetic_faces(15, 2)
    write_video_cache(tmp_path / "video_test.npz", {"1_1": frames}, {"1_1": 3})
    off = vf.clip_rows(vf.load_video_cache("test", tmp_path), "1_1")
    ring = vf.FaceRing()
    for k, fs in enumerate(frames + [[], [], []]):           # three trailing frames without a face
        ring.add(k / FPS, fs, W, H)
    live = ring.rows()
    assert ring.last_t == pytest.approx(17 / FPS)
    et = np.arange(0, 2, vf.HOP_S) + vf.HOP_S / 2
    ev = np.random.default_rng(0).normal(size=len(et))
    for rule in vf.RULES:
        for t0, t1 in ((-np.inf, np.inf), (0.3, 1.2)):
            a, b = vf.face_evidence(off, et, ev, rule, t0, t1), vf.face_evidence(live, et, ev, rule, t0, t1)
            assert (a.track, a.n_frames, a.n_tracks, a.n_window_frames) == (b.track, b.n_frames, b.n_tracks, b.n_window_frames)
            assert np.allclose(vf.vision_features_B(a), vf.vision_features_B(b), atol=2e-3), (rule, t0)
            assert np.allclose(vf.vision_features_A(a), vf.vision_features_A(b), atol=1e-4), (rule, t0)


def test_face_ring_window_and_trim():
    ring = vf.FaceRing(max_s=1.0)
    frames = synthetic_faces(30, 3)
    for k, fs in enumerate(frames):
        ring.add(k / FPS, fs, W, H)
    rows = ring.rows()
    assert rows.frame_t.min() >= 2.9 - 1.0 - 1e-9 and np.isclose(rows.frame_t.max(), 2.9)
    w = ring.rows(2.5, 2.8)
    assert np.allclose(np.unique(w.frame_t), [2.5, 2.6, 2.7])
    assert len(vf.FaceRing().rows()) == 0 and vf.FaceRing().last_t is None


def test_face_rows_concat_and_take():
    rows, _, _ = three_tracks()
    a, b = rows.window(0, 1.0), rows.window(1.0, 5.0)
    c = vf.FaceRows.concat([a, b])
    assert len(c) == len(rows) and np.allclose(np.sort(c.frame_t), rows.frame_t)
    key = lambda r: sorted(zip(r.t.round(6), r.track, r.area))  # noqa: E731
    assert key(c) == key(rows)
    assert len(rows.take(rows.track == 2)) == 15


# ---------------------------------------------------------------- heads

def lr_head(cls, n_in: int, classes: list[str], seed: int = 0, **meta):
    rng = np.random.default_rng(seed)
    p = heads.head_params(heads.lr_params(rng.normal(size=(len(classes), n_in)), rng.normal(size=len(classes))),
                          rng.normal(size=n_in), rng.uniform(0.5, 2, n_in), classes, 1.3, **meta)
    return cls(p)


def test_vision_heads_roundtrip_and_contracts(tmp_path):
    for cls, n_in, classes in ((heads.VisionCalibrator, 16, labels.EMOTIONS), (heads.VisionCalibrator, 528, labels.EMOTIONS),
                               (heads.Fusion3, 29, labels.EMOTIONS), (heads.Sentiment3, 29, labels.SENTIMENTS)):
        h = lr_head(cls, n_in, classes, rule="asd", variant="B" if n_in == 528 else "A")
        path = tmp_path / f"{cls.__name__}_{n_in}.npz"
        h.save(path)
        g = cls.load(path)
        x = np.random.default_rng(1).normal(size=(4, n_in))
        assert np.allclose(g.predict_log_proba_batch(x), h.predict_log_proba_batch(x))
        assert g.temperature == pytest.approx(1.3) and g.classes == classes
    v = heads.VisionCalibrator.load(tmp_path / "VisionCalibrator_528.npz")
    assert v.uses_embedding and v.rule == "asd" and v.variant == "B"
    assert not heads.VisionCalibrator.load(tmp_path / "VisionCalibrator_16.npz").uses_embedding
    for cls, n_in, classes in ((heads.VisionCalibrator, 24, labels.EMOTIONS), (heads.VisionCalibrator, 16, labels.SENTIMENTS),
                               (heads.Fusion3, 19, labels.EMOTIONS), (heads.Sentiment3, 29, labels.EMOTIONS),
                               (heads.Sentiment3, 19, labels.SENTIMENTS)):
        with pytest.raises(ValueError):
            lr_head(cls, n_in, classes)
    # the two-way heads still refuse the fusion3 layout
    with pytest.raises(ValueError):
        lr_head(heads.Fusion, 29, labels.EMOTIONS)


def test_calibrated_face_uses_the_head_layout():
    rows, et, ev = three_tracks()
    rows.emb = np.ones((len(rows), 512), np.float32)
    e = vf.face_evidence(rows, et, ev, "asd")
    for n_in in (16, 528):
        h = lr_head(heads.VisionCalibrator, n_in, labels.EMOTIONS)
        x = vf.vision_features_A(e) if n_in == 16 else vf.vision_features_B(e)
        assert np.allclose(vf.calibrated_face(h, e), np.exp(h.predict_log_proba(x)))
        assert np.allclose(vf.calibrated_face(h, vf.FaceEvidence.absent()), 1 / 7)


def test_fer_order_is_mapped_to_meld():
    # a face that is clearly happy in fer order gives joy in the evidence
    z = np.full(7, -5.0)
    z[FER_CLASSES.index("happy")] = 5.0
    rows = rows_of([{"track": 0, "frames": range(4), "area": 0.02, "mouth": 0.1, "logits": z}])
    e = vf.face_evidence(rows, None, None, "largest")
    assert labels.EMOTIONS[int(np.argmax(e.p_face))] == "joy" and e.to_dict()["present"]
