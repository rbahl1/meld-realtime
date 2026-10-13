"""the commit policy of extension spec 5: meld_rt.commit (tick features, the numpy policy
the service loads), tools/train_commit_policy.py (fitted q-iteration, baselines, paired
bootstrap over dialogues) on synthetic traces whose right answer is known, and
tools/build_commit_traces.py on a synthetic workspace with stand-in experts (vision mode
with the fold heads of train_vision.py, and --no-vision)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import build_commit_traces as bct  # noqa: E402
import train_commit_policy as tcp  # noqa: E402

from meld_rt import heads  # noqa: E402
from meld_rt import vision_features as vf  # noqa: E402
from meld_rt.commit import FEATURES, CommitPolicy, entropy_bits, is_uniform, tick_features, tick_sequence  # noqa: E402
from meld_rt.discrepancy import jsd  # noqa: E402

N = 7
UNIFORM = np.full(N, 1 / N)
PRIOR = np.array([0.45, 0.12, 0.03, 0.07, 0.17, 0.03, 0.13])


def onehot_post(c: int, amp: float, rng) -> np.ndarray:
    z = rng.normal(0, 0.3, N)
    z[c] += amp
    p = np.exp(z - z.max())
    return p / p.sum()


# ---------------------------------------------------------------- synthetic traces with a known answer

def synth_traces(path: Path, n_dialogues: int, seed: int, mode: str = "vision") -> dict:
    """traces in the build_commit_traces.py layout. each clip's evidence turns clear at a time tau:
    from then on the fused posterior is sharp on the gold class with a confident transcript. before it,
    a tick is soft on a random class, or (one in three) a decoy: sharp on a wrong class with an asr
    confidence below 0.45. the right policy emits at the first clear tick; a threshold on the max
    probability also emits on decoys, a policy that reads the asr confidence does not."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dialogues):
        for u in range(int(rng.integers(3, 9))):
            y = int(rng.choice(N, p=PRIOR))
            dur = float(rng.uniform(0.6, 7.0))
            tau = float(rng.uniform(0.0, 6.0))
            nt = len([t for t in bct.TICKS_S if t < dur + 0.512])   # the builder's rule: the vad segment is still open
            rows.append((d, u, y, dur, tau, nt))
    n = len(rows)
    S = bct.N_SLOTS
    out = {f: np.full((n, S, N), np.nan, np.float32) for f in bct.POST_FIELDS}
    t = np.full((n, S), np.nan)
    asr_conf = np.full((n, S), np.nan, np.float32)
    n_words = np.zeros((n, S), np.int64)
    face_present = np.zeros((n, S), bool)
    presence = np.zeros((n, S), np.float32)
    text_ok = np.zeros((n, S), bool)
    for i, (d, u, y, dur, tau, nt) in enumerate(rows):
        t_end = dur + 0.664
        for j, tt in enumerate([*bct.TICKS_S[:nt], None]):
            slot = j if tt is not None else bct.END
            time_s = tt if tt is not None else t_end
            clear = time_s >= tau or (tt is None and tau <= dur)
            if clear:
                p, conf = onehot_post(y, 3.5, rng), rng.uniform(0.6, 1.0)
            elif rng.uniform() < 1 / 3:
                p, conf = onehot_post(int((y + rng.integers(1, N)) % N), 3.5, rng), rng.uniform(0.3, 0.45)
            else:
                p, conf = onehot_post(int(rng.integers(0, N)), 0.4, rng), rng.uniform(0.3, 1.0)
            out["fused"][i, slot] = p
            out["p_text"][i, slot] = onehot_post(int(np.argmax(p)), 1.0, rng)
            out["p_audio"][i, slot] = onehot_post(int(rng.integers(0, N)), 0.5, rng)
            fp = mode == "vision" and (i % 3 != 0)
            out["p_face"][i, slot] = onehot_post(int(np.argmax(p)), 0.8, rng) if fp else UNIFORM
            face_present[i, slot], presence[i, slot] = fp, 0.9 if fp else 0.0
            t[i, slot], asr_conf[i, slot], n_words[i, slot], text_ok[i, slot] = time_s, conf, int(2 * time_s), True
    keys = np.array([f"{d}_{u}" for d, u, *_ in rows])
    arrays = {"keys": keys, "dialogue": np.array([r[0] for r in rows], np.int64), "utterance": np.array([r[1] for r in rows], np.int64),
              "gold": np.array([r[2] for r in rows], np.int64), "fold": np.full(n, -1, np.int64),
              "duration_s": np.array([r[3] for r in rows]), "t_end": np.array([r[3] + 0.664 for r in rows]),
              "n_ticks": np.array([r[5] for r in rows], np.int64), "t": t, **out, "asr_conf": asr_conf, "presence": presence,
              "n_words": n_words, "face_present": face_present, "text_ok": text_ok, "from_cache": np.zeros((n, S), bool),
              "asr_text": np.full((n, S), "", dtype="<U4"), "fused_end_ungated": out["fused"][:, bct.END].copy(),
              "end_degraded": np.zeros(n, bool),
              "meta": np.array(json.dumps({"mode": mode, "split": "synthetic", "endpoint": {"final_p50_s": 0.152, "vad_tail_s": 0.512}}))}
    np.savez(path, **arrays)
    return arrays


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("commit")
    traces = root / "traces"
    traces.mkdir()
    for split, n, seed in (("train", 260, 1), ("dev", 90, 2), ("test", 120, 3)):
        synth_traces(traces / f"commit_traces_{split}.npz", n, seed)
    out_heads, out_res = root / "heads", root / "results"
    assert tcp.main(["--traces", str(traces), "--heads", str(out_heads), "--results", str(out_res), "--costs", "0.05,0.2"]) == 0
    return {"root": root, "traces": traces, "heads": out_heads, "res": json.loads((out_res / "commit_policy.json").read_text(encoding="utf-8")),
            "md": (out_res / "commit_policy.md").read_text(encoding="utf-8")}


def test_fqi_beats_or_matches_the_baselines_where_the_answer_is_known(synth):
    res = synth["res"]
    for split in ("dev", "test"):
        for c in ("0.05", "0.2"):
            m = res[split][c]["methods"]
            best_base = max(m[b]["mean_reward"] for b in ("endpoint", "fixed", "threshold"))
            assert m["fqi"]["mean_reward"] >= best_base - 0.01, (split, c, m)
    # at the deployed cost the decoys make the threshold commit wrong labels; fqi reads the asr
    # confidence and beats every baseline with a paired interval above zero
    pf = res["test"]["0.05"]["paired_fqi_minus"]
    assert all(pf[b]["above_zero"] for b in ("endpoint", "fixed", "threshold")), pf
    assert res["test"]["0.05"]["rl_adds_value_over"] == {b: True for b in ("endpoint", "fixed", "threshold")}
    m = res["test"]["0.05"]["methods"]
    assert m["fqi"]["early_share"] > 0.3 and m["fqi"]["mean_commit_s"] < m["endpoint"]["mean_commit_s"]
    # the known answer: emit at the first clear tick (confident transcript and a sharp posterior)
    ep = tcp.episodes(bct.load_traces(synth["traces"] / "commit_traces_test.npz"))
    oracle = tcp.rollout(ep, (np.nan_to_num(ep.maxp) > 0.6) & (np.nan_to_num(ep.X[:, :, FEATURES.index("asr_conf")]) >= 0.55), 0.05)
    assert m["fqi"]["mean_reward"] >= oracle["reward"].mean() - 0.02
    assert res["train"]["p_correct"]["weights"]["asr_conf"] > 0


def test_choices_are_made_on_dev_and_reported(synth):
    res = synth["res"]
    for c in ("0.05", "0.2"):
        rc = res["train"]["costs"][c]
        grid = rc["fqi"]["grid"]
        assert set(grid) == {f"{k},alpha={a:g}" for k, a in tcp.candidates()}
        best = max(v["dev_reward"] for v in grid.values())
        assert rc["fqi"]["dev_reward"] == best
        assert res["dev"][c]["methods"]["fqi"]["mean_reward"] == pytest.approx(best, abs=1e-6)
        for b, params in (("fixed", tcp.FIXED_T), ("threshold", tcp.TAUS)):
            g = rc["baselines"][b]["dev_grid"]
            assert set(g) == {str(p) for p in params}
            assert g[str(rc["baselines"][b]["param"])] == max(g.values())
            assert res["dev"][c]["methods"][b]["mean_reward"] == pytest.approx(max(g.values()), abs=1e-6)
        for v in grid.values():
            assert 1 <= v["n_iterations"] <= tcp.MAX_ITER
    assert "## Choices made on dev" in synth["md"] and "## Test: fqi minus each baseline" in synth["md"]


def test_exported_policy_loads_with_numpy_only_and_matches_training(synth):
    path = synth["heads"] / "commit_policy.npz"
    pol = CommitPolicy.load(path)
    assert pol.c == 0.05 and pol.feature_names == list(FEATURES) and pol.summary["c"] == 0.05
    ep = tcp.episodes(bct.load_traces(synth["traces"] / "commit_traces_dev.npz"))
    ro = tcp.rollout(ep, tcp.policy_emit(pol, ep), 0.05)
    assert ro["reward"].mean() == pytest.approx(synth["res"]["dev"]["0.05"]["methods"]["fqi"]["mean_reward"], abs=1e-6)
    tick = {"t_s": 1.024, "k": 1, "fused": np.array([0.1, 0.05, 0.05, 0.1, 0.6, 0.05, 0.05]), "prev_fused": None,
            "same_label_run": 1, "p_text": UNIFORM, "p_audio": np.array([0.2, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1]), "p_face": UNIFORM,
            "face_present": False, "asr_conf": 0.8, "n_words": 3}
    code = ("import sys, json, numpy as np; from meld_rt.commit import CommitPolicy; "
            f"p = CommitPolicy.load({str(path)!r}); t = json.loads(sys.argv[1]); "
            "t = {k: (np.array(v) if isinstance(v, list) else v) for k, v in t.items()}; "
            "e, qe, qw = p.decide(p.features(t)); "
            "bad = [m for m in ('sklearn', 'scipy', 'torch', 'onnxruntime', 'cv2', 'pandas') if m in sys.modules]; "
            "print(json.dumps({'emit': e, 'qe': qe, 'qw': qw, 'bad': bad}))")
    arg = json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tick.items()})
    r = subprocess.run([sys.executable, "-c", code, arg], cwd=ROOT, capture_output=True, text=True, check=True)
    got = json.loads(r.stdout)
    assert got["bad"] == []
    e, qe, qw = pol.decide(pol.features(tick))
    assert (got["emit"], got["qe"], got["qw"]) == (e, pytest.approx(qe), pytest.approx(qw))


# ---------------------------------------------------------------- features and export

def test_tick_features_follow_the_spec():
    f = np.array([0.6, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05])
    prev = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05])
    tick = {"t_s": 2.048, "k": 2, "fused": f, "prev_fused": prev, "same_label_run": 2,
            "p_text": np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]), "p_audio": np.array([0.1, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05]),
            "p_face": np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]), "face_present": True, "asr_conf": 0.7, "n_words": 5}
    x = dict(zip(FEATURES, tick_features(tick)))
    assert x["max_p"] == pytest.approx(0.6) and x["margin"] == pytest.approx(0.5)
    assert x["entropy_bits"] == pytest.approx(entropy_bits(f)) and x["entropy_bits"] == pytest.approx(-(f * np.log2(f)).sum())
    assert x["t_s"] == 2.048 and x["k"] == 2 and x["same_label_run"] == 2
    assert x["stability"] == pytest.approx(1 - jsd(prev, f))
    assert (x["max_p_text"], x["max_p_audio"], x["max_p_face"]) == (pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.4))
    assert x["face_present"] == 1 and x["agreement"] == 2 and x["n_words_div10"] == 0.5 and x["asr_conf"] == 0.7
    # first tick: stability 1; a uniform text posterior and an absent face do not vote
    x = dict(zip(FEATURES, tick_features({**tick, "prev_fused": None, "p_text": UNIFORM, "face_present": False})))
    assert x["stability"] == 1.0 and x["agreement"] == 0 and x["face_present"] == 0
    assert is_uniform(UNIFORM) and not is_uniform(f)


def test_tick_sequence_keeps_the_service_bookkeeping():
    fused = [onehot_post(c, 3, np.random.default_rng(i)) for i, c in enumerate([4, 4, 0, 0])]
    seq = tick_sequence([1.0, 2.0, 3.0, 4.0], fused, [UNIFORM] * 4, [UNIFORM] * 4, [UNIFORM] * 4, [False] * 4, [0.5] * 4, [1, 2, 3, 4])
    assert [s["same_label_run"] for s in seq] == [1, 2, 1, 2]
    assert [s["k"] for s in seq] == [1, 2, 3, 4] and seq[0]["prev_fused"] is None
    assert np.allclose(seq[2]["prev_fused"], seq[1]["fused"])
    # the service's k for live times (segment start to the provisional's end)
    assert [s["k"] for s in tick_sequence([1.024, 2.048], fused[:2], [UNIFORM] * 2, [UNIFORM] * 2, [UNIFORM] * 2, [False] * 2,
                                          [0.5] * 2, [1, 2])] == [1, 2]


@pytest.mark.parametrize("kind", ["ridge", "mlp"])
def test_q_wait_export_matches_sklearn(kind, tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, len(FEATURES)))
    y = np.tanh(X[:, 0]) + 0.3 * X[:, 3] + rng.normal(0, 0.1, 400)
    q = tcp.QWait(kind, 0.01, 0)
    q.fit(X, y)
    pol = CommitPolicy.build(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), rng.normal(0, 1, len(FEATURES)), 0.1, 0.05, q.export(), "t")
    assert np.allclose(pol.q_wait(X), q.model.predict(X), atol=1e-8)
    pol.save(tmp_path / "p.npz")
    back = CommitPolicy.load(tmp_path / "p.npz")
    assert np.allclose(back.q_wait(X), pol.q_wait(X)) and np.allclose(back.q_emit(X), pol.q_emit(X))
    assert back.qw_kind == kind


def test_policy_rejects_other_features():
    pol = CommitPolicy.build(np.zeros(14), np.ones(14), np.zeros(14), 0.0, 0.05, {"kind": "ridge", "w": np.zeros(14), "b": 0.0}, "x")
    p = dict(pol.params)
    p["feature_names"] = np.array(list(FEATURES[:-1]) + ["other"])
    with pytest.raises(ValueError):
        CommitPolicy(p)


# ---------------------------------------------------------------- bootstrap and rollouts

def test_paired_bootstrap_resamples_whole_dialogues():
    # every dialogue has a mean difference of zero: resampling dialogues always gives 0, resampling clips would not
    groups = np.repeat(np.arange(30), 2)
    a = np.tile([1.0, 0.0], 30)
    b = np.tile([0.0, 1.0], 30)
    r = tcp.paired_bootstrap_reward(a, b, groups)
    assert r["n_groups"] == 30 and r["n"] == 60 and r["ci95"] == [0.0, 0.0] and not r["excludes_zero"]
    per_clip = tcp.paired_bootstrap_reward(a, b, np.arange(60))
    assert per_clip["ci95"][0] < -0.1 and per_clip["ci95"][1] > 0.1
    # same draws as metrics.bootstrap_by_group: the dialogue picks of resample r are rng.integers(0, G, G) in order
    rng = np.random.default_rng(0)
    d = np.random.default_rng(5).normal(0.1, 1, 60)
    g = np.repeat(np.arange(20), 3)
    draws = []
    for _ in range(1000):
        pick = rng.integers(0, 20, size=20)
        draws.append(np.concatenate([d[g == k] for k in pick]).mean())
    r = tcp.paired_bootstrap_reward(d, np.zeros(60), g)
    assert r["ci95"] == [round(float(np.percentile(draws, 2.5)), 6), round(float(np.percentile(draws, 97.5)), 6)]
    assert r["diff"] == round(float(d.mean()), 6)


def test_rollout_and_baselines(tmp_path):
    synth_traces(tmp_path / "t.npz", 20, 7)
    ep = tcp.episodes(bct.load_traces(tmp_path / "t.npz"))
    end = tcp.rollout(ep, tcp.baseline_emit(ep, "endpoint"), 0.1)
    assert (end["slot"] == bct.END).all() and np.allclose(end["t"], ep.t[:, bct.END])
    fx = tcp.rollout(ep, tcp.baseline_emit(ep, "fixed", 2), 0.1)
    assert ((fx["slot"] == 1) == (ep.n_ticks >= 2)).all() and ((fx["slot"] == bct.END) == (ep.n_ticks < 2)).all()
    th = tcp.rollout(ep, tcp.baseline_emit(ep, "threshold", 0.5), 0.1)
    for i in range(ep.n):
        hit = [j for j in range(ep.n_ticks[i]) if ep.maxp[i, j] >= 0.5]
        assert th["slot"][i] == (hit[0] if hit else bct.END)
    assert np.allclose(th["reward"], (th["label"] == ep.gold) - 0.1 * th["t"])


# ---------------------------------------------------------------- trace builder on a synthetic workspace

class FakeSV:
    """sensevoice stand-in: a transcript and posteriors from the prefix length, deterministic."""

    def __init__(self):
        self.calls = []

    def run(self, pcm):
        n = len(pcm)
        self.calls.append(n)
        rng = np.random.default_rng(n)
        emo = rng.dirichlet(np.ones(9)).astype(np.float32)
        ev = rng.dirichlet(np.ones(11)).astype(np.float32)
        words = ["so", "what", "did", "you", "do"][: max(0, int(n / 16000 * 2) - 1)]
        return type("R", (), {"text": " ".join(words), "asr_conf": 0.2 if n < 20000 else 0.8, "emo_probs": emo, "event_probs": ev,
                              "enc_pooled": rng.normal(0, 1, 1024).astype(np.float32), "tokens_per_s": len(words) / (n / 16000)})()


class FakeText:
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return np.random.default_rng(abs(hash(text)) % (2 ** 32)).normal(0, 1, 768).astype(np.float32)


@pytest.fixture(scope="module")
def tav(tmp_path_factory):
    """the synthetic workspace of tests/test_train_vision.py (two-way heads, vision heads, fold heads), with
    every wav rewritten to the duration its audio cache row gives."""
    import train_heads
    import train_vision
    from test_train_pipeline import subset, synth_caches, write_csv
    from test_train_vision import synth_video

    root = tmp_path_factory.mktemp("tav")
    meld, cache, out, audio = root / "meld", root / "cache", root / "heads", root / "audio"
    for d in (meld, cache, out, audio):
        d.mkdir()
    rng = np.random.default_rng(0)
    centers_t, centers_e = 2.0 * rng.standard_normal((7, 768)), 2.0 * rng.standard_normal((7, 1024))
    for split, n in (("train", 40), ("dev", 12), ("test", 15)):
        utts = subset(split, n)
        write_csv(utts, meld / f"{split}_sent_emo.csv")
        synth_caches(utts, split, cache, rng, centers_t, centers_e)
        synth_video(utts, split, cache, audio, {"train": 1, "dev": 2, "test": 3}[split])
    cfg = yaml.safe_load((ROOT / "configs" / "heads.yaml").read_text(encoding="utf-8"))
    cfg["text_ctx"]["grid"] = {"k": [0], "class_weight": ["none"], "learning_rate_init": [0.001]}
    cfg["text_ctx"]["mlp"].update({"max_epochs": 4, "patience": 2})
    cfg["lr"]["C"] = [1]
    cfg["fusion"]["cross_fit_folds"] = 3
    (root / "heads.yaml").write_text(yaml.safe_dump(cfg))
    base = ["--cache", str(cache), "--config", str(root / "heads.yaml"), "--meld", str(meld)]
    assert train_heads.main([*base, "--out", str(out)]) == 0
    assert train_vision.main([*base, "--heads", str(out), "--audio", str(audio), "--post-splits", "dev"]) == 0
    for split in ("train", "dev", "test"):
        z = np.load(cache / f"audio_{split}.npz")
        for k, dur in zip(z["keys"], z["duration_s"]):
            d, u = str(k).split("_")
            t = np.arange(round(float(dur) * 16000)) / 16000
            x = (0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
            sf.write(audio / split / f"dia{d}_utt{u}.wav", x, 16000, subtype="PCM_16")
    return {"root": root, "meld": meld, "cache": cache, "heads": out, "audio": audio}


def run_builder(tav, out_dir, *extra):
    sv, te = FakeSV(), FakeText()
    argv = ["--cache", str(tav["cache"]), "--heads", str(tav["heads"]), "--audio", str(tav["audio"]), "--meld", str(tav["meld"]),
            "--out-dir", str(out_dir), *extra]
    assert bct.main(argv, experts=(sv, te)) == 0
    return sv, te


def _lg(p) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, np.float64), 1e-8, 1))


def _cached(z: dict, r: int, sfx: str) -> types.SimpleNamespace:
    """the sensevoice values of audio cache row r (sfx '' for the full clip, '_p2' for the 2 s prefix)."""
    return types.SimpleNamespace(text=str(z["asr_text" + sfx][r]), asr_conf=float(z["asr_conf" + sfx][r]), emo_probs=z["emo_probs" + sfx][r],
                                 event_probs=z["event_probs" + sfx][r], tokens_per_s=float(z["tokens_per_s" + sfx][r]),
                                 enc_pooled=z["enc_pooled" + sfx][r], prosody=z["prosody" + sfx][r])


def expected_slot(hs: list, res, prosody, text_ctx: tuple[int, int], face: tuple, pcm_env: np.ndarray, t1: float | None,
                  gated: bool | None = None) -> dict:
    """one slot recomputed by hand: the text head on the transcript (k = 0: the turn flags only; uniform under the
    asr gate), the audio calibrator, the face over [0, t1) with the envelope of pcm_env cut at t1 (the whole clip
    when t1 is None), fusion3; the endpoint publishes uniform when its transcript fails the gate."""
    th, au, vc, f3 = hs
    turn, k = text_ctx
    ok = bool(res.text.strip()) and res.asr_conf >= 0.3
    p_text = np.exp(th.predict_log_proba(heads.TextContextHead.features(FakeText().encode(res.text), [], [], turn, k))) if ok else UNIFORM
    enc = np.asarray(res.enc_pooled) if au.uses_encoder_states else None
    p_audio = np.exp(au.predict_log_proba(heads.AudioCalibrator.features(res.emo_probs, res.event_probs, prosody, enc)))
    rows, rule = face
    et, ev = vf.audio_envelope(pcm_env)
    e = vf.face_evidence(rows, et, ev, rule) if t1 is None else vf.face_evidence(rows, et[et < t1], ev[et < t1], rule, 0.0, t1)
    p_face = vf.calibrated_face(vc, e)
    x3 = vf.build_fusion3_features(_lg(p_text), _lg(p_audio), _lg(p_face), res.asr_conf, res.event_probs[heads.LAUGHTER_IDX],
                                   res.emo_probs[heads.UNKNOWN_IDX], e.present, e.presence if e.present else 0.0)
    fused = np.exp(f3.predict_log_proba(x3))
    gated = t1 is None if gated is None else gated     # only the endpoint's final is gated
    return {"ok": ok, "p_text": p_text, "p_audio": p_audio, "p_face": p_face, "present": bool(e.present),
            "fused": fused if (ok or not gated) else UNIFORM}


def assert_slot(d: dict, i: int, j: int, exp: dict) -> None:
    assert d["text_ok"][i, j] == exp["ok"] and d["face_present"][i, j] == exp["present"]
    for f in ("p_text", "p_audio", "p_face", "fused"):
        assert np.allclose(d[f][i, j], exp[f], atol=1e-5), (i, j, f)


def test_traces_vision_mode_with_fold_heads(tav, tmp_path):
    from meld_rt.data import dialogues, load_split

    sv, _ = run_builder(tav, tmp_path)
    cache, hd = tav["cache"], tav["heads"]
    manifest = json.loads((hd / "manifest.json").read_text(encoding="utf-8"))
    rule = manifest["vision"]["rule"]
    fold_of = json.loads((cache / "folds" / "fold_of_train.json").read_text(encoding="utf-8"))
    timing = bct.endpoint_timing(bct.BENCH)
    assert timing["vad_tail_s"] == 0.512 and timing["final_p50_s"] == 0.152
    seen = {"tail": 0, "cached_ok": 0, "end_ok": 0}
    for split in ("train", "dev", "test"):
        d = bct.load_traces(tmp_path / f"commit_traces_{split}.npz")
        m = d["meta"]
        assert m["mode"] == "vision" and m["rule"] == rule and m["clips"] == len(d["keys"]) > 0
        z = {k: v for k, v in np.load(cache / f"audio_{split}.npz").items()}
        row_of = {str(k): i for i, k in enumerate(z["keys"])}
        video = vf.load_video_cache(split, cache)
        turn_of = {f"{u.dialogue_id}_{u.utterance_id}": n for g in dialogues(load_split(split, tav["meld"])) for n, u in enumerate(g)}
        for i, k in enumerate(d["keys"]):
            k = str(k)
            r = row_of[k]
            dur = float(z["duration_s"][r])
            nt = int(d["n_ticks"][i])
            # a tick while the vad segment is open: t < duration + the 512 ms endpoint silence
            assert nt == len([t for t in bct.TICKS_S if t < dur + 0.512 - 1e-9])
            assert d["duration_s"][i] == pytest.approx(dur, abs=1e-3)
            assert d["t_end"][i] == pytest.approx(dur + 0.512 + 0.152, abs=1e-3)
            assert np.allclose(d["t"][i, :nt], bct.TICKS_S[:nt]) and np.isnan(d["t"][i, nt:bct.END]).all()
            assert np.isnan(d["fused"][i, nt:bct.END]).all() and np.allclose(d["fused"][i, :nt].sum(axis=1), 1, atol=1e-5)
            for j in range(nt):
                t = bct.TICKS_S[j]
                assert d["in_tail"][i, j] == (t > dur + 1e-9)
                # 2 s and 4 s prefixes inside the clip come from the cache (cut at exactly that prefix); 1 s, 3 s and
                # every tick past the clip's end from sensevoice
                assert d["from_cache"][i, j] == (t in (2.0, 4.0) and dur >= t - 1e-9)
            if split == "train":
                assert d["fold"][i] == fold_of[k]
        def hs_of(k, split=split):
            """the heads the builder must have used: the clip's fold heads on train, the deployed ones elsewhere."""
            if split == "train":
                pre = cache / "folds" / f"fold{fold_of[k]}_"
                return [heads.TextContextHead.load(f"{pre}text_ctx.npz"), heads.AudioCalibrator.load(f"{pre}audio_cal.npz"),
                        heads.VisionCalibrator.load(f"{pre}vision_cal.npz"), heads.Fusion3.load(f"{pre}fusion3.npz")]
            return [heads.TextContextHead.load(hd / "text_ctx.npz"), heads.AudioCalibrator.load(hd / "audio_cal.npz"),
                    heads.VisionCalibrator.load(hd / "vision_cal.npz"), heads.Fusion3.load(hd / "fusion3.npz")]

        def pcm_of(k, split=split):
            x, _ = sf.read(str(tav["audio"] / split / f"dia{k.split('_')[0]}_utt{k.split('_')[1]}.wav"), dtype="int16")
            return x.astype(np.float32) / 32768

        def ctx(k, hs, turn_of=turn_of):
            assert hs[0].k == 0    # the fixture's text head: no previous utterances, the turn flags only
            return turn_of[k], hs[0].k

        # 1) the 1 s tick of the first clip with a tick (fakesv: conf 0.2 on 16000 samples, the text gated to uniform)
        i = int(np.flatnonzero(d["n_ticks"] >= 1)[0])
        k = str(d["keys"][i])
        pcm, hs = pcm_of(k), hs_of(k)
        res = FakeSV().run(pcm[:16000])
        assert d["asr_text"][i, 0] == res.text
        assert_slot(d, i, 0, expected_slot(hs, res, [0, 0, 0, res.tokens_per_s], ctx(k, hs), (vf.clip_rows(video, k), rule), pcm, 1.0))
        assert not d["text_ok"][i, 0]
        # 2) a 2 s tick past the clip's end (in the vad tail): sensevoice on the clip zero-padded to 2 s, so
        # fakesv's three words at conf 0.8 pass the gate; the face over [0, 2 s) with the padded envelope
        tails = np.flatnonzero(d["in_tail"][:, 1] & (d["n_ticks"] >= 2))
        assert len(tails)
        i = int(tails[0])
        k = str(d["keys"][i])
        pcm, hs = pcm_of(k), hs_of(k)
        assert len(pcm) < 32000
        padded = np.concatenate([pcm, np.zeros(32000 - len(pcm), np.float32)])
        res = FakeSV().run(padded)
        assert d["asr_text"][i, 1] == res.text == "so what did" and d["asr_conf"][i, 1] == pytest.approx(0.8)
        exp = expected_slot(hs, res, [0, 0, 0, res.tokens_per_s], ctx(k, hs), (vf.clip_rows(video, k), rule), padded, 2.0)
        assert exp["ok"] and not np.allclose(exp["p_text"], UNIFORM)
        assert_slot(d, i, 1, exp)
        seen["tail"] += 1
        # 3) a cached 2 s tick whose transcript passes the gate (the cache's _p2 values, the clip's own text context)
        cached = [j for j in np.flatnonzero(d["from_cache"][:, 1] & d["text_ok"][:, 1])]
        assert cached
        i = int(cached[0])
        k = str(d["keys"][i])
        pcm, hs, res = pcm_of(k), hs_of(k), _cached(z, row_of[k], "_p2")
        assert_slot(d, i, 1, expected_slot(hs, res, [0, 0, 0, res.tokens_per_s], ctx(k, hs), (vf.clip_rows(video, k), rule), pcm, 2.0))
        seen["cached_ok"] += 1
        # 4) the endpoint of a clip whose full transcript passes the gate: cached full-clip values with the prosody,
        # the face over the whole clip; and one that fails the gate publishes uniform
        for i in (int(np.flatnonzero(~d["end_degraded"])[0]), int(np.flatnonzero(d["end_degraded"])[0])):
            k = str(d["keys"][i])
            pcm, hs, res = pcm_of(k), hs_of(k), _cached(z, row_of[k], "")
            exp = expected_slot(hs, res, res.prosody, ctx(k, hs), (vf.clip_rows(video, k), rule), pcm, None)
            assert d["end_degraded"][i] == (not exp["ok"])
            assert_slot(d, i, bct.END, exp)
            ungated = expected_slot(hs, res, res.prosody, ctx(k, hs), (vf.clip_rows(video, k), rule), pcm, None, gated=False)
            assert np.allclose(d["fused_end_ungated"][i], ungated["fused"], atol=1e-5)
            seen["end_ok"] += exp["ok"]
        assert all(np.allclose(d["fused"][j, bct.END], UNIFORM) for j in np.flatnonzero(d["end_degraded"]))
        if split == "test":
            assert "fused_accuracy_by_slot" not in m       # no test label is read for a number
        else:
            assert "end" in m["fused_accuracy_by_slot"]
        assert m["tail_ticks"] == int(d["in_tail"].sum())
    assert seen == {"tail": 3, "cached_ok": 3, "end_ok": 3}
    assert sv.calls and all(n in (16000, 32000, 48000) for n in sv.calls) and 32000 in sv.calls


def test_traces_no_vision_mode_and_limit(tav, tmp_path):
    run_builder(tav, tmp_path, "--no-vision", "--limit", "5", "--splits", "dev")
    d = bct.load_traces(tmp_path / "commit_traces_dev.npz")
    assert d["meta"]["mode"] == "no_vision" and len(d["keys"]) == 5
    assert not d["face_present"].any()
    valid = np.isfinite(d["p_face"][..., 0])
    assert np.allclose(d["p_face"][valid], 1 / 7)
    # two-way fusion of the deployed heads at the endpoint
    z = np.load(tav["cache"] / "audio_dev.npz")
    fh, au = heads.Fusion.load(tav["heads"] / "fusion.npz"), heads.AudioCalibrator.load(tav["heads"] / "audio_cal.npz")
    i = int(np.flatnonzero(~d["end_degraded"])[0])
    r = list(map(str, z["keys"])).index(str(d["keys"][i]))
    enc = z["enc_pooled"][r] if au.uses_encoder_states else None
    pa = np.exp(au.predict_log_proba(heads.AudioCalibrator.features(z["emo_probs"][r], z["event_probs"][r], z["prosody"][r], enc)))
    assert np.allclose(d["p_audio"][i, bct.END], pa, atol=1e-6)
    lg = lambda p: np.log(np.clip(p, 1e-8, 1))  # noqa: E731
    x = heads.build_fusion_features(lg(d["p_text"][i, bct.END].astype(np.float64)), lg(pa), z["asr_conf"][r],
                                    z["event_probs"][r][heads.LAUGHTER_IDX], z["emo_probs"][r][heads.UNKNOWN_IDX])
    assert np.allclose(d["fused"][i, bct.END], np.exp(fh.predict_log_proba(x)), atol=1e-5)


def test_builder_output_trains_a_policy(tav, tmp_path):
    run_builder(tav, tmp_path)
    assert tcp.main(["--traces", str(tmp_path), "--heads", str(tmp_path / "h"), "--results", str(tmp_path / "r"), "--costs", "0.05",
                     "--skip-test"]) == 0
    res = json.loads((tmp_path / "r" / "commit_policy.json").read_text(encoding="utf-8"))
    assert "test" not in res and res["inputs"]["train"]["mode"] == "vision"
    pol = CommitPolicy.load(tmp_path / "h" / "commit_policy.npz")
    assert pol.summary["test"] is None and pol.c == 0.05


def test_builder_resumes_from_a_checkpoint(tav, tmp_path, monkeypatch):
    monkeypatch.setattr(bct, "CHECKPOINT_EVERY", 4)
    calls = {"n": 0}
    real = bct.clip_trace

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 7:
            raise RuntimeError("stop")
        return real(*a, **k)

    monkeypatch.setattr(bct, "clip_trace", flaky)
    with pytest.raises(RuntimeError, match="stop"):
        run_builder(tav, tmp_path, "--splits", "dev")
    assert bct.partial_path(tmp_path / "commit_traces_dev.npz").exists()
    monkeypatch.setattr(bct, "clip_trace", real)
    run_builder(tav, tmp_path / "ref", "--splits", "dev")
    run_builder(tav, tmp_path, "--splits", "dev")
    a, b = bct.load_traces(tmp_path / "commit_traces_dev.npz"), bct.load_traces(tmp_path / "ref" / "commit_traces_dev.npz")
    assert a["meta"]["clips_this_run"] == b["meta"]["clips"] - 6   # the six clips before the stop were saved
    for f in ("keys", "fused", "p_text", "p_audio", "p_face", "t", "n_ticks", "asr_text"):
        assert np.array_equal(a[f], b[f], equal_nan=a[f].dtype.kind == "f"), f
    assert not bct.partial_path(tmp_path / "commit_traces_dev.npz").exists()


def test_resume_refuses_another_video_cache(tav, tmp_path, monkeypatch):
    """the checkpoint's config covers cache/video_{split}.npz: a resume after the video cache changed stops."""
    cache = tmp_path / "cache"
    shutil.copytree(tav["cache"], cache)
    ws = {**tav, "cache": cache}
    monkeypatch.setattr(bct, "CHECKPOINT_EVERY", 4)
    real, calls = bct.clip_trace, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 6:
            raise RuntimeError("stop")
        return real(*a, **k)

    monkeypatch.setattr(bct, "clip_trace", flaky)
    with pytest.raises(RuntimeError, match="stop"):
        run_builder(ws, tmp_path / "out", "--splits", "dev")
    monkeypatch.setattr(bct, "clip_trace", real)
    with np.load(cache / "video_dev.npz") as z:
        v = {k: z[k] for k in z.files}
    v["f_score"] = (v["f_score"] * 0.5).astype(v["f_score"].dtype)   # a re-extracted video cache, same heads
    np.savez(cache / "video_dev.npz", **v)
    with pytest.raises(SystemExit, match="other settings"):
        run_builder(ws, tmp_path / "out", "--splits", "dev")


def test_limited_traces_never_reach_the_repo(tav, tmp_path, monkeypatch):
    """traces built with --limit stay out of the repository's cache/, heads/ and results/ (ROOT points at a scratch copy)."""
    fake_root = tmp_path / "repo"
    for sub in ("cache", "heads", "results"):
        (fake_root / sub).mkdir(parents=True)
    monkeypatch.setattr(bct, "ROOT", fake_root)
    monkeypatch.setattr(tcp, "ROOT", fake_root)
    # the builder: --limit into the repo's cache/ (its default --out-dir) stops before any clip
    with pytest.raises(SystemExit, match="--limit"):
        run_builder(tav, fake_root / "cache", "--limit", "3", "--splits", "dev")
    assert not list((fake_root / "cache").iterdir())
    # the trainer: vision-mode traces with a limit are not exported to heads/ or results/, elsewhere they are
    traces = tmp_path / "traces"
    traces.mkdir()
    for split, n, seed in (("train", 20, 1), ("dev", 10, 2)):
        a = synth_traces(traces / f"commit_traces_{split}.npz", n, seed)
        a["meta"] = np.array(json.dumps({**json.loads(str(a["meta"])), "limit": 60}))
        np.savez(traces / f"commit_traces_{split}.npz", **a)
    with pytest.raises(SystemExit, match="--limit"):
        tcp.main(["--traces", str(traces), "--skip-test", "--costs", "0.05"])
    assert not list((fake_root / "heads").iterdir()) and not list((fake_root / "results").iterdir())
    assert tcp.main(["--traces", str(traces), "--heads", str(tmp_path / "h"), "--results", str(tmp_path / "r"), "--skip-test",
                     "--costs", "0.05"]) == 0


def test_offline_ticks_are_the_services_full_window_ticks(tmp_path):
    """meld_rt.service (package B) end-pointing and provisionals on a clip whose speech fills [0, duration): the
    real VadSegmenter with its speech probability stubbed (speech frames 0.9, silence 0), Service._on_audio and
    _maybe_provisional. the full-window provisionals are the commit ticks; the builder must give the same number
    per clip (tick_times) at about the same times (TICKS_S): no tick the service does not make, none it skips."""
    pytest.importorskip("cv2")
    from test_service_vision import bare_service

    from meld_rt import protocol as P
    from meld_rt.service import ServiceConfig

    if not (ServiceConfig().models / ServiceConfig().vad).exists():
        pytest.skip("models not fetched")
    tail_s = bct.endpoint_timing(bct.BENCH)["vad_tail_s"]
    for dur in (0.3, 0.45, 0.6, 1.2, 1.6, 2.2, 2.6, 3.3, 4.5, 6.0):
        svc = bare_service(tmp_path / f"d{dur}")
        st = svc._stream("s")
        n_speech = round(dur * P.SR)
        pushed = {"n": 0}

        def prob(frame, pushed=pushed, n_speech=n_speech):
            i = pushed["n"]
            pushed["n"] += 1
            return 0.9 if i * P.FRAME_SAMPLES < n_speech else 0.0

        st.vad._prob = prob
        jobs = []
        for seq in range(int((dur + 2.0) / P.FRAME_S)):
            svc._on_audio(st, {"n_samples": P.FRAME_SAMPLES, "seq": seq, "t_capture_ns": 0}, np.zeros(P.FRAME_SAMPLES, np.int16).tobytes(), 0)
            while (j := svc._jobs.get(0)) is not None:
                jobs.append(j)
        finals = [j for j in jobs if j.kind == "final"]
        assert len(finals) == 1 and finals[0].vad_tail_ms == pytest.approx(tail_s * 1000)
        prov = [j for j in jobs if j.kind == "provisional"]
        seg0 = finals[0].t_media_start
        full = [round(j.t_media_end - j.t_media_start, 3) for j in prov if j.window == "full"]
        assert all(j.t_media_start == seg0 for j in prov if j.window == "full")
        assert all(j.t_media_end - seg0 > svc.cfg.provisional_window_s for j in prov if j.window != "full")
        nt = len(bct.tick_times(n_speech))
        assert len(full) == nt, (dur, full, nt)
        assert np.all(np.asarray(full) - np.asarray(bct.TICKS_S[:nt]) >= 0) and np.allclose(full, bct.TICKS_S[:nt], atol=0.1), (dur, full)
    # a long clip: every full-window tick the service makes, and no more
    assert len(full) == len(bct.TICKS_S) and tcp.FIXED_T == tuple(int(t) for t in bct.TICKS_S)


def test_no_vision_traces_are_never_exported_to_the_repo(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    monkeypatch.setattr(tcp, "ROOT", fake_root)     # the repo's heads/ and results/ stay untouched if the guard fails
    for split, n, seed in (("train", 20, 1), ("dev", 10, 2)):
        synth_traces(tmp_path / f"commit_traces_{split}.npz", n, seed, mode="no_vision")
    with pytest.raises(SystemExit, match="no_vision"):
        tcp.main(["--traces", str(tmp_path), "--skip-test", "--costs", "0.05"])
    assert not (fake_root / "heads").exists() and not (fake_root / "results").exists()
    assert tcp.main(["--traces", str(tmp_path), "--heads", str(tmp_path / "h"), "--results", str(tmp_path / "r"), "--skip-test",
                     "--costs", "0.05"]) == 0


def test_service_commit_tick_with_the_real_policy(tmp_path):
    """meld_rt.service._commit_tick (package B) with a CommitPolicy whose answer is known: P(correct) =
    sigmoid(20 max_p - 10), Q_wait = 0.6, c = 0.05. a soft first tick waits, a sharp second tick emits."""
    pytest.importorskip("cv2")
    from test_service_vision import bare_service

    from meld_rt.service import Job, Stream, VisionResult

    w = np.zeros(len(FEATURES))
    w[FEATURES.index("max_p")] = 20.0
    pol = CommitPolicy.build(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), w, -10.0, 0.05,
                             {"kind": "ridge", "w": np.zeros(len(FEATURES)), "b": 0.6}, "known")
    pol.save(tmp_path / "commit_policy.npz")
    svc = bare_service(tmp_path)
    svc.commit_policy = CommitPolicy.load(tmp_path / "commit_policy.npz")
    svc.commit_policy_name = svc.commit_policy.name
    st = Stream("s", vad=None)
    vis = VisionResult(False, UNIFORM, UNIFORM)

    def job(t_end):
        return Job("provisional", "s", 1, None, None, "full", [0, 1], 0.0, t_end, 0, 7, 0.0, None, [])

    soft = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    sharp = np.eye(7)[4] * 0.85 + 0.15 / 7
    assert svc._commit_tick(st, job(1.024), soft, UNIFORM, UNIFORM, vis, 0.9, 2) is None
    c = svc._commit_tick(st, job(2.048), sharp, UNIFORM, sharp, vis, 0.9, 5)
    assert c is not None and c["label"] == "joy" and c["t_commit_s"] == 2.048 and c["policy"] == "known"
    x = tick_features(st.ticks[1][1])
    assert x[FEATURES.index("same_label_run")] == 1 and x[FEATURES.index("agreement")] == 1   # audio agrees, text uniform
    e, qe, qw = svc.commit_policy.decide(x)
    assert e and c["q_emit"] == round(qe, 4) and c["q_wait"] == round(qw, 4) == 0.6
    assert qe == pytest.approx(1 / (1 + np.exp(-(20 * x[0] - 10))) - 0.05 * 2.048)
