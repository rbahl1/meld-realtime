"""train_heads.py and evaluate.py end to end on csv subsets with synthetic text
and audio caches: the audio and fusion stage runs here until MELD audio exists."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import evaluate  # noqa: E402
import train_heads  # noqa: E402

from meld_rt import heads, labels  # noqa: E402
from meld_rt.data import MELD_DIR, dialogues, emotion_ids, key, load_split  # noqa: E402

COLS = ["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"]
SV_OF = {m: labels.SV_EMO_TOKENS.index(t) for t, m in labels.SV_TO_MELD.items()}


def subset(split: str, n_dialogues: int):
    """the first dialogues of a split, extended until every emotion appears."""
    dias = dialogues(load_split(split, MELD_DIR))
    out, k = [], n_dialogues
    while k <= len(dias):
        out = [u for d in dias[:k] for u in d]
        if len({u.emotion for u in out}) == len(labels.EMOTIONS):
            return out
        k += 5
    raise AssertionError("not every emotion present")


def write_csv(utts, path: Path) -> None:
    def ts(s: float) -> str:
        return f"00:00:{s:06.3f}".replace(".", ",")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i, u in enumerate(utts):
            w.writerow({"Sr No.": i, "Utterance": u.text, "Speaker": u.speaker, "Emotion": u.emotion, "Sentiment": u.sentiment,
                        "Dialogue_ID": u.dialogue_id, "Utterance_ID": u.utterance_id, "Season": 1, "Episode": 1,
                        "StartTime": ts(u.start_s % 60), "EndTime": ts(u.end_s % 60)})


def synth_caches(utts, split: str, cache: Path, rng, centers_t, centers_e) -> None:
    y = emotion_ids(utts)
    n = len(utts)
    emb = (centers_t[y] + 1.5 * rng.standard_normal((n, 768))).astype(np.float32)
    np.save(cache / f"text_{split}.npy", emb)
    (cache / f"text_{split}_keys.json").write_text(json.dumps([key(u) for u in utts]))
    logit9 = rng.standard_normal((n, 9))
    logit9[np.arange(n), [SV_OF[labels.EMOTIONS[c]] for c in y]] += 2.0
    emo = np.exp(logit9) / np.exp(logit9).sum(1, keepdims=True)
    ev = rng.uniform(size=(n, 11))
    ev /= ev.sum(1, keepdims=True)
    np.savez(cache / f"audio_{split}.npz", keys=np.array([key(u) for u in utts]), emo_probs=emo.astype(np.float32),
             event_probs=ev.astype(np.float32), enc_pooled=(centers_e[y] + rng.standard_normal((n, 1024))).astype(np.float32),
             prosody=rng.standard_normal((n, 4)).astype(np.float32), asr_text=np.array([u.text for u in utts]),
             asr_conf=rng.uniform(0.5, 1.0, n).astype(np.float32), tokens_per_s=np.full(n, 3.0, np.float32),
             duration_s=np.full(n, 2.0, np.float32))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    meld, cache, out = root / "meld", root / "cache", root / "heads"
    for d in (meld, cache, out):
        d.mkdir()
    rng = np.random.default_rng(0)
    centers_t, centers_e = 2.0 * rng.standard_normal((7, 768)), 2.0 * rng.standard_normal((7, 1024))
    for split, n in (("train", 40), ("dev", 12), ("test", 15)):
        utts = subset(split, n)
        write_csv(utts, meld / f"{split}_sent_emo.csv")
        synth_caches(utts, split, cache, rng, centers_t, centers_e)
    cfg = yaml.safe_load((ROOT / "configs" / "heads.yaml").read_text())
    cfg["text_ctx"]["grid"] = {"k": [0, 1], "class_weight": ["none", "balanced"], "learning_rate_init": [0.001]}
    cfg["text_ctx"]["mlp"].update({"max_epochs": 6, "patience": 3})
    cfg["lr"]["C"] = [0.1, 1]
    cfg["fusion"]["cross_fit_folds"] = 3
    (root / "heads.yaml").write_text(yaml.safe_dump(cfg))
    return root, meld, cache, out


def test_train_full_stage_and_evaluate(workspace):
    root, meld, cache, out = workspace
    assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    man = json.loads((out / "manifest.json").read_text())
    for name in ("text_ctx.npz", "text_ctx_k0.npz", "audio_cal_A.npz", "audio_cal_B.npz", "audio_cal.npz", "fusion.npz", "sentiment.npz"):
        assert (out / name).exists(), name
    assert man["audio_fusion"]["status"] == "trained" and man["audio_rows_dropped"] == {"train": 0, "dev": 0}
    assert man["audio_fusion"]["cross_fit"]["folds"] == 3
    assert sum(man["audio_fusion"]["cross_fit"]["fold_sizes"]) == len(load_split("train", meld))
    assert 0.0 < man["thresholds"]["jsd_flag"] <= 1.0 and man["thresholds_pending"] == {}
    assert man["sentiment"]["variant"] == "fused"
    assert heads.Fusion.load(out / "fusion.npz").input_dim == 19
    assert heads.SentimentHead.load(out / "sentiment.npz").input_dim == 19
    cal = heads.AudioCalibrator.load(out / "audio_cal.npz")
    assert cal.input_dim in (24, 1048)
    assert heads.TextContextHead.load(out / "text_ctx.npz").k in (0, 1)
    # synthetic signal is strong: audio and fused heads beat the majority rate on dev
    assert man["audio_fusion"]["fusion"]["dev"]["weighted_f1"] > 0.5

    md, js = root / "RESULTS.md", root / "metrics.json"
    assert evaluate.main(["--cache", str(cache), "--heads", str(out), "--meld", str(meld), "--out", str(md), str(js)]) == 0
    met = json.loads(js.read_text())
    done = {n for n, r in met["conditions"].items() if r["status"] == "done"}
    assert {"majority", "text_k0", "text_best_k", "text_runtime_regime", "audio_raw_tag", "audio_cal_A", "audio_cal_B",
            "fused_stacked", "fused_poe", "fused_poe_uncalibrated", "sentiment_fused"} <= done
    assert "sentiment_text_only" not in done and met["conditions"]["early_commit"]["status"].startswith("pending")
    r = met["conditions"]["fused_stacked"]
    assert r["n"] == len(load_split("test", meld)) and len(r["confusion"]) == 7 and 0 <= r["ece"] <= 1
    assert "abstain_rate" in met["conditions"]["audio_raw_tag"]
    text = md.read_text()
    assert "| fused_stacked | done |" in text and "pending" in text


def test_train_text_only_on_subset(workspace):
    root, meld, cache, _ = workspace
    out = root / "heads_text_only"
    assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld), "--text-only"]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["text_only"] and man["sentiment"]["variant"] == "text_only" and man["thresholds"]["jsd_flag"] is None
    assert man["audio_rows_dropped"] is None
    assert man["thresholds_pending"] == {"jsd_flag": "needs MELD audio"}
    sent = heads.SentimentHead.load(out / "sentiment.npz")
    assert sent.variant == "text_only" and sent.input_dim == heads.TextContextHead.input_dim_for(sent.k)
    assert float(sent.params["C"]) in (0.1, 1.0)
    assert not (out / "fusion.npz").exists()


def test_audio_stage_drops_rows_without_a_clip(workspace):
    """MELD.Raw lacks a few clips: the audio stage trains on the rest and counts the
    dropped rows per split. the cache rows are shuffled and its encoder states are
    zeros (no --encoder-states), so calibrator B is skipped."""
    root, meld, cache, _ = workspace
    drop_cache, out = root / "cache_drop", root / "heads_drop"
    drop_cache.mkdir()
    rng = np.random.default_rng(1)
    dropped = {}
    for split, n_drop in (("train", 3), ("dev", 2)):
        for name in (f"text_{split}.npy", f"text_{split}_keys.json"):
            (drop_cache / name).write_bytes((cache / name).read_bytes())
        with np.load(cache / f"audio_{split}.npz") as z:
            d = {k: z[k] for k in z.files}
        n = len(d["keys"])
        gone = set(rng.choice(n, n_drop, replace=False).tolist())
        rows = rng.permutation([i for i in range(n) if i not in gone])
        d = {k: v[rows] for k, v in d.items()}
        d["enc_pooled"] = np.zeros_like(d["enc_pooled"])
        np.savez(drop_cache / f"audio_{split}.npz", **d)
        dropped[split] = [str(k) for i, k in enumerate(np.load(cache / f"audio_{split}.npz")["keys"]) if i in gone]
    cfg = yaml.safe_load((root / "heads.yaml").read_text())
    cfg["text_ctx"]["grid"] = {"k": [0], "class_weight": ["none"], "learning_rate_init": [0.001]}
    cfg["lr"]["C"] = [1]
    (root / "heads_drop.yaml").write_text(yaml.safe_dump(cfg))
    assert train_heads.main(["--cache", str(drop_cache), "--out", str(out), "--config", str(root / "heads_drop.yaml"),
                             "--meld", str(meld)]) == 0
    man = json.loads((out / "manifest.json").read_text())
    af = man["audio_fusion"]
    assert man["audio_rows_dropped"] == {"train": 3, "dev": 2} and af["status"] == "trained"
    for split in ("train", "dev"):
        assert af["rows"][split]["dropped_keys"] == dropped[split]
        assert af["rows"][split]["cached"] == len(load_split(split, meld)) - len(dropped[split])
    assert sum(af["cross_fit"]["fold_sizes"]) == len(load_split("train", meld)) - 3
    assert af["audio_cal_B"]["status"].startswith("skipped") and "all zero" in af["audio_cal_B"]["status"]
    assert af["audio_cal"]["variant"] == "A" and heads.AudioCalibrator.load(out / "audio_cal.npz").input_dim == 24
    assert (out / "fusion.npz").exists() and man["sentiment"]["variant"] == "fused"


def test_evaluate_leaves_out_test_rows_without_a_clip(workspace):
    """a few test clips missing: audio and fused rows are scored on the rows that have a clip,
    say how many were left out, and carry the text head's f1 on those same rows."""
    root, meld, cache, out = workspace
    if not (out / "fusion.npz").exists():
        assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    ev_cache = root / "cache_eval_drop"
    ev_cache.mkdir(exist_ok=True)
    for name in ("text_test.npy", "text_test_keys.json"):
        (ev_cache / name).write_bytes((cache / name).read_bytes())
    with np.load(cache / "audio_test.npz") as z:
        d = {k: z[k] for k in z.files}
    n = len(d["keys"])
    np.savez(ev_cache / "audio_test.npz", **{k: v[2:] for k, v in d.items()})
    md, js = root / "RESULTS_drop.md", root / "metrics_drop.json"
    assert evaluate.main(["--cache", str(ev_cache), "--heads", str(out), "--meld", str(meld), "--out", str(md), str(js)]) == 0
    met = json.loads(js.read_text())["conditions"]
    for name in ("audio_raw_tag", "fused_stacked", "fused_poe_uncalibrated", "sentiment_fused"):
        r = met[name]
        assert r["status"] == "done" and r["n"] == n - 2 and r["n_no_audio"] == 2, name
        assert "left out" in r["note"]
    assert 0.0 <= met["fused_stacked"]["text_best_k_same_rows_weighted_f1"] <= 1.0
    assert met["text_best_k"]["n"] == n and "n_no_audio" not in met["text_best_k"]


def test_audit_sample_and_tally(workspace, tmp_path):
    """the disagreement audit draws flagged test rows scored as the service scores them,
    and the tally counts the listener's marks without copying any utterance text."""
    import audit_disagreement as audit

    root, meld, cache, out = workspace
    if not (out / "fusion.npz").exists():
        assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    sheet = tmp_path / "audit.csv"
    # without heads/audio_cal.npz the audit scores the raw sensevoice posterior, as the service does,
    # which is also what the manifest's jsd threshold was chosen on
    no_cal = tmp_path / "heads_no_cal"
    no_cal.mkdir()
    for name in ("text_ctx.npz", "manifest.json"):
        (no_cal / name).write_bytes((out / name).read_bytes())
    thr = json.loads((out / "manifest.json").read_text())["thresholds"]["jsd_flag"]
    assert audit.main(["sample", "--n", "5", "--cache", str(cache), "--heads", str(no_cal), "--meld", str(meld),
                       "--audio", str(tmp_path), "--out", str(sheet)]) == 0
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 0 < len(rows) <= 5 and all(r["direction"] != "none" and float(r["jsd"]) > thr - 1e-4 for r in rows)
    assert all(r[c] == "" for r in rows for c in audit.LISTENER_COLUMNS)
    for r, mark in zip(rows, ["text", "audio", "both", "neither", "text"]):
        r["right_branch"], r["laughter_source"] = mark, "none"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = tmp_path / "audit.md"
    assert audit.main(["tally", "--csv", str(sheet), "--out", str(md)]) == 0
    text = md.read_text()
    assert f"{len(rows)} of {len(rows)} rows are marked" in text and f"threshold was {thr:g}" in text
    assert not any(r["utterance"] in text for r in rows if len(r["utterance"]) > 12)


def test_audit_draw_is_seeded_and_keeps_order():
    import audit_disagreement as audit

    rows = list(range(100))
    a, b = audit.draw(rows, 10, 0), audit.draw(rows, 10, 0)
    assert a == b and a == sorted(a) and len(set(a)) == 10
    assert audit.draw(rows[:3], 10, 0) == [0, 1, 2]
