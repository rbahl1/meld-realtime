"""tools/cache_audio_features.py on three proxy clips plus one missing row, through
--manifest: array names, shapes and prefixes, the meta and wer files, equality
with the expert the service builds, and a run resumed from its checkpoint."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import cache_audio_features as caf  # noqa: E402

from meld_rt.experts.prosody import prosody_scalars  # noqa: E402
from meld_rt.service import Service, ServiceConfig  # noqa: E402

PROXY = ROOT / "proxy" / "manifest.json"
MODELS = [ROOT / "models/sensevoice/model.int8.onnx", ROOT / "models/sensevoice/tokens.txt", ROOT / "models/roberta-base-11.onnx"]
THREADS = 2
PREFIXES = (2.0, 4.0)
N_CLIPS = 3

pytestmark = pytest.mark.skipif(not (all(p.exists() for p in MODELS) and PROXY.exists()), reason="models or proxy clips missing")


def proxy_rows() -> list[dict]:
    rows = json.loads(PROXY.read_text())[:N_CLIPS]
    return [{**r, "file": str(ROOT / "proxy" / r["file"])} for r in rows]


def write_manifest(d: Path, with_missing: bool = True) -> Path:
    rows = proxy_rows() + ([{"file": str(d / "gone.wav"), "key": "999_0", "text": "not there"}] if with_missing else [])
    p = d / "manifest.json"
    p.write_text(json.dumps(rows))
    return p


def args(manifest: Path, out: Path, prefixes: str = "2,4") -> list[str]:
    return ["--split", "dev", "--manifest", str(manifest), "--out", str(out), "--prefixes", prefixes, "--threads", str(THREADS)]


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    out = d / "audio_dev.npz"
    assert caf.main(args(write_manifest(d), out)) == 0
    return d, out, load(out)


def pcm_of(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="int16")
    assert sr == 16000 and x.ndim == 1
    return x.astype(np.float32) / 32768.0  # what the service's ingest makes of the same samples


def test_format(run):
    d, _, z = run
    rows = proxy_rows()
    n = len(rows)
    assert [str(k) for k in z["keys"]] == [r["key"] for r in rows]
    shapes = {"emo_probs": (n, 9), "event_probs": (n, 11), "enc_pooled": (n, 1024), "prosody": (n, 4),
              "asr_text": (n,), "asr_conf": (n,), "tokens_per_s": (n,), "duration_s": (n,)}
    for sfx in ["", "_p2", "_p4"]:
        for name, shape in shapes.items():
            assert z[name + sfx].shape == shape, name + sfx
        assert np.allclose(z["emo_probs" + sfx].sum(1), 1.0, atol=1e-4)
        assert np.allclose(z["event_probs" + sfx].sum(1), 1.0, atol=1e-4)
    assert set(z) == {"keys"} | {f + s for f in shapes for s in ["", "_p2", "_p4"]}
    assert z["asr_text"].dtype.kind == "U" and all(t.strip() for t in z["asr_text"])
    assert not z["enc_pooled"].any(), "zeros without --encoder-states"
    assert np.allclose(z["duration_s"], [r["duration_s"] for r in rows], atol=1e-3)
    dur = z["duration_s"]
    assert (dur > 2.0).any() and (dur < 4.0).any(), "the clips exercise both prefix paths"
    for p in PREFIXES:
        sfx = f"_p{p:g}"
        assert np.allclose(z["duration_s" + sfx], np.minimum(dur, p), atol=1e-6)
        for i in np.flatnonzero(dur <= p):  # a clip no longer than the prefix repeats its full values
            assert np.array_equal(z["emo_probs" + sfx][i], z["emo_probs"][i]) and z["asr_text" + sfx][i] == z["asr_text"][i]
    meta = json.loads((d / "audio_dev_meta.json").read_text())
    assert meta["rows"] == n + 1 and meta["cached"] == n
    assert meta["missing"] == {"count": 1, "keys": ["999_0"]} and meta["failed"]["count"] == 0
    assert not (d / "audio_dev.partial.npz").exists()
    wer = json.loads((d / "wer_dev.json").read_text())
    assert wer["n"] == n and wer["gold"] == "manifest text" and 0.0 <= wer["wer"] < 1.0
    assert wer["reference_words"] == wer["hits"] + wer["substitutions"] + wer["deletions"]


def test_matches_service_expert(run):
    """the cache rows equal what the service's own expert gives for the same samples."""
    _, _, z = run
    svc = Service(ServiceConfig(threads=THREADS, text_threads=1))
    svc._load()
    assert not svc.sv.with_encoder_states
    for i, r in enumerate(proxy_rows()):
        pcm = pcm_of(r["file"])
        res = svc.sv.run(pcm)
        assert np.abs(res.emo_probs - z["emo_probs"][i]).max() <= 1e-6
        assert np.abs(res.event_probs - z["event_probs"][i]).max() <= 1e-6
        assert res.text == z["asr_text"][i] and abs(res.asr_conf - z["asr_conf"][i]) <= 1e-6
        assert np.abs(prosody_scalars(pcm, res.tokens_per_s) - z["prosody"][i]).max() <= 1e-5
        if len(pcm) > 2 * 16000:
            assert np.abs(svc.sv.run(pcm[:2 * 16000]).emo_probs - z["emo_probs_p2"][i]).max() <= 1e-6


def test_resume_from_checkpoint(run, tmp_path, monkeypatch):
    _, _, ref = run
    man, out = write_manifest(tmp_path, with_missing=False), tmp_path / "audio_dev.npz"
    real, calls = caf.row_features, []

    def interrupt_on_second(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(caf, "CHECKPOINT_EVERY", 1)
    monkeypatch.setattr(caf, "row_features", interrupt_on_second)
    assert caf.main(args(man, out)) == 130
    assert not out.exists()
    assert [str(k) for k in load(tmp_path / "audio_dev.partial.npz")["keys"]] == [proxy_rows()[0]["key"]]
    monkeypatch.setattr(caf, "row_features", real)
    with pytest.raises(SystemExit, match="other settings"):
        caf.main(args(man, out, prefixes="2"))
    assert caf.main(args(man, out)) == 0
    meta = json.loads((tmp_path / "audio_dev_meta.json").read_text())
    assert meta["resumed_with"] == 1 and meta["clips_this_run"] == N_CLIPS - 1
    z = load(out)
    assert set(z) == set(ref) and [str(k) for k in z["keys"]] == [str(k) for k in ref["keys"]]
    for k in ref:
        if ref[k].dtype.kind == "U":
            assert list(z[k]) == list(ref[k]), k
        else:
            assert np.abs(z[k] - ref[k]).max() <= 1e-6, k


def test_normalize_for_wer():
    assert caf.normalize("He\u2019s  lost\u2014it\u2026 \x92Okay!") == "hes lost it okay"
    assert caf.normalize("...") == ""
    w = caf.corpus_wer(["Oh my God!", "...", "you\u2019re a genius"], ["oh my god", "uh", "you're genius"])
    assert w["n"] == 2 and w["n_empty_reference_excluded"] == 1 and w["deletions"] == 1 and w["wer"] == pytest.approx(1 / 6, abs=1e-6)
    assert caf.base_of("tokens_per_s_p2") == "tokens_per_s" and caf.base_of("emo_probs_p1.5") == "emo_probs"


def test_work_list_from_csv(tmp_path):
    """without --manifest the rows are the csv's, in load_split order, at clip_path."""
    import csv

    cols = ["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"]
    with (tmp_path / "test_sent_emo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, (d, u, text) in enumerate([(3, 1, "Second."), (3, 0, "First \u2019one\u2019.")]):
            w.writerow({"Sr No.": i, "Utterance": text, "Speaker": "Joey", "Emotion": "joy", "Sentiment": "positive",
                        "Dialogue_ID": d, "Utterance_ID": u, "Season": 1, "Episode": 1,
                        "StartTime": "0:00:01,000", "EndTime": "0:00:02,500"})
    work = caf.work_list("test", tmp_path / "audio", tmp_path, None)
    assert [(k, p, t) for k, p, t in work] == [("3_0", tmp_path / "audio/test/dia3_utt0.wav", "First \u2019one\u2019."),
                                              ("3_1", tmp_path / "audio/test/dia3_utt1.wav", "Second.")]
