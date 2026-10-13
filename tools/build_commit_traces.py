"""build the commit-policy traces (extension spec 5): for every MELD clip of a split, what
the service sees at each full-window provisional tick and at the endpoint, and the fused
posterior there. tools/train_commit_policy.py reads them.

ticks: the service's full-window provisionals, where its commit policy decides
(Service._maybe_provisional, _commit_tick). in vision mode they sit on the segment's 1 Hz
grid at the first frame end at or after k s (1.024, 2.016, 3.008, 4.0 s) and the window is
full up to 4 s, so offline t = 1, 2, 3, 4 s (tests/test_commit.py runs the service's
schedule against this). a tick exists while the vad segment is open, i.e. until the
endpoint silence has run out: t < duration + the vad tail (0.512 s), so a clip also gets
the ticks that fall in its endpoint silence (a 0.6 s clip has one at 1 s).
per tick, as Service._provisional_tav computes a vision-mode provisional:
  - sensevoice (meld_rt.service.build_sensevoice) on the audio prefix [0, t): transcript,
    asr confidence, emotion and event posteriors, tokens per second. past the clip's end
    (in_tail) the prefix is the clip zero-padded to t, the silence the live buffer holds.
    a 2 s or 4 s prefix inside the clip is read from cache/audio_{split}.npz (its _p2 / _p4
    columns, written by tools/cache_audio_features.py with the same expert on the same
    samples) unless --no-prefix-cache; every other prefix is run here.
  - the calibrated audio posterior with the provisional prosody (zeros but the speech rate).
  - roberta on the prefix transcript through the text head; uniform when the transcript is
    empty or its asr confidence is below min_asr_conf (0.3).
  - face evidence over [0, t) with the manifest's speaker rule and the audio envelope of the
    (padded) prefix up to t, through the vision calibrator (uniform without a face). the
    replay client sends every frame of the clip on the media clock, so a tail tick sees them all.
  - fusion3 over the three posteriors.
endpoint (slot 4): t_end = duration + the vad tail (0.512 s) + the p50 final latency of
utterances up to 5 s on the MELD replay bench (results/replay_bench_vm_meld.json, read from
the file). the full-clip sensevoice values and prosody come from cache/audio_{split}.npz, the
text head reads the full sensevoice transcript (every split, as live: the service never sees
gold text), the face evidence covers the whole clip, then fusion3. a transcript that fails
the asr gate makes the final uniform with the label neutral (Service._run_final), and so is
the stored endpoint posterior (fused_end_ungated keeps the ungated one).

heads: train clips use the fold heads of tools/train_vision.py (cache/folds/, the clip's own
fold from fold_of_train.json; train clips outside it are skipped), dev and test the
deployed heads. --no-vision (before the vision heads exist): uniform face, the deployed
two-way fusion (heads/fusion.npz) and the deployed heads for every split, train included
(in sample, to test the code only).

output <out-dir>/commit_traces_{split}.npz, one row per clip in csv order (the csv rows with
a cached audio clip); slot 4 (END) is the endpoint, tick slots at or beyond n_ticks are nan:
  keys, dialogue, utterance, gold (MELD emotion id), fold (-1 outside train), duration_s,
  t_end, n_ticks, t (n, 5), fused / p_text / p_audio / p_face (n, 5, 7) float32, asr_conf,
  presence (n, 5) float32, n_words (n, 5), face_present, text_ok, from_cache, in_tail (n, 5)
  bool, asr_text (n, 5), fused_end_ungated (n, 7), end_degraded (n,), meta (json string); and
  commit_traces_{split}_meta.json. resumable: a checkpoint every 500 clips. --limit writes
  only outside the repository's cache/ (pass --out-dir).

usage: python tools/build_commit_traces.py --splits train,dev,test [--no-vision] [--limit N]
           [--threads 4] [--text-threads 2] [--out-dir cache/]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from cache_audio_features import read_clip  # noqa: E402
from train_heads import file_record, read_audio_cache  # noqa: E402

from meld_rt import heads, labels  # noqa: E402
from meld_rt import vision_features as vf  # noqa: E402
from meld_rt.data import SPLITS, clip_path, context_rows, dialogues, emotion_ids, key, load_split  # noqa: E402

SR = 16000
TICKS_S = (1.0, 2.0, 3.0, 4.0)     # the service's vision-mode full-window provisionals (live 1.024, 2.016, 3.008, 4.0 s)
N_SLOTS = len(TICKS_S) + 1
END = len(TICKS_S)                 # the endpoint slot
N_EMO = len(labels.EMOTIONS)
UNIFORM = np.full(N_EMO, 1.0 / N_EMO)
BENCH = ROOT / "results" / "replay_bench_vm_meld.json"
DEFAULT_TAIL_MS = 512.0            # 500 ms endpoint silence = 16 frames of 32 ms
CHECKPOINT_EVERY = 500
PROGRESS_EVERY = 100
POST_FIELDS = ("fused", "p_text", "p_audio", "p_face")
SLOT_FIELDS = ("asr_conf", "presence", "n_words", "face_present", "text_ok", "from_cache", "in_tail", "asr_text")
CLIP_FIELDS = ("dialogue", "utterance", "gold", "fold", "duration_s", "t_end", "n_ticks", "end_degraded")


# ---------------------------------------------------------------- timing

def endpoint_timing(bench: Path) -> dict:
    """the endpoint's extra time over the clip: the vad tail and the p50 service latency of finals
    of utterances up to 5 s, both from the MELD replay bench."""
    d = json.loads(Path(bench).read_text())
    p50_ms = float(d["results"]["by_bucket"]["le5s"]["service_total_ms"]["p50"])
    tails = Counter(float(u["vad_tail_ms"]) for u in d.get("utterances", []) if u.get("vad_tail_ms") is not None)
    tail_ms = tails.most_common(1)[0][0] if tails else DEFAULT_TAIL_MS   # a forced final has no tail; the mode is the setting
    return {"vad_tail_s": round(tail_ms / 1000.0, 3), "final_p50_s": round(p50_ms / 1000.0, 3), "source": str(bench),
            "fields": "results.by_bucket.le5s.service_total_ms.p50, the most common utterances[].vad_tail_ms",
            "final_p50_ms_exact": p50_ms, "vad_tail_counts_ms": {f"{k:g}": v for k, v in sorted(tails.items())}}


def tick_times(n_samples: int, tail_s: float = DEFAULT_TAIL_MS / 1000.0) -> list[float]:
    """tick times while the vad segment of a clip of n_samples is open: t < duration + tail_s (the
    endpoint silence; the segment closes on its last frame, before that frame's provisional)."""
    return [t for t in TICKS_S if round(t * SR) < n_samples + round(tail_s * SR)]


# ---------------------------------------------------------------- heads

@dataclass
class HeadSet:
    text: heads.TextContextHead
    audio: heads.AudioCalibrator
    vision: heads.VisionCalibrator | None   # None: no vision (uniform face, two-way fusion)
    fusion: heads.NpzHead                   # Fusion3 with vision, else the two-way Fusion
    name: str


def need(path: Path, hint: str) -> Path:
    if not Path(path).exists():
        raise SystemExit(f"{path} is missing ({hint})")
    return Path(path)


def deployed_heads(heads_dir: Path, vision: bool) -> HeadSet:
    t = heads.TextContextHead.load(need(heads_dir / "text_ctx.npz", "tools/train_heads.py"))
    a = heads.AudioCalibrator.load(need(heads_dir / "audio_cal.npz", "tools/train_heads.py"))
    if not vision:
        return HeadSet(t, a, None, heads.Fusion.load(need(heads_dir / "fusion.npz", "tools/train_heads.py")), "deployed_two_way")
    v = heads.VisionCalibrator.load(need(heads_dir / "vision_cal.npz", "tools/train_vision.py"))
    f3 = heads.Fusion3.load(need(heads_dir / "fusion3.npz", "tools/train_vision.py"))
    return HeadSet(t, a, v, f3, "deployed")


def fold_heads(folds_dir: Path) -> tuple[dict[int, HeadSet], dict[str, int]]:
    fold_of = json.loads(need(folds_dir / "fold_of_train.json", "tools/train_vision.py").read_text())
    sets = {}
    for f in sorted({int(v) for v in fold_of.values()}):
        p = {n: need(folds_dir / f"fold{f}_{n}.npz", "tools/train_vision.py") for n in ("text_ctx", "audio_cal", "vision_cal", "fusion3")}
        sets[f] = HeadSet(heads.TextContextHead.load(p["text_ctx"]), heads.AudioCalibrator.load(p["audio_cal"]),
                          heads.VisionCalibrator.load(p["vision_cal"]), heads.Fusion3.load(p["fusion3"]), f"fold{f}")
    return sets, {str(k): int(v) for k, v in fold_of.items()}


def head_files(heads_dir: Path, folds_dir: Path, vision: bool, split: str) -> dict:
    names = ["text_ctx.npz", "audio_cal.npz"] + (["vision_cal.npz", "fusion3.npz"] if vision else ["fusion.npz"])
    if vision and split == "train":
        return {p.name: file_record(p) for p in sorted(folds_dir.glob("fold*_*.npz"))} | {
            "fold_of_train.json": file_record(folds_dir / "fold_of_train.json")}
    return {n: file_record(heads_dir / n) for n in names}


# ---------------------------------------------------------------- experts and posteriors

class TextMemo:
    """the text expert with a cache by transcript (prefix transcripts often repeat)."""

    def __init__(self, expert):
        self.expert, self.memo, self.runs, self.hits = expert, {}, 0, 0

    def encode(self, text: str) -> np.ndarray:
        if text in self.memo:
            self.hits += 1
        else:
            self.memo[text] = np.asarray(self.expert.encode(text), dtype=np.float32)
            self.runs += 1
        return self.memo[text]


def sv_values(r) -> dict:
    enc = getattr(r, "enc_pooled", None)
    return {"emo_probs": np.asarray(r.emo_probs, np.float64), "event_probs": np.asarray(r.event_probs, np.float64),
            "asr_text": str(r.text), "asr_conf": float(r.asr_conf), "tokens_per_s": float(r.tokens_per_s),
            "enc_pooled": None if enc is None else np.asarray(enc, np.float64)}


def cached_values(row: dict, sfx: str) -> dict:
    return {"emo_probs": np.asarray(row["emo_probs" + sfx], np.float64), "event_probs": np.asarray(row["event_probs" + sfx], np.float64),
            "asr_text": str(row["asr_text" + sfx]), "asr_conf": float(row["asr_conf" + sfx]),
            "tokens_per_s": float(row["tokens_per_s" + sfx]), "enc_pooled": np.asarray(row["enc_pooled" + sfx], np.float64)
            if "enc_pooled" + sfx in row else None, "prosody": np.asarray(row["prosody" + sfx], np.float64)}


def padded_prefix(pcm: np.ndarray, t: float) -> np.ndarray:
    """pcm[:t], zero-padded when t is past the clip's end (the silence of the vad tail)."""
    n = round(t * SR)
    return pcm[:n] if n <= len(pcm) else np.concatenate([pcm, np.zeros(n - len(pcm), pcm.dtype)])


def prefix_values(sv, pcm: np.ndarray, t: float, row: dict, reuse: bool) -> tuple[dict, bool]:
    """(sensevoice values on the (padded) prefix [0, t), read from the cache?). a cached _p{t} column
    is used when the prefix lies inside the clip and the cache cut the clip at exactly t."""
    sfx = f"_p{t:g}"
    if reuse and round(t * SR) <= len(pcm) and "emo_probs" + sfx in row and abs(float(row["duration_s" + sfx]) - t) < 1e-3:
        return cached_values(row, sfx), True
    return sv_values(sv.run(padded_prefix(pcm, t))), False


def asr_ok(text: str, conf: float, min_conf: float) -> bool:
    """the service's gate: a transcript that is not empty and whose confidence reaches min_asr_conf."""
    return bool(text.strip()) and conf >= min_conf


def text_posterior(head: heads.TextContextHead, emb: np.ndarray | None, ctx: tuple[list, list, int]) -> np.ndarray:
    if emb is None:
        return UNIFORM.copy()
    prev, same, turn = ctx
    x = heads.TextContextHead.features(emb, prev, same, turn, head.k)
    return np.exp(head.predict_log_proba(x))


def audio_posterior(head: heads.AudioCalibrator, v: dict, prosody4) -> np.ndarray:
    enc = v.get("enc_pooled") if head.uses_encoder_states else None
    if head.uses_encoder_states and (enc is None or not np.any(enc)):
        raise ValueError("the audio calibrator reads encoder states; run with the edited sensevoice graph")
    x = heads.AudioCalibrator.features(v["emo_probs"], v["event_probs"], np.asarray(prosody4, np.float64), enc)
    return np.exp(head.predict_log_proba(x))


def face_posterior(hs: HeadSet, rows, env: tuple[np.ndarray, np.ndarray], rule: str, t1: float | None) -> tuple[np.ndarray, bool, float]:
    """(calibrated face posterior, present, presence) over [0, t1) with the envelope up to t1, or
    over the whole clip when t1 is None. no vision: uniform, absent."""
    if hs.vision is None or rows is None:
        return UNIFORM.copy(), False, 0.0
    et, ev = env
    if t1 is None:
        e = vf.face_evidence(rows, et, ev, rule)
    else:
        m = et < t1
        e = vf.face_evidence(rows, et[m], ev[m], rule, 0.0, t1)
    return vf.calibrated_face(hs.vision, e), bool(e.present), float(e.presence) if e.present else 0.0


def fused_posterior(hs: HeadSet, p_text, p_audio, p_face, present: bool, presence: float, v: dict) -> np.ndarray:
    def lg(p):
        return np.log(np.clip(np.asarray(p, dtype=np.float64), heads.EPS, 1.0))

    laugh, unk = float(v["event_probs"][heads.LAUGHTER_IDX]), float(v["emo_probs"][heads.UNKNOWN_IDX])
    if hs.vision is not None:
        x = vf.build_fusion3_features(lg(p_text), lg(p_audio), lg(p_face), v["asr_conf"], laugh, unk, present, presence)
    else:
        x = heads.build_fusion_features(lg(p_text), lg(p_audio), v["asr_conf"], laugh, unk)
    return np.exp(hs.fusion.predict_log_proba(x))


# ---------------------------------------------------------------- one clip

def empty_record() -> dict:
    r = {f: np.full((N_SLOTS, N_EMO), np.nan, np.float32) for f in POST_FIELDS}
    r.update(t=np.full(N_SLOTS, np.nan), asr_conf=np.full(N_SLOTS, np.nan, np.float32), presence=np.zeros(N_SLOTS, np.float32),
             n_words=np.zeros(N_SLOTS, np.int64), face_present=np.zeros(N_SLOTS, bool), text_ok=np.zeros(N_SLOTS, bool),
             from_cache=np.zeros(N_SLOTS, bool), in_tail=np.zeros(N_SLOTS, bool), asr_text=[""] * N_SLOTS)
    return r


def clip_trace(pcm: np.ndarray, row: dict, hs: HeadSet, sv, text: TextMemo, face_rows, rule: str, ctx: tuple[list, list, int],
               extra_s: float, min_conf: float, reuse: bool, stats: dict, tail_s: float = DEFAULT_TAIL_MS / 1000.0) -> tuple[dict, np.ndarray]:
    """(record, full-transcript embedding for the dialogue context of later clips)."""
    rec = empty_record()
    ticks = tick_times(len(pcm), tail_s)
    no_env = (np.zeros(0), np.zeros(0))
    env = vf.audio_envelope(pcm) if hs.vision is not None else no_env
    # the ticks' envelope runs over the zero tail as the live one does; inside the clip it equals env
    env_ticks = vf.audio_envelope(padded_prefix(pcm, max(ticks))) if hs.vision is not None and ticks and round(max(ticks) * SR) > len(pcm) else env
    for j, t in enumerate(ticks):
        v, cached = prefix_values(sv, pcm, t, row, reuse)
        stats["sv_cached" if cached else "sv_runs"] += 1
        ok = asr_ok(v["asr_text"], v["asr_conf"], min_conf)
        p_text = text_posterior(hs.text, text.encode(v["asr_text"]) if ok else None, ctx)
        p_audio = audio_posterior(hs.audio, v, [0.0, 0.0, 0.0, v["tokens_per_s"]])
        p_face, present, presence = face_posterior(hs, face_rows, env_ticks, rule, t)
        fused = fused_posterior(hs, p_text, p_audio, p_face, present, presence, v)
        _put(rec, j, t, fused, p_text, p_audio, p_face, v, present, presence, ok, cached)
        rec["in_tail"][j] = round(t * SR) > len(pcm)
    full = cached_values(row, "")
    duration = len(pcm) / SR
    ok = asr_ok(full["asr_text"], full["asr_conf"], min_conf)
    emb_full = text.encode(full["asr_text"]) if (ok or hs.text.k > 0) else None
    p_text = text_posterior(hs.text, emb_full if ok else None, ctx)
    p_audio = audio_posterior(hs.audio, full, full["prosody"])
    p_face, present, presence = face_posterior(hs, face_rows, env, rule, None)
    fused = fused_posterior(hs, p_text, p_audio, p_face, present, presence, full)
    rec["fused_end_ungated"] = fused.astype(np.float32)
    rec["end_degraded"] = not ok
    _put(rec, END, duration + extra_s, fused if ok else UNIFORM, p_text, p_audio, p_face, full, present, presence, ok, True)
    rec.update(duration_s=duration, t_end=duration + extra_s, n_ticks=len(ticks))
    return rec, emb_full


def _put(rec: dict, j: int, t: float, fused, p_text, p_audio, p_face, v: dict, present: bool, presence: float, ok: bool,
         cached: bool) -> None:
    rec["t"][j] = t
    for f, p in (("fused", fused), ("p_text", p_text), ("p_audio", p_audio), ("p_face", p_face)):
        rec[f][j] = np.asarray(p, np.float32)
    rec["asr_conf"][j], rec["n_words"][j], rec["asr_text"][j] = v["asr_conf"], len(v["asr_text"].split()), v["asr_text"]
    rec["face_present"][j], rec["presence"][j], rec["text_ok"][j], rec["from_cache"][j] = present, presence, ok, cached


# ---------------------------------------------------------------- arrays and checkpoints

def stack(records: dict[str, dict], keys: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"keys": np.array(keys, dtype=str)}
    if not keys:
        out.update({f: np.zeros((0, N_SLOTS, N_EMO), np.float32) for f in POST_FIELDS})
        out.update(t=np.zeros((0, N_SLOTS)), fused_end_ungated=np.zeros((0, N_EMO), np.float32))
        out.update({f: np.zeros((0, N_SLOTS)) for f in SLOT_FIELDS}, **{f: np.zeros(0) for f in CLIP_FIELDS})
        return out
    for f in POST_FIELDS + ("t", "fused_end_ungated") + SLOT_FIELDS:
        vals = [records[k][f] for k in keys]
        out[f] = np.array(vals, dtype=str) if f == "asr_text" else np.stack(vals)
    for f, dt in (("dialogue", np.int64), ("utterance", np.int64), ("gold", np.int64), ("fold", np.int64),
                  ("duration_s", np.float64), ("t_end", np.float64), ("n_ticks", np.int64), ("end_degraded", bool)):
        out[f] = np.array([records[k][f] for k in keys], dtype=dt)
    return out


def unstack(arrays: dict) -> dict[str, dict]:
    recs = {}
    for i, k in enumerate(str(x) for x in arrays["keys"]):
        r = {f: arrays[f][i] for f in POST_FIELDS + ("t", "fused_end_ungated") + SLOT_FIELDS + CLIP_FIELDS}
        r["asr_text"] = [str(s) for s in r["asr_text"]]
        recs[k] = r
    return recs


def partial_path(out: Path) -> Path:
    return out.with_name(out.stem + ".partial.npz")


def save_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_traces(path: Path) -> dict:
    """a traces file as a dict of arrays, with meta parsed."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"])) if "meta" in d else {}
    return d


# ---------------------------------------------------------------- one split

class SplitBuild:
    """the traces of one split: inputs, heads, the clip loop and its checkpoint."""

    def __init__(self, split: str, a: argparse.Namespace, sv, text: TextMemo, timing: dict, out: Path):
        self.split, self.a, self.sv, self.text, self.timing, self.out = split, a, sv, text, timing, out
        self.utts = load_split(split, a.meld)
        loaded = read_audio_cache(a.cache, split, self.utts)
        if loaded is None:
            raise SystemExit(f"{a.cache / f'audio_{split}.npz'} is missing (tools/cache_audio_features.py)")
        self.aud, self.keep, self.no_audio = loaded
        self.vision = not a.no_vision
        folds_dir = a.cache / "folds"
        manifest = json.loads((a.heads / "manifest.json").read_text()) if (a.heads / "manifest.json").exists() else {}
        self.deployed = deployed_heads(a.heads, self.vision)
        self.rule, self.fold_sets, self.fold_of, self.video = "largest", None, None, None
        if self.vision:
            self.rule = str((manifest.get("vision") or {}).get("rule") or self.deployed.vision.rule or "")
            if self.rule not in vf.RULES:
                raise SystemExit(f"no speaker rule in {a.heads / 'manifest.json'} (vision.rule); run tools/train_vision.py")
            if split == "train":
                self.fold_sets, self.fold_of = fold_heads(folds_dir)
            self.video = vf.load_video_cache(split, need(a.cache / f"video_{split}.npz", "tools/extract_video_features.py --merge").parent)
        self.k_ctx = self.deployed.text.k
        if self.fold_sets and any(h.text.k != self.k_ctx for h in self.fold_sets.values()):
            raise SystemExit("fold text heads and the deployed text head differ in k")
        self.y = emotion_ids(self.utts)
        self.pos = {key(u): i for d in dialogues(self.utts) for i, u in enumerate(d)}
        self.ctx_of = {key(u): prev for u, prev in context_rows(self.utts, self.k_ctx)}
        self.work = list(range(len(self.keep)))[:a.limit] if a.limit is not None else list(range(len(self.keep)))
        self.min_conf = float(a.min_asr_conf)
        self.tail_s = float(timing["vad_tail_s"])
        self.extra_s = self.tail_s + timing["final_p50_s"]
        self.config = json.dumps({"split": split, "vision": self.vision, "rule": self.rule if self.vision else None,
                                  "reuse_prefix_cache": not a.no_prefix_cache, "extra_s": self.extra_s, "min_asr_conf": self.min_conf,
                                  "ticks_s": list(TICKS_S), "tick_rule": "t < duration + vad_tail_s", "vad_tail_s": self.tail_s,
                                  "heads": head_files(a.heads, folds_dir, self.vision, split),
                                  "audio_cache": file_record(a.cache / f"audio_{split}.npz"),
                                  "video_cache": file_record(a.cache / f"video_{split}.npz") if self.vision else None}, sort_keys=True)
        self.records: dict[str, dict] = {}
        self.order: list[str] = []
        self.emb_full: dict[str, np.ndarray] = {}
        self.stats = {"sv_runs": 0, "sv_cached": 0, "skipped_no_fold": 0, "text_runs0": text.runs, "text_hits0": text.hits}
        self.done_now = 0

    @property
    def partial(self) -> Path:
        return partial_path(self.out)

    def checkpoint(self) -> None:
        save_npz(self.partial, {**stack(self.records, [x for x in self.order if x in self.records]),
                                "meta": np.array(json.dumps({"config": self.config}))})

    def resume(self) -> None:
        if not self.partial.exists():
            return
        d = load_traces(self.partial)
        if str(d["meta"].get("config")) != self.config:
            raise SystemExit(f"{self.partial} was written with other settings; delete it to start over")
        self.records = unstack(d)
        print(f"{self.split}: resuming with {len(self.records)} clips from {self.partial}", flush=True)

    def clip(self, j: int) -> bool:
        """trace of csv-kept row j; True when it was computed in this run."""
        a, u = self.a, self.utts[self.keep[j]]
        k = key(u)
        if self.fold_of is not None and k not in self.fold_of:
            self.stats["skipped_no_fold"] += 1
            return False
        self.order.append(k)
        if k in self.records:
            if self.k_ctx > 0:   # context for later clips: the full transcript's embedding, ungated (evaluate.asr_emb)
                self.emb_full[k] = self.text.encode(str(self.aud["asr_text"][j]))
            return False
        hs = self.fold_sets[self.fold_of[k]] if self.fold_of is not None else self.deployed
        row = {f: self.aud[f][j] for f in self.aud}
        pcm = read_clip(clip_path(u, a.audio))
        if abs(len(pcm) / SR - float(row["duration_s"])) > 1e-3:
            raise SystemExit(f"{k}: the wav lasts {len(pcm) / SR:.3f} s, cache/audio_{self.split}.npz says {float(row['duration_s']):.3f} s")
        prev = self.ctx_of[k]
        ctx = ([self.emb_full.get(key(q), np.zeros(heads.TEXT_DIM, np.float32)) for q in prev], [q.speaker == u.speaker for q in prev],
               self.pos[k])
        rows = vf.clip_rows(self.video, k) if self.vision else None
        rec, ef = clip_trace(pcm, row, hs, self.sv, self.text, rows, self.rule, ctx, self.extra_s, self.min_conf,
                             not a.no_prefix_cache, self.stats, self.tail_s)
        if ef is not None:
            self.emb_full[k] = ef
        rec.update(dialogue=u.dialogue_id, utterance=u.utterance_id, gold=int(self.y[self.keep[j]]),
                   fold=self.fold_of[k] if self.fold_of is not None else -1)
        self.records[k] = rec
        self.done_now += 1
        if self.done_now % CHECKPOINT_EVERY == 0:
            self.checkpoint()
        return True

    def run(self) -> dict:
        t_split = time.perf_counter()
        self.resume()
        t0 = time.perf_counter()
        try:
            for n_done, j in enumerate(self.work, 1):
                if self.clip(j) and self.done_now % PROGRESS_EVERY == 0:
                    el = time.perf_counter() - t0
                    print(f"{self.split}: {n_done}/{len(self.work)} clips, {el / self.done_now * 1000:.0f} ms per clip, "
                          f"about {el / self.done_now * (len(self.work) - n_done) / 60:.1f} min left", flush=True)
        except BaseException:
            if self.records:
                self.checkpoint()
                print(f"{self.split}: stopped, {len(self.records)} clips saved to {self.partial}", file=sys.stderr, flush=True)
            raise
        wall = time.perf_counter() - t0
        arrays = stack(self.records, self.order)
        meta = split_meta(self.split, self.a, arrays, y_known=self.split != "test", vision=self.vision, rule=self.rule, timing=self.timing,
                          extra_s=self.extra_s, min_conf=self.min_conf, no_audio=self.no_audio, stats=self.stats, text=self.text,
                          done_now=self.done_now, wall=wall, in_sample=not self.vision and self.split == "train", config=self.config)
        meta["wall_s_total"] = round(time.perf_counter() - t_split, 1)
        save_npz(self.out, {**arrays, "meta": np.array(json.dumps(meta))})
        self.out.with_name(self.out.stem + "_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        self.partial.unlink(missing_ok=True)
        return meta


def build_split(split: str, a: argparse.Namespace, sv, text: TextMemo, timing: dict, out: Path) -> dict:
    """traces of one split -> out; returns the meta record."""
    return SplitBuild(split, a, sv, text, timing, out).run()


def split_meta(split: str, a, arrays: dict, y_known: bool, vision: bool, rule: str, timing: dict, extra_s: float, min_conf: float,
               no_audio: list, stats: dict, text: TextMemo, done_now: int, wall: float, in_sample: bool, config: str) -> dict:
    n = len(arrays["keys"])
    nt = arrays["n_ticks"].astype(np.int64) if n else np.zeros(0, np.int64)
    meta = {"split": split, "clips": n, "mode": "vision" if vision else "no_vision",
            "fusion": "fusion3" if vision else "two-way fusion (heads/fusion.npz), uniform face",
            "heads": ("fold heads (cache/folds/, the clip's fold)" if vision and split == "train" else "deployed heads")
            + ("; in sample on train, for testing only" if in_sample else ""),
            "rule": rule if vision else None, "ticks_s": list(TICKS_S), "tick_rule": "t < duration + endpoint.vad_tail_s (the vad segment is open)",
            "endpoint": {**timing, "extra_s": round(extra_s, 3)},
            "min_asr_conf": min_conf, "reuse_prefix_cache": not a.no_prefix_cache, "limit": a.limit,
            "csv_rows_without_audio": len(no_audio), "skipped_no_fold": stats["skipped_no_fold"],
            "n_ticks_hist": {str(i): int((nt == i).sum()) for i in range(len(TICKS_S) + 1)},
            "tick_rows": int(nt.sum()), "tail_ticks": int(arrays["in_tail"].sum()) if n else 0,
            "clips_with_tail_ticks_only": int(((nt > 0) & (arrays["in_tail"][:, 0])).sum()) if n else 0,
            "sv_runs": stats["sv_runs"], "sv_from_cache": stats["sv_cached"],
            "text_runs": text.runs - stats["text_runs0"], "text_cache_hits": text.hits - stats["text_hits0"], "clips_this_run": done_now, "wall_s": round(wall, 1),
            "ms_per_clip": round(wall / done_now * 1000, 1) if done_now else None, "threads": a.threads, "text_threads": a.text_threads,
            "end_degraded": int(arrays["end_degraded"].sum()) if n else 0, "config": config,
            "command": "python tools/build_commit_traces.py " + " ".join(a.argv)}
    if n:
        ticks = np.arange(N_SLOTS - 1)[None, :] < nt[:, None]
        meta["text_ok_share_ticks"] = round(float(arrays["text_ok"][:, :END][ticks].mean()), 4) if ticks.any() else None
        meta["face_present_share_ticks"] = round(float(arrays["face_present"][:, :END][ticks].mean()), 4) if ticks.any() else None
        meta["face_present_share_end"] = round(float(arrays["face_present"][:, END].mean()), 4)
    if y_known and n:   # never on test: the trace builder reads no test label for a number
        acc = {}
        for j in range(N_SLOTS):
            has = nt > j if j < END else np.ones(n, bool)
            if has.any():
                pred = arrays["fused"][has, j].argmax(axis=1)
                acc["end" if j == END else f"t{TICKS_S[j]:g}"] = {"n": int(has.sum()), "accuracy": round(float((pred == arrays["gold"][has]).mean()), 4)}
        meta["fused_accuracy_by_slot"] = acc
    return meta


# ---------------------------------------------------------------- main

def build_experts(a: argparse.Namespace, need_enc: bool):
    from meld_rt.experts.text import TextExpert
    from meld_rt.service import ServiceConfig, build_sensevoice

    cfg = ServiceConfig(models=a.models, threads=a.threads, text_threads=a.text_threads)
    sv = build_sensevoice(cfg, with_encoder_states=need_enc)
    te = TextExpert(cfg.models / cfg.roberta, cfg.tokenizer_dir, threads=cfg.text_threads)
    return sv, te


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--cache", type=Path, default=ROOT / "cache", help="audio and video caches, folds/")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads")
    ap.add_argument("--audio", type=Path, default=ROOT / "audio")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: --cache")
    ap.add_argument("--bench", type=Path, default=BENCH, help="replay bench json with the endpoint latency")
    ap.add_argument("--no-vision", action="store_true", help="uniform face, deployed two-way fusion, deployed heads everywhere")
    ap.add_argument("--no-prefix-cache", action="store_true", help="run sensevoice on the 2 s and 4 s prefixes too")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips of each split (testing; needs an --out-dir outside cache/)")
    ap.add_argument("--threads", type=int, default=4, help="sensevoice intra-op threads")
    ap.add_argument("--text-threads", type=int, default=2, help="roberta intra-op threads")
    ap.add_argument("--min-asr-conf", type=float, default=None, help="default: ServiceConfig.min_asr_conf")
    return ap


def main(argv: list[str] | None = None, experts=None) -> int:
    """experts: (sensevoice, text expert) to use instead of the real ones (tests)."""
    a = parser().parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    splits = [s for s in a.splits.split(",") if s]
    if any(s not in SPLITS for s in splits):
        raise SystemExit(f"--splits must name {SPLITS}")
    if a.min_asr_conf is None:
        from meld_rt.service import ServiceConfig
        a.min_asr_conf = ServiceConfig().min_asr_conf
    out_dir = a.out_dir or a.cache
    if a.limit is not None and Path(out_dir).resolve() == (ROOT / "cache").resolve():
        # a subset must never replace the full traces that train_commit_policy.py reads by default
        raise SystemExit("--limit writes test traces only: pass an --out-dir outside the repository's cache/")
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = endpoint_timing(a.bench)
    if experts is None:
        cal = heads.AudioCalibrator.load(need(a.heads / "audio_cal.npz", "tools/train_heads.py"))
        t_load = time.perf_counter()
        experts = build_experts(a, cal.uses_encoder_states)
        print(f"experts loaded in {time.perf_counter() - t_load:.1f} s", flush=True)
    sv, te = experts
    text = TextMemo(te)
    for s in splits:
        out = out_dir / f"commit_traces_{s}.npz"
        m = build_split(s, a, sv, text, timing, out)
        print(f"{s}: {m['clips']} clips, {m['tick_rows']} ticks, {m['sv_runs']} sensevoice runs, {m['sv_from_cache']} prefixes "
              f"from the cache, {m['text_runs']} text runs, {m['wall_s']} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
