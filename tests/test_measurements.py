"""measurement tools (WP8): parameter counting, bench clip building, the replay
bench's bucketing and analysis on synthetic messages, and the renderer. no
sensevoice or roberta session is created here; everything runs in seconds."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bench_components as bc  # noqa: E402
import count_params as cp  # noqa: E402
import render_measurements as rm  # noqa: E402
import replay_bench as rb  # noqa: E402

SILERO = ROOT / "models/silero_vad.onnx"
EN_WAV = ROOT / "models/sensevoice/test_wavs/en.wav"
SENTIMENT = ROOT / "heads/sentiment.npz"


# -- count_params ------------------------------------------------------------

@pytest.mark.skipif(not SILERO.exists(), reason="models not fetched")
def test_count_onnx_silero_initializers():
    c = cp.count_onnx(SILERO)
    assert c["params"] == 155_908          # design 0: silero 0.156 M
    assert c["by_dtype"] == {"FLOAT": 155_908} and c["int8_elements"] == 0
    assert c["bytes"] == SILERO.stat().st_size and c["n_initializers"] > 0


@pytest.mark.skipif(not SENTIMENT.exists(), reason="heads not trained")
def test_count_npz_counts_weights_not_standardizer():
    c = cp.count_npz(SENTIMENT)
    with np.load(SENTIMENT) as z:
        expect = sum(int(z[k].size) for k in cp.WEIGHT_KEYS if k in z)
        std = sum(int(z[k].size) for k in cp.STANDARDIZER_KEYS if k in z)
    assert c["params"] == expect and c["standardizer_elements"] == std and c["kind"] in ("lr", "mlp")


@pytest.mark.skipif(not SILERO.exists(), reason="models not fetched")
def test_build_marks_missing_models_and_pending_heads(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy(SILERO, models / "silero_vad.onnx")
    heads = tmp_path / "heads"
    heads.mkdir()
    res = cp.build(models, heads)
    by = {r["component"]: r for r in res["rows"]}
    assert by["Silero VAD v4"]["params"] == 155_908
    assert by["SenseVoice-small int8"]["params"] is None and by["SenseVoice-small int8"]["note"] == "file missing"
    assert by["Fusion"]["params"] is None and by["Fusion"]["note"] == "pending"
    assert res["total_inference_path"] == 155_908 and res["piper"] is None
    assert set(res["heads_pending"]) == {"TextContextHead", "AudioCalibrator", "Fusion", "SentimentHead"}
    res["generated_at"] = "test"
    md = cp.render(res, models, heads)
    assert "0.156 M" in md and "pending" in md and "not on disk" in md


# -- bench_components --------------------------------------------------------

@pytest.mark.skipif(not EN_WAV.exists(), reason="models not fetched")
def test_build_clips_slices_and_tiles(tmp_path):
    rows = bc.build_clips(EN_WAV, tmp_path)
    assert [r["label"] for r in rows] == ["2", "3.3", "5", "7.15", "10", "15"]
    assert [r["source"] for r in rows] == ["slice"] * 4 + ["tile"] * 2
    for r, d in zip(rows, bc.DURATIONS):
        x, sr = sf.read(r["file"], dtype="int16")
        assert sr == 16000 and abs(len(x) / sr - d) < 1e-3 and r["duration_s"] == pytest.approx(d, abs=1e-3)
        assert np.abs(x).max() > 0


@pytest.mark.skipif(not (EN_WAV.exists() and SILERO.exists()), reason="models not fetched")
def test_trim_silence_cuts_to_the_vad_span():
    x, _ = sf.read(str(EN_WAV), dtype="float32")
    y = bc.trim_silence(x, SILERO)
    # en.wav: 1.1 s of leading silence and room tone after the last word (WP3/WP6 reports)
    assert 4.0 < len(y) / 16000 < len(x) / 16000 - 1.0
    rows = bc.build_clips(EN_WAV, Path(bc.ROOT / "bench_clips/_test_trim"), (3.0, 12.0), trim=True, vad_model=SILERO)
    assert [r["name"] for r in rows] == ["en_dense_3s.wav", "en_dense_12s.wav"] and rows[1]["source"].startswith("tile")
    shutil.rmtree(bc.ROOT / "bench_clips/_test_trim")


def test_stats_and_timed_count_calls():
    s = bc.stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["p50"] == 3.0 and s["min"] == 1.0 and s["n"] == 5 and s["p95"] == pytest.approx(4.8)
    calls = []
    out = bc.timed(lambda: calls.append(1), runs=5, warmup=2)
    assert len(out) == 5 and len(calls) == 7 and all(t >= 0 for t in out)


# -- replay_bench ------------------------------------------------------------

def test_bucket_edges():
    assert rb.bucket_of(0.5) == "le5s" and rb.bucket_of(5.0) == "le5s"
    assert rb.bucket_of(5.01) == "le8s" and rb.bucket_of(8.0) == "le8s"
    assert rb.bucket_of(8.1) == "le15s" and rb.bucket_of(15.3) == "le15s"


def test_collect_clips_globs_sorted_first_n(tmp_path):
    for i in (3, 1, 2):
        (tmp_path / f"c{i}.wav").write_bytes(b"")
    got = rb.collect_clips([str(tmp_path / "*.wav")], 2)
    assert [p.name for p in got] == ["c1.wav", "c2.wav"]
    got = rb.collect_clips([f"{tmp_path}/c3.wav,{tmp_path}/c1.wav", str(tmp_path / "c1.wav")], 0)
    assert [p.name for p in got] == ["c3.wav", "c1.wav"]


def _final(utt, t_start, t_end, recv, emitted, capture, stages, text="hello there"):
    return (emitted + 500_000, "state.final", {
        "kind": "final", "stream_id": "bench", "utt_id": utt, "t_media_start": t_start, "t_media_end": t_end,
        "t_speech_end_capture_ns": capture, "t_service_recv_ns": recv, "t_emitted_ns": emitted, "continued_from": None,
        "asr": {"text": text, "conf": 0.9}, "fused": {"label": "neutral", "mode": "poe"},
        "timing_ms": {"vad_tail": 400.0, "service_total": (emitted - recv) / 1e6, "provisional_in_flight": 0.0, **stages},
        "degraded": []})


def _prov(utt, t_end, window):
    return (0, "state.provisional", {"stream_id": "bench", "utt_id": utt, "t_media_end": t_end, "window": window,
                                      "stability": 0.9, "timing_ms": {"frontend": 2.0, "sensevoice": 50.0}})


def test_analyze_synthetic_stream():
    ns = 1_000_000
    stages = {"frontend": 5.0, "sensevoice": 100.0, "prosody": 10.0, "text_encoder": 20.0, "heads": 1.0}
    clips = [{"name": "a.wav", "duration_s": 3.0, "media_start": 0.0, "media_end": 3.0},
             {"name": "b.wav", "duration_s": 6.0, "media_start": 4.0, "media_end": 10.0}]
    msgs = [
        (0, "health", {"stream_id": "bench", "dropped_provisionals": 0, "queue_s": 0.0, "rss_mb": 900.0}),
        _prov(1, 1.0, "full"), _prov(1, 2.0, "full"),
        _final(1, 0.0, 3.0, recv=1000 * ns, emitted=1150 * ns, capture=999 * ns, stages=stages),
        (1151 * ns, "response", {"stream_id": "bench", "utt_id": 1, "t_emitted_ns": 1150 * ns + 300_000, "text": "ok"}),
        _prov(2, 5.0, "full"), _prov(2, 6.0, "full"), _prov(2, 7.0, "full"), _prov(2, 8.0, "full"), _prov(2, 9.0, "tail"), _prov(2, 11.0, "tail"),
        _final(2, 4.0, 10.0, recv=8000 * ns, emitted=8300 * ns, capture=7999 * ns, stages=stages),
        (8301 * ns, "response", {"stream_id": "bench", "utt_id": 2, "t_emitted_ns": 8300 * ns + 400_000, "text": "ok"}),
        (0, "error", {"stream_id": "other", "code": "audio_gap"}),
        (0, "health", {"stream_id": "bench", "dropped_provisionals": 1, "queue_s": 0.2, "rss_mb": 950.0}),
    ]
    r = rb.analyze(msgs, clips, "bench")
    assert r["n_finals"] == 2 and r["n_responses"] == 2 and r["n_provisionals"] == 8 and r["errors"] == {}
    assert r["by_bucket"]["le5s"]["n"] == 1 and r["by_bucket"]["le8s"]["n"] == 1 and r["by_bucket"]["le15s"]["n"] == 0
    assert r["by_bucket"]["le5s"]["service_total_ms"]["p50"] == 150.0 and r["by_bucket"]["le5s"]["pass"] is True
    assert r["by_bucket"]["le8s"]["service_total_ms"]["p50"] == 300.0 and r["by_bucket"]["le8s"]["pass"] is True
    assert r["by_bucket"]["le15s"]["pass"] is None
    assert r["reply_after_final_ms"]["max"] == 0.4 and r["reply_after_final_ms"]["pass"] is True
    assert r["by_bucket"]["le5s"]["response_ms"]["p50"] == pytest.approx(150.3)
    assert r["by_bucket"]["le5s"]["client_ms"]["p50"] == pytest.approx(151.5)
    cad = r["provisionals"]
    assert cad["full_window_interval_s"]["n"] == 4 and cad["full_window_interval_s"]["p50"] == 1.0
    assert cad["tail_window_interval_s"]["n"] == 1 and cad["tail_window_interval_s"]["p50"] == 2.0
    assert cad["full_to_tail_switch_interval_s"]["n"] == 1 and cad["dropped_provisionals"] == 1 and cad["pass"] is True
    assert cad["first_provisional_offset_s"]["p50"] == 1.0
    assert r["worker_compute_s"] == pytest.approx((2 * 136.0 + 8 * 52.0) / 1000)
    assert [u["clip"] for u in r["utterances"]] == ["a.wav", "b.wav"] and r["clips_without_final"] == []
    assert r["max_queue_s"] == 0.2 and r["health_rss_mb_last"] == 950.0


def test_clip_attribution_uses_the_speech_end_not_the_preroll_start():
    """the vad pre-roll puts a segment's start inside the previous clip's silent tail; a
    meld clip cut at the end of its subtitle ends well before the vad's end of speech. a
    final belongs to the clip whose span (clip plus the silence after it) holds its end."""
    ns = 1_000_000
    stages = {"frontend": 5.0, "sensevoice": 100.0, "prosody": 10.0, "text_encoder": 20.0, "heads": 1.0}
    clips = [{"name": "a.wav", "duration_s": 3.0, "media_start": 0.0, "media_end": 3.0},
             {"name": "b.wav", "duration_s": 2.0, "media_start": 4.0, "media_end": 6.0},
             {"name": "c.wav", "duration_s": 2.0, "media_start": 7.0, "media_end": 9.0}]
    msgs = [
        (0, "health", {"stream_id": "bench", "dropped_provisionals": 0, "queue_s": 0.0, "rss_mb": 900.0}),
        # speech stops 0.576 s after a's end, inside the silence before b
        _final(1, 0.0, 3.576, recv=1000 * ns, emitted=1150 * ns, capture=999 * ns, stages=stages),
        # starts 0.256 s before clip b (pre-roll in a's tail), speech stops 0.032 s after b's end
        _final(2, 3.744, 6.032, recv=2000 * ns, emitted=2150 * ns, capture=1999 * ns, stages=stages),
        # a forced split of clip c: both pieces end inside c's span
        _final(3, 6.8, 8.0, recv=3000 * ns, emitted=3150 * ns, capture=2999 * ns, stages=stages),
        _final(4, 8.0, 9.5, recv=4000 * ns, emitted=4150 * ns, capture=3999 * ns, stages=stages),
    ]
    r = rb.analyze(msgs, clips, "bench")
    assert [u["clip"] for u in r["utterances"]] == ["a.wav", "b.wav", "c.wav", "c.wav"]
    assert r["clips_without_final"] == [] and r["clips_with_several_finals"] == ["c.wav#2"] and r["finals_unattributed"] == []
    assert r["clip_attribution"] == rb.ATTRIBUTION_RULE and r["finals_ending_after_clip"] == 3 and r["max_end_after_clip_s"] == 0.576
    # the segment start of utt 2 lies in a's span, so attributing by the start would put it on a.wav
    assert rb.clip_of(3.744, clips) == 0 and rb.clip_of(6.032, clips) == 1 and rb.clip_of(-0.1, clips) is None


def test_reattribute_rewrites_only_the_attribution(tmp_path):
    """a bench file written under an older rule gets the current attribution from its
    stored utterances and clip spans. the latencies stay as they were."""
    clips = [{"name": "a.wav", "duration_s": 3.0, "media_start": 0.0, "media_end": 3.0},
             {"name": "b.wav", "duration_s": 2.0, "media_start": 4.0, "media_end": 6.0}]
    utts = [{"utt_id": 1, "clip": None, "clip_index": None, "clip_duration_s": None, "t_media_start": 0.0, "segment_s": 3.4,
             "service_total_ms": 120.0},
            {"utt_id": 2, "clip": "b.wav", "clip_index": 1, "clip_duration_s": 2.0, "t_media_start": 3.744, "segment_s": 2.288,
             "service_total_ms": 130.0}]
    old_note = "a final is attributed to the clip whose media span holds its t_media_end, 0.25 s past the clip end allowed"
    d = {"command": "python tools/replay_bench.py --clips x", "config": {"clips": clips}, "notes": ["one", old_note],
         "results": {"clip_attribution": "t_media_end within [media_start, media_end + 0.25 s]", "clips_without_final": ["a.wav#0"],
                     "clips_with_several_finals": [], "finals_unattributed": [1], "by_bucket": {"le5s": {"n": 2}}},
         "utterances": utts}
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(d))
    att = rb.reattribute(path, "python tools/replay_bench.py --reattribute bench.json")
    out = json.loads(path.read_text(encoding="utf-8"))
    assert att["clips_without_final"] == [] and att["finals_unattributed"] == [] and att["finals_ending_after_clip"] == 2
    assert [u["clip"] for u in out["utterances"]] == ["a.wav", "b.wav"] and [u["service_total_ms"] for u in out["utterances"]] == [120.0, 130.0]
    assert out["results"]["clip_attribution"] == rb.ATTRIBUTION_RULE and out["results"]["by_bucket"] == {"le5s": {"n": 2}}
    assert out["notes"] == ["one", rb.ATTRIBUTION_NOTE] and out["reattributed"]["previous_rule"].endswith("+ 0.25 s]")
    assert rm.counted(att["clips_without_final"], "clips without a final") == "Clips without a final: none"
    assert rm.counted(["a.wav#0", "b.wav#1"], "clips without a final") == "2 clips without a final (a.wav#0, b.wav#1)"


# -- render_measurements -----------------------------------------------------

def test_render_without_json_files_has_placeholders_only(tmp_path):
    md = rm.render(ROOT, tmp_path)
    assert md.count("[not yet measured: run") == 3
    assert md.count(rm.NOT_YET) >= 12 and "## Laptop (user-run)" in md and "## Container (host `vm`)" in md
    assert "## Host that rendered this file" in md and "avx512_vnni" in md and "python 3." in md


def test_render_puts_other_hosts_under_the_laptop_heading(tmp_path):
    """bench json files recorded on another host replace the laptop placeholders;
    the container keeps its own; results/laptop_notes.md is included verbatim."""
    src = sorted((ROOT / "results").glob("replay_bench_*.json"))
    if not src:
        pytest.skip("no committed replay bench json")
    r = json.loads(src[0].read_text(encoding="utf-8"))
    (tmp_path / "replay_bench_vm.json").write_text(json.dumps(r))
    r["host"] = "laptop1"
    (tmp_path / "replay_bench_laptop1.json").write_text(json.dumps(r))
    (tmp_path / rm.LAPTOP_NOTES).write_text("mic session: no dropouts")
    md = rm.render(ROOT, tmp_path)
    container, laptop = md.split("## Laptop (user-run)")
    assert "replay_bench_vm.json" in container and "replay_bench_laptop1.json" in laptop
    assert rm.NOT_YET not in laptop and "mic session: no dropouts" in laptop


@pytest.mark.skipif(not (ROOT / "results/params.json").exists(), reason="count_params.py has not run")
def test_render_from_the_committed_json_files():
    md = rm.render(ROOT, ROOT / "results")
    assert "unreadable" not in md and "### Parameters" in md
    params = json.loads((ROOT / "results/params.json").read_text(encoding="utf-8"))
    assert f"{params['total_inference_path'] / 1e6:.3f} M" in md
    for p in (ROOT / "results").glob("replay_bench_*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        assert p.name in md
        for b in r["results"]["by_bucket"].values():
            if b["n"]:
                assert f"{b['service_total_ms']['p50']:.1f} / {b['service_total_ms']['p95']:.1f}" in md
    for p in (ROOT / "results").glob("bench_components_*.json"):
        b = json.loads(p.read_text(encoding="utf-8"))
        t = max(b["by_threads"], key=int)
        assert f"{b['by_threads'][t]['sensevoice_ms']['2']['p50']:.1f}" in md


def test_render_gives_video_benches_and_the_generator_their_own_sections(tmp_path):
    """replay benches with video are not rendered as audio replay sections; the llm run reports its
    reply numbers without the bank's target; the generator bench has one row per file and thread count."""
    vids = [p for p in sorted((ROOT / "results").glob("replay_bench_*.json")) if json.loads(p.read_text(encoding="utf-8"))["config"].get("video")]
    gens = sorted((ROOT / "results").glob("generator_bench_*.json"))
    if not vids or not gens:
        pytest.skip("no committed video or generator bench json")
    for p in vids + gens:
        shutil.copy(p, tmp_path / p.name)
    md = rm.render(ROOT, tmp_path)
    assert "### Replay bench (" not in md and md.count("### Replay bench with video") == len(vids)
    for p in vids:
        r = json.loads(p.read_text(encoding="utf-8"))
        sec = md.split(f"(`{p.name}`")[1].split("\n### ")[0]
        rp = r["results"]["reply_after_final_ms"]
        if r["config"]["reply"] == "llm":
            fd = r["results"]["vision"]["reply"]["first_delta_ms"]
            assert f"{fd['p50']} / {fd['p95']}" in sec and "is the bank's | n/a |" in sec
        else:
            assert f"| {rp['target_p95_ms']} ms p95 | {rm.pf(rp['pass'])} |" in sec
        assert f"share {r['results']['vision']['commits']['rate_before_endpoint']}" in sec
        assert f"{r['results']['vision']['video']['vision_frame_ms_mean']} over" in sec and f"peak {r['memory']['peak_rss_mb']} MB" in sec
    for p in gens:
        g = json.loads(p.read_text(encoding="utf-8"))
        for f, by_t in g["results"].items():
            for t in (k for k in by_t if k.isdigit()):
                sm = by_t[t]["summary"]
                assert f"| `{f}` | {t} | {sm['first_token_ms']['p50']} / {sm['first_token_ms']['p95']} |" in md
    container, laptop = md.split("## Laptop (user-run)")
    assert "### Reply generator bench" in container and rm.NOT_YET in laptop


def test_commit_hash_reads_git_files_without_running_git():
    h = rm.commit_hash(ROOT)
    assert h == "unknown" or len(h.split()[0]) == 12
    assert rm.commit_hash(Path("/nonexistent")) == "unknown"
