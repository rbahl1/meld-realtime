"""tools/extract_audio.py on fake raw trees in tmp_path: the umich layout (mp4 with
video, 48 khz 6-channel aac) and a mirror-style tree (validation/ and train/
folders, wav and mp4). one csv row has no clip and one has a corrupt mp4. the
same dia0_utt0 name exists in train and dev with different audio."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import extract_audio  # noqa: E402

COLS = ["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"]
DEV_CLIPS = [(0, 0), (1, 0), (1, 1)]
DEV_NO_CLIP = (1, 2)
DEV_CORRUPT = (1, 3)
TRAIN_CLIP = (0, 0)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def write_csv(path: Path, rows: list[tuple[int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i, (d, u) in enumerate(rows):
            w.writerow({"Sr No.": i + 1, "Utterance": f"line {d} {u}", "Speaker": "Ross", "Emotion": "neutral",
                        "Sentiment": "neutral", "Dialogue_ID": d, "Utterance_ID": u, "Season": 1, "Episode": 1,
                        "StartTime": "00:00:01,000", "EndTime": "00:00:03,000"})


def to_mp4(src: Path, dst: Path) -> None:
    """a MELD-like clip: a small video stream plus the wav as 48 khz 6-channel aac."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x48:r=5", "-i", str(src), "-shortest",
           "-c:v", "mpeg4", "-c:a", "aac", "-ar", "48000", "-ac", "6", str(dst))


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> dict:
    """four source wavs: three for dev (proxy clips when present, else tones) and
    a train clip of a different length."""
    d = tmp_path_factory.mktemp("src")
    proxy = sorted((ROOT / "proxy").glob("dev_*.wav"))[:3]
    out = {}
    for i, rc in enumerate(DEV_CLIPS):
        p = d / f"dev_{i}.wav"
        if len(proxy) == 3:
            shutil.copy(proxy[i], p)
        else:
            ffmpeg("-f", "lavfi", "-i", f"sine=frequency={300 + 100 * i}:duration={1.0 + 0.5 * i}", "-ar", "16000", str(p))
        out[("dev", rc)] = p
    p = d / "train_0.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=5.5", "-ar", "44100", "-ac", "2", str(p))
    out[("train", TRAIN_CLIP)] = p
    return out


@pytest.fixture(scope="module")
def meld(tmp_path_factory) -> Path:
    m = tmp_path_factory.mktemp("meld")
    write_csv(m / "dev_sent_emo.csv", DEV_CLIPS + [DEV_NO_CLIP, DEV_CORRUPT])
    write_csv(m / "train_sent_emo.csv", [TRAIN_CLIP])
    return m


def umich_tree(root: Path, sources: dict) -> Path:
    raw = root / "MELD.Raw"
    for rc in DEV_CLIPS:
        to_mp4(sources[("dev", rc)], raw / "dev" / "dev_splits_complete" / f"dia{rc[0]}_utt{rc[1]}.mp4")
    (raw / "dev" / "dev_splits_complete" / f"dia{DEV_CORRUPT[0]}_utt{DEV_CORRUPT[1]}.mp4").write_bytes(
        np.random.default_rng(0).bytes(4096))
    tr = raw / "train" / "train_splits"
    to_mp4(sources[("train", TRAIN_CLIP)], tr / "dia0_utt0.mp4")
    (tr / "._dia0_utt0.mp4").write_bytes(b"\x00\x05\x16\x07")  # macos resource fork, must be ignored
    (raw / "test.tar.gz").write_bytes(b"")
    return raw


def mirror_tree(root: Path, sources: dict) -> Path:
    raw = root / "hf_mirror"
    val = raw / "data" / "validation"
    val.mkdir(parents=True)
    rc = DEV_CLIPS[0]
    ffmpeg("-i", str(sources[("dev", rc)]), "-ar", "44100", "-ac", "2", str(val / f"dia{rc[0]}_utt{rc[1]}.wav"))
    for rc in DEV_CLIPS[1:]:
        to_mp4(sources[("dev", rc)], val / "clips" / f"dia{rc[0]}_utt{rc[1]}.mp4")
    (val / f"dia{DEV_CORRUPT[0]}_utt{DEV_CORRUPT[1]}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    to_mp4(sources[("train", TRAIN_CLIP)], raw / "data" / "train" / "dia0_utt0.mp4")
    return raw


def check_outputs(out: Path, sources: dict) -> None:
    for (split, (d, u)), src in sources.items():
        p = out / split / f"dia{d}_utt{u}.wav"
        info = sf.info(str(p))
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16"), p
        # aac adds a few ms of padding, the train and dev dia0_utt0 differ by seconds
        assert abs(info.duration - sf.info(str(src)).duration) < 0.05, p
    assert not list(out.rglob("*.part.wav"))
    assert sorted(p.name for p in (out / "dev").glob("*.wav")) == sorted(f"dia{d}_utt{u}.wav" for d, u in DEV_CLIPS)


def check_missing(out: Path, layout: str) -> dict:
    rep = json.loads((out / "missing.json").read_text())
    no_clip, corrupt = "{}_{}".format(*DEV_NO_CLIP), "{}_{}".format(*DEV_CORRUPT)
    assert rep["count"] == 2
    assert rep["keys"] == {"train": [], "dev": [no_clip, corrupt]}
    dev = rep["splits"]["dev"]
    assert dev["no_clip"] == [no_clip] and list(dev["failed"]) == [corrupt]
    assert dev["failed"][corrupt]["error"].strip(), "ffmpeg error tail recorded"
    assert dev["rows"] == 5 and dev["converted"] + dev["existing"] == 3
    assert rep["splits"]["train"]["count"] == 0
    if layout == "umich":
        assert dev["found_by"]["layout"] + dev["existing"] == 4  # the corrupt mp4 counts as found
    return rep


def test_umich_layout(tmp_path, sources, meld):
    raw = umich_tree(tmp_path, sources)
    out = tmp_path / "audio"
    args = ["--raw", str(tmp_path), "--out", str(out), "--meld", str(meld), "--split", "dev,train", "--jobs", "2"]
    assert extract_audio.main(args) == 0
    check_outputs(out, sources)
    rep = check_missing(out, "umich")
    assert rep["splits"]["dev"]["converted"] == 3 and rep["splits"]["train"]["converted"] == 1
    assert (raw / "dev" / "dev_splits_complete" / "dia0_utt0.mp4").exists()

    # rerun with --delete-mp4: nothing converted again, the converted sources go, the corrupt one stays
    mtime = (out / "dev" / "dia0_utt0.wav").stat().st_mtime_ns
    assert extract_audio.main(args + ["--delete-mp4"]) == 0
    rep = check_missing(out, "umich")
    assert rep["splits"]["dev"]["converted"] == 0 and rep["splits"]["dev"]["existing"] == 3
    assert rep["splits"]["dev"]["deleted_mp4"] == 3 and rep["splits"]["train"]["deleted_mp4"] == 1
    assert (out / "dev" / "dia0_utt0.wav").stat().st_mtime_ns == mtime
    assert sorted(p.name for p in (raw / "dev" / "dev_splits_complete").iterdir()) == ["dia1_utt3.mp4"]
    assert not (raw / "train" / "train_splits" / "dia0_utt0.mp4").exists()
    check_outputs(out, sources)


def test_mirror_layout(tmp_path, sources, meld):
    raw = mirror_tree(tmp_path, sources)
    out = tmp_path / "audio"
    assert extract_audio.main(["--raw", str(raw), "--out", str(out), "--meld", str(meld), "--split", "dev,train",
                               "--jobs", "1", "--delete-mp4"]) == 0
    check_outputs(out, sources)
    rep = check_missing(out, "mirror")
    assert rep["splits"]["dev"]["found_by"]["search"] == 4  # three clips plus the corrupt wav
    # wav sources are never deleted, mp4 sources are
    assert (raw / "data" / "validation" / "dia0_utt0.wav").exists()
    assert not list((raw / "data" / "validation" / "clips").glob("*.mp4"))

    # a rerun for one split keeps the other split's entry
    assert extract_audio.main(["--raw", str(raw), "--out", str(out), "--meld", str(meld), "--split", "train"]) == 0
    rep = json.loads((out / "missing.json").read_text())
    assert set(rep["splits"]) == {"train", "dev"} and rep["count"] == 2


def test_split_names():
    assert extract_audio.split_of(["MELD.Raw", "train", "train_splits"]) == "train"
    assert extract_audio.split_of(["MELD.Raw", "dev", "dev_splits_complete"]) == "dev"
    assert extract_audio.split_of(["test", "output_repeated_splits_test"]) == "test"
    assert extract_audio.split_of(["data", "validation"]) == "dev"
    assert extract_audio.split_of(["train", "final_videos_test"]) == "test"  # nearest part wins
    assert extract_audio.split_of(["MELD.Raw", "clips"]) is None
    assert extract_audio.CLIP_RE.fullmatch("dia12_utt3.MP4")["d"] == "12"
    assert extract_audio.CLIP_RE.fullmatch("dia12_utt3.part.wav") is None


def test_nothing_found(tmp_path, meld):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "dev.tar.gz").write_bytes(b"")
    out = tmp_path / "audio"
    assert extract_audio.main(["--raw", str(tmp_path / "empty"), "--out", str(out), "--meld", str(meld), "--split", "dev"]) == 1
    rep = json.loads((out / "missing.json").read_text())
    assert rep["count"] == 5 and "dev.tar.gz" in rep["splits"]["dev"]["hint"]
