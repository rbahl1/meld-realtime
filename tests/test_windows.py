"""the windows branches, run on linux by passing or patching the platform: the host
clock, utf-8 stdio, ffmpeg discovery, the wasapi and camera options of the mic client,
the monitor's ~ path, the bench host records without /proc and os.getloadavg, the
provenance rows, the demo runner that stands in for demo.sh and demo_tav.sh, and the
utf-8 summary child. no model and no service here."""

from __future__ import annotations

import csv
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from meld_rt import protocol as P
from meld_rt.clients import mic, monitor, replay, utf8_stdio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import audit_disagreement as audit  # noqa: E402
import bench_components as bc  # noqa: E402
import render_demo_docs as rdd  # noqa: E402
import render_measurements as rm  # noqa: E402
import replay_bench as rb  # noqa: E402

import demo  # noqa: E402

NOWHERE = Path("/nonexistent/meld-realtime-test")


# -- clock -------------------------------------------------------------------

def test_host_clock_is_the_performance_counter_on_windows_only():
    """python 3.12's monotonic clock on windows is GetTickCount64 (15.6 ms steps)."""
    assert P.host_clocks("win32") == (time.perf_counter_ns, time.perf_counter)
    assert P.host_clocks("linux") == (time.monotonic_ns, time.monotonic)
    assert P.host_clocks("darwin") == (time.monotonic_ns, time.monotonic)
    a, b = P.now_ns(), P.now_s()
    assert abs(a / 1e9 - b) < 1.0
    if sys.platform != "win32":
        assert (P._CLOCK_NS, P._CLOCK_S) == (time.monotonic_ns, time.monotonic)


def test_replay_paces_on_the_protocol_clock(monkeypatch):
    """send_frames reads P.now_s, so a t0 from P.now_s is on the same clock on every platform."""
    calls = []
    monkeypatch.setattr(P, "now_s", lambda: calls.append(1) or 100.0)
    sent = []
    push = types.SimpleNamespace(send_multipart=sent.append)
    assert replay.send_frames(push, [np.zeros(P.FRAME_SAMPLES, np.int16)] * 3, "s", realtime=True, t0=100.0) == 3
    assert len(sent) == 3 and len(calls) == 3


# -- stdio -------------------------------------------------------------------

def test_utf8_stdio_on_windows_only(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    utf8_stdio("linux")
    assert out.encoding == "cp1252"
    utf8_stdio("win32")
    assert out.encoding == "utf-8" and err.encoding == "utf-8"
    out.write("transcript 你好\n")
    out.flush()
    assert out.buffer.getvalue() == "transcript 你好\n".encode()


# -- ffmpeg ------------------------------------------------------------------

def test_missing_ffmpeg_is_a_clear_error_on_windows(monkeypatch):
    monkeypatch.setattr(replay.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="winget install --id Gyan.FFmpeg"):
        replay.ffmpeg_tool("ffprobe", "win32")
    assert replay.ffmpeg_tool("ffprobe", "linux") == "ffprobe"   # linux: subprocess reports it, as before
    monkeypatch.setattr(replay.shutil, "which", lambda name: "C:/ffmpeg/bin/" + name + ".exe")
    assert replay.ffmpeg_tool("ffmpeg", "win32") == "ffmpeg"


# -- mic client --------------------------------------------------------------

class FakeSd:
    """sounddevice with one input device on the given host api."""

    def __init__(self, api: str):
        self.api = api

    def query_devices(self, device, kind):
        return {"hostapi": 3, "name": "Microphone Array"}

    def query_hostapis(self, index):
        assert index == 3
        return {"name": self.api}

    class WasapiSettings:
        def __init__(self, auto_convert=False):
            self.auto_convert = auto_convert


def test_wasapi_input_gets_auto_convert_on_windows(monkeypatch):
    monkeypatch.setattr(mic, "sd", FakeSd("Windows WASAPI"))
    extra, api = mic.stream_options(None, "win32")
    assert api == "Windows WASAPI" and extra["extra_settings"].auto_convert is True
    monkeypatch.setattr(mic, "sd", FakeSd("MME"))
    assert mic.stream_options(None, "win32") == ({}, "MME")
    monkeypatch.setattr(mic, "sd", None)          # linux never asks sounddevice
    assert mic.stream_options(None, "linux") == ({}, "")


class FakeStream:
    """an InputStream that calls back twice with silent 32 ms blocks and reports its latency."""
    seen: ClassVar[dict] = {}

    def __init__(self, device, channels, samplerate, dtype, blocksize, callback, **extra):
        FakeStream.seen = {"extra": extra}
        self.blocksize, self.callback, self.latency = blocksize, callback, 0.09

    def __enter__(self):
        def run():
            for _ in range(2):
                self.callback(np.zeros((self.blocksize, 1), np.float32), self.blocksize, None, None)
        threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_mic_start_line_names_host_api_and_latency_on_windows(monkeypatch, platform):
    fake = FakeSd("Windows WASAPI")
    fake.InputStream = FakeStream
    monkeypatch.setattr(mic, "sd", fake)
    monkeypatch.setattr(sys, "platform", platform)
    sent, out = [], io.StringIO()
    n = mic.capture(types.SimpleNamespace(send_multipart=sent.append), "mic", None, 16000, seconds=0.3, out=out)
    assert n == 2 and sent[0][0] == b"ctl" and sent[-1][0] == b"ctl"
    first = out.getvalue().splitlines()[0]
    if platform == "win32":
        assert "host_api=Windows WASAPI input_latency=0.09 s" in first and FakeStream.seen["extra"]["extra_settings"].auto_convert
    else:
        assert first == "capturing device=None sr=16000 -> mic; ctrl-c to stop" and FakeStream.seen["extra"] == {}


class FakeCapture:
    def __init__(self, *args):
        self.args = args

    def set(self, *_):
        return True

    def isOpened(self):
        return True

    def getBackendName(self):
        return {(0,): "MSMF"}.get(self.args, "DSHOW")

    def release(self):
        pass


def test_camera_backend_choice(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    cap = mic.CameraSender("tcp://127.0.0.1:1", "mic", 0)._open()
    assert cap.args == (0,)                          # opencv's own order, as before
    cam = mic.CameraSender("tcp://127.0.0.1:1", "mic", 0, backend="dshow")
    assert cam._open().args == (0, cv2.CAP_DSHOW) and cam.backend_name == "DSHOW"
    assert mic.CameraSender("tcp://127.0.0.1:1", "mic", 1, backend="msmf")._open().args == (1, cv2.CAP_MSMF)
    with pytest.raises(ValueError):
        mic.CameraSender("tcp://127.0.0.1:1", "mic", 0, backend="v4l2")


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_mic_main_turns_off_msmf_hw_transforms_on_windows(monkeypatch, platform):
    """with --camera on windows, OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS defaults to 0 before opencv opens the camera."""
    made = {}

    class Cam:
        sent = before_audio = 0
        error = None
        backend_name = "DSHOW"

        def __init__(self, addr, sid, index, fps, backend):
            made.update(index=index, backend=backend, env=os.environ.get("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"))

        def start(self):
            pass

        def stop(self):
            pass

        def set_audio0(self, t):
            pass

    monkeypatch.delenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(mic, "sd", FakeSd("MME"))
    monkeypatch.setattr(mic, "CameraSender", Cam)
    monkeypatch.setattr(mic, "capture", lambda *a, **k: 0)
    monkeypatch.setattr(mic, "connect_push", lambda addr: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(mic, "utf8_stdio", lambda: None)
    assert mic.main(["--camera", "0", "--camera-backend", "dshow"]) == 0
    assert made["backend"] == "dshow" and made["index"] == 0
    assert made["env"] == ("0" if platform == "win32" else None)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_mic_list_devices_adds_the_rates_on_windows(monkeypatch, capsys, platform):
    devices = [{"index": 0, "name": "Microphone Array", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100.0},
               {"index": 1, "name": "Speakers", "hostapi": 0, "max_input_channels": 0, "default_samplerate": 48000.0},
               {"index": 2, "name": "Microphone Array", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000.0}]
    fake = types.SimpleNamespace(query_devices=lambda: devices, query_hostapis=lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}])
    monkeypatch.setattr(mic, "sd", fake)
    monkeypatch.setattr(mic, "utf8_stdio", lambda: None)
    monkeypatch.setattr(sys, "platform", platform)
    assert mic.main(["--list-devices"]) == 0
    extra = [line for line in capsys.readouterr().out.splitlines() if line.startswith("input ")]
    assert extra == (["input 0: MME, default 44100 Hz", "input 2: Windows WASAPI, default 48000 Hz"] if platform == "win32" else [])


def test_mic_error_names_libportaudio_on_linux_only(monkeypatch, capsys):
    monkeypatch.setattr(mic, "sd", None)
    monkeypatch.setattr(mic, "_SD_ERROR", OSError("PortAudio library not found"), raising=False)
    monkeypatch.setattr(mic, "utf8_stdio", lambda: None)
    for platform, want in (("linux", True), ("win32", False)):
        monkeypatch.setattr(sys, "platform", platform)
        assert mic.main([]) == 2
        assert ("libportaudio2" in capsys.readouterr().err) is want


# -- monitor -----------------------------------------------------------------

@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_monitor_expands_tilde_on_windows(monkeypatch, tmp_path, platform):
    """windows powershell 5.1 and cmd pass ~ through. on linux the shell has already expanded it."""
    home, cwd = tmp_path / "home", tmp_path / "repo"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(monitor, "utf8_stdio", lambda: None)
    assert monitor.main(["--addr", "tcp://127.0.0.1:1", "--jsonl", "~/session.jsonl", "--max-s", "0.2", "--quiet"]) == 0
    if platform == "win32":
        assert (home / "session.jsonl").exists() and not (cwd / "~").exists()
    else:
        assert (cwd / "~" / "session.jsonl").exists()


def test_monitor_log_rows_end_in_lf(tmp_path):
    log = tmp_path / "log.jsonl"
    mon = monitor.Monitor("tcp://127.0.0.1:1", log, quiet=True)
    try:
        mon.handle("error", {"stream_id": "s", "code": "bad_message", "msg": "line one\nline two"})
        mon.handle("health", {"rss_mb": 1.0})
    finally:
        mon.close()
    data = log.read_bytes()
    assert data.count(b"\n") == 2 and b"\r\n" not in data


# -- bench host records ------------------------------------------------------

def test_loadavg_without_os_getloadavg(monkeypatch):
    if hasattr(os, "getloadavg"):
        assert len(bc.loadavg()) == 3 and all(isinstance(x, float) for x in bc.loadavg())
    monkeypatch.delattr(os, "getloadavg", raising=False)
    assert bc.loadavg() == []
    assert rb.loadavg is bc.loadavg                   # the replay bench takes the guarded one
    assert "os.getloadavg" not in Path(rb.__file__).read_text(encoding="utf-8")


def test_cpu_name_on_windows_from_win32_processor(monkeypatch):
    monkeypatch.setitem(sys.modules, "_wmi", types.SimpleNamespace(
        exec_query=lambda q: "Name=Intel(R) Core(TM) Ultra 7 155H\0" if q == "SELECT Name FROM Win32_Processor" else ""))
    assert bc.cpu_name_fallback("win32") == ("Intel(R) Core(TM) Ultra 7 155H", "Win32_Processor")

    def fail(q):
        raise OSError("wmi unavailable")
    monkeypatch.setitem(sys.modules, "_wmi", types.SimpleNamespace(exec_query=fail))
    assert bc.cpu_name_fallback("win32")[1] == "platform.processor()"
    assert bc.cpu_name_fallback("linux")[1] == "platform.processor()"


def test_flags_without_proc_come_from_numpy():
    flags = bc.cpu_flags_fallback(bc.FLAGS_OF_INTEREST)
    assert list(flags) == list(bc.FLAGS_OF_INTEREST)
    assert all(flags[f] is None for f in ("amx_tile", "amx_int8", "amx_bf16"))     # numpy does not report amx
    try:
        proc = set(next(line for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
                        if line.startswith("flags")).split(":", 1)[1].split())
    except (OSError, StopIteration):
        pytest.skip("no /proc/cpuinfo flags to compare with")
    for f in bc.NUMPY_FLAGS:
        if flags[f] is not None:
            assert flags[f] == (f in proc), f


def test_machine_record_without_proc(monkeypatch):
    m = bc.machine(meminfo=NOWHERE, cpuinfo=NOWHERE)
    assert m["mem_mb"]["MemTotal"] > 0 and m["mem_mb"]["MemAvailable"] > 0
    cpu = m["cpu"]
    assert cpu["flags_source"] == "numpy" and cpu["model_source"] in ("Win32_Processor", "platform.processor()")
    assert set(cpu["flags"]) == set(bc.FLAGS_OF_INTEREST)
    # with /proc the record is what it was: no source keys
    if Path("/proc/cpuinfo").exists():
        assert "flags_source" not in bc.cpu_info() and all(isinstance(v, bool) for v in bc.cpu_info()["flags"].values())


def test_render_host_block_without_lscpu_or_proc(monkeypatch):
    monkeypatch.setattr(rm, "run", lambda cmd: "")
    cpu = rm.lscpu_info(cpuinfo=NOWHERE)
    assert cpu["flags_source"] == "numpy" and cpu["flags"]["amx_tile"] is None and cpu["model_name"]
    count, mem = rm.cpu_count(), rm.memory(meminfo=NOWHERE)
    assert count == ("os.cpu_count()", str(os.cpu_count())) and mem[0] == "psutil" and mem[1].endswith(" MB")
    head = "\n".join(rm.header(ROOT, cpu, count, mem))
    assert "amx_tile unknown" in head and f"- CPU ({cpu['source']}): " in head and "flags of interest (from numpy):" in head
    # the labels name the fallback sources, not the linux commands
    assert f"- CPU count (os.cpu_count()): {os.cpu_count()}" in head and "- memory (psutil):" in head
    assert "nproc" not in head and "free -m" not in head
    assert rm.host_sources(cpu, count, mem) == f"`{cpu['source']}`, numpy's CPU detection, `os.cpu_count()`, psutil"
    assert rm.yes_no(True) == "yes" and rm.yes_no(False) == "no" and rm.yes_no(None) == "unknown"


def test_render_host_block_keeps_the_linux_labels(tmp_path):
    import shutil
    if not all(shutil.which(c) for c in ("lscpu", "nproc", "free")) or not Path("/proc/cpuinfo").exists():
        pytest.skip("needs lscpu, nproc, free and /proc")
    md = rm.render(ROOT, tmp_path)
    assert "from `lscpu`, `nproc`, `free -m`, the running interpreter and `.git`." in md
    assert "\n- CPU (lscpu): " in md and "\n- nproc: " in md and "\n- free -m:\n" in md and "\n- flags of interest: " in md
    assert "CPU count" not in md and "memory (psutil)" not in md and "(from numpy)" not in md


def test_render_bench_section_of_a_windows_record(tmp_path):
    src = sorted((ROOT / "results").glob("bench_components_*.json"))
    if not src:
        pytest.skip("no committed component bench")
    b = json.loads(src[0].read_text(encoding="utf-8"))
    linux_lines = rm.bench_section(b, src[0])
    b["host"] = "LAPTOP"
    b["machine"]["cpu"] = {"model_name": "Intel(R) Core(TM) Ultra 7 155H", "model_source": "Win32_Processor", "nproc": 22,
                           "flags": {f: (None if f.startswith("amx") else f in ("avx2", "fma")) for f in bc.FLAGS_OF_INTEREST},
                           "flags_source": "numpy"}
    b["machine"]["mem_mb"] = {"MemTotal": 32000, "MemAvailable": 20000}
    text = "\n".join(rm.bench_section(b, tmp_path / "bench_components_LAPTOP.json"))
    assert "Ultra 7 155H (from Win32_Processor), nproc 22, MemTotal 32000 MB" in text
    assert "Flags of interest (from numpy): avx2 yes, fma yes, avx512f no" in text and "amx_tile unknown" in text
    assert "unknown" not in "\n".join(linux_lines) and "(from numpy)" not in "\n".join(linux_lines)


def test_laptop_notes_saved_with_a_byte_order_mark(tmp_path):
    (tmp_path / rm.LAPTOP_NOTES).write_text("mic session: no dropouts", encoding="utf-8-sig")
    md = rm.render(ROOT, tmp_path)
    assert "\ufeff" not in md and "\nmic session: no dropouts\n" in md


def test_audit_tally_reads_a_csv_with_a_byte_order_mark(tmp_path):
    """excel's "csv utf-8" adds a byte order mark, and the first column must keep its name."""
    sheet, md = tmp_path / "audit.csv", tmp_path / "audit.md"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["right_branch", "laughter_source", "direction", "threshold"])
        w.writerows([["text", "none", "mixed", "0.38"], ["audio", "speaker", "mixed", "0.38"]])
    assert audit.main(["tally", "--csv", str(sheet), "--out", str(md)]) == 0
    assert "2 of 2 rows are marked" in md.read_text(encoding="utf-8")


# -- provenance --------------------------------------------------------------

def test_service_matches_provenance_rows_written_with_backslashes(tmp_path):
    from meld_rt.service import Service, ServiceConfig

    models = tmp_path / "models"
    (models / "sensevoice").mkdir(parents=True)
    for rel in ("silero_vad.onnx", "sensevoice/model.int8.onnx", "roberta-base-11.onnx"):
        (models / rel).write_bytes(b"x")
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"files": [{"path": "models\\silero_vad.onnx", "sha256": "a" * 64},
                                          {"path": "models/sensevoice/model.int8.onnx", "sha256": "b" * 64}]}), encoding="utf-8")
    cfg = ServiceConfig(models=models, heads=tmp_path / "heads", provenance=prov)
    v = Service._versions(types.SimpleNamespace(cfg=cfg, llm_model=None))
    assert v["vad"] == "a" * 12 and v["sensevoice"] == "b" * 12 and v["roberta"] == "size:1"


def test_provenance_rows_and_file_are_posix_and_lf(tmp_path, monkeypatch):
    import fetch_models as fm

    dest = tmp_path / "models"
    (dest / "sub").mkdir(parents=True)
    (dest / "sub" / "a.bin").write_bytes(b"core file")
    monkeypatch.setattr(fm, "TABLE", [fm.Asset("a", "http://127.0.0.1:9/a", "sub/a.bin", sha256=None)])
    prov = tmp_path / "prov.json"
    assert fm.main(["--dest", str(dest), "--dl-dir", str(tmp_path / "dl"), "--provenance", str(prov), "--check"]) == 0
    row = json.loads(prov.read_text(encoding="utf-8"))["files"][0]
    assert Path(row["path"]) == (dest / "sub" / "a.bin").resolve() and "\\" not in row["path"]
    assert b"\r\n" not in prov.read_bytes()


# -- demo runner -------------------------------------------------------------

def bash_commands(script: str, env: dict) -> list[tuple[str, list[str]]]:
    """(echo line, python arguments) of each `uv run python -m` step of a demo script, with env substituted."""
    text = (ROOT / script).read_text(encoding="utf-8").split("<<'EOF'", 1)[0]
    text = text.replace('${LLM_ARGS[@]+"${LLM_ARGS[@]}"} ', "")
    lines, buf = [], ""
    for line in text.splitlines():
        buf += line.rstrip("\\").strip() + " "
        if not line.endswith("\\"):
            lines.append(buf.strip())
            buf = ""
    out, echo = [], None

    def sub(s: str) -> str:
        for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
            s = s.replace("${" + k + "}", v).replace("$" + k, v)
        return s
    for line in lines:
        if line.startswith('echo "=='):
            echo = sub(line[len('echo "'):-1])
        elif line.startswith("uv run python -m"):
            out.append((echo, shlex.split(sub(line.removesuffix("&").strip()))[3:]))
    return out


def test_demo_runner_matches_demo_sh(monkeypatch):
    for k in ("THREADS", "ENDPOINT_MS", "N_FINALS", "DEMO_LOG"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(ROOT)
    p = demo.plan(False, "models/sensevoice/test_wavs/en.wav")
    want = bash_commands("demo.sh", {"THREADS": "4", "ENDPOINT_MS": "500", "N_FINALS": "1", "LOG": p["log"], "CLIP": p["clip"]})
    assert p["log"] == "results/demo_log.jsonl" and len(want) == 3
    assert [(e, c[1 + len(rdd.utf8_flags()):]) for e, c in p["steps"]] == want
    monkeypatch.setenv("THREADS", "2")
    monkeypatch.setenv("DEMO_LOG", "out/x.jsonl")
    p = demo.plan(False, "models/sensevoice/test_wavs/en.wav")
    assert p["steps"][0][1][p["steps"][0][1].index("--threads") + 1] == "2" and p["log"] == "out/x.jsonl"


def test_demo_runner_matches_demo_tav_sh(monkeypatch, tmp_path):
    clip = tmp_path / "dia4_utt7.mp4"
    clip.write_bytes(b"")
    for k in ("THREADS", "ENDPOINT_MS", "N_FINALS", "DEMO_LOG", "VIDEO_FPS", "REPLY", "LLM_DIR", "DEMO_MP4"):
        monkeypatch.delenv(k, raising=False)
    p = demo.plan(True, str(clip))
    want = bash_commands("demo_tav.sh", {"THREADS": "4", "ENDPOINT_MS": "500", "N_FINALS": "1", "LOG": p["log"], "CLIP": str(clip),
                                         "VIDEO_FPS": "10", "REPLY": "llm"})
    assert p["log"] == "results/demo_tav_log.jsonl" and [(e, c[1 + len(rdd.utf8_flags()):]) for e, c in p["steps"]] == want
    monkeypatch.setenv("LLM_DIR", "models/llm/Qwen2.5-0.5B-Instruct")
    monkeypatch.setenv("DEMO_MP4", str(clip))
    svc = demo.plan(True, None)["steps"][0][1]
    assert svc[svc.index("--reply") + 2:svc.index("--reply") + 4] == ["--llm-dir", "models/llm/Qwen2.5-0.5B-Instruct"]


def test_demo_runner_clip_rules(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEMO_MP4", raising=False)
    p = demo.plan(False, None)
    assert p["clip"] == demo.FALLBACK_WAV and p["notes"] == [f"no clip at {demo.DEFAULT_WAV}; using {demo.FALLBACK_WAV}"]
    with pytest.raises(SystemExit) as e:
        demo.plan(True, None)
    assert e.value.code == 2 and "no mp4 at '': give one as the first argument or set DEMO_MP4" in capsys.readouterr().err


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_demo_runner_uses_utf8_mode_on_windows_only(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    cmd = demo.python("-m", "meld_rt.service")
    assert cmd == [sys.executable, *(["-X", "utf8"] if platform == "win32" else []), "-m", "meld_rt.service"]


class FakePopen:
    made: ClassVar[list] = []

    def __init__(self, cmd):
        self.cmd, self.returncode, self.terminated = cmd, None, False
        FakePopen.made.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, -15

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize("replay_rc", [0, 3])
def test_demo_runner_steps_exit_codes_and_cleanup(monkeypatch, tmp_path, capsys, replay_rc):
    monkeypatch.chdir(ROOT)                                         # demo.main changes to the root, this restores the cwd
    log = tmp_path / "logs" / "demo.jsonl"
    log.parent.mkdir()
    log.write_text("old\n", encoding="utf-8")
    monkeypatch.setenv("DEMO_LOG", str(log))
    FakePopen.made = []
    calls = []
    monkeypatch.setattr(demo.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(demo.subprocess, "call", lambda cmd: calls.append(cmd) or (replay_rc if len(calls) == 1 else 0))
    rc = demo.main([str(ROOT / "models/sensevoice/test_wavs/en.wav")])
    out = capsys.readouterr().out.splitlines()
    svc, mon = FakePopen.made
    assert svc.cmd[svc.cmd.index("-m") + 1] == "meld_rt.service" and mon.cmd[mon.cmd.index("-m") + 1] == "meld_rt.clients.monitor"
    assert log.read_bytes() == b""                                  # : > "$LOG"
    assert svc.terminated and mon.terminated == bool(replay_rc)     # a monitor that already exited is left alone
    assert out[0].startswith("== service: threads=4") and out[1].startswith("== monitor:") and out[2].startswith("== replay:")
    if replay_rc:
        assert rc == 3 and len(calls) == 1 and "== summary" not in out          # set -e
    else:
        assert rc == 0 and out[-1] == "== summary" and calls[1][-3:] == ["-c", rdd.summary_code(ROOT), str(log)]


def test_demo_runner_summary_is_the_script_summary():
    """the summary demo.py runs is demo.sh's own block, and prints what render_demo_docs prints."""
    log = ROOT / "results/demo_log.jsonl"
    if not log.exists():
        pytest.skip("no committed demo log")
    r = subprocess.run(demo.python("-c", rdd.summary_code(ROOT, "demo.sh"), str(log)), capture_output=True, text=True,
                       encoding="utf-8", check=True)
    assert r.stdout.rstrip("\n") == rdd.demo_summary(ROOT, log)


def test_summary_child_runs_in_utf8_mode_on_windows(monkeypatch, tmp_path):
    """the summary of a log with a transcript outside cp1252 comes through the -X utf8 child."""
    src = ROOT / "results/demo_log.jsonl"
    if not src.exists():
        pytest.skip("no committed demo log")
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if r["topic"] == "state.final":
            r["asr"]["text"] = "你好 hello"
    log = tmp_path / "log.jsonl"
    log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert rdd.utf8_flags("win32") == ["-X", "utf8"] and rdd.utf8_flags("linux") == []
    monkeypatch.setattr(sys, "platform", "win32")
    assert "'你好 hello'" in rdd.demo_summary(ROOT, log)
