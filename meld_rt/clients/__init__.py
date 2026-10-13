"""clients: replay (wav or typed text, the measured path), mic (sounddevice),
monitor (subscriber that prints and logs). shared socket helpers live here."""

from __future__ import annotations

import sys
import time

import zmq

from meld_rt import protocol as P

SEND_TIMEOUT_MS = 5000


def utf8_stdio(platform: str | None = None) -> None:
    """windows: print utf-8 to stdout and stderr. a pipe or a file there gets the ansi code page,
    which cannot encode every transcript (ex. cjk from sensevoice). other platforms are left as they are."""
    if (platform or sys.platform) != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def connect_push(addr: str = P.IN_ADDR) -> zmq.Socket:
    """push socket to the service's pull; sends fail after SEND_TIMEOUT_MS rather than hang."""
    s = zmq.Context.instance().socket(zmq.PUSH)
    s.setsockopt(zmq.SNDHWM, 0)
    s.setsockopt(zmq.SNDTIMEO, SEND_TIMEOUT_MS)
    s.setsockopt(zmq.LINGER, 2000)
    s.connect(addr)
    return s


def connect_sub(addr: str = P.OUT_ADDR, topics: tuple[str, ...] = ()) -> zmq.Socket:
    """sub socket on the service's pub; empty topics means everything."""
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.RCVHWM, 0)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(addr)
    for t in topics or ("",):
        s.setsockopt_string(zmq.SUBSCRIBE, t)
    return s


def wait_ready(out_addr: str = P.OUT_ADDR, timeout_s: float = 60.0, loaded: bool = True) -> dict | None:
    """block until a health message arrives (and, with loaded, until it says
    models_loaded); returns it, or None on timeout."""
    sub = connect_sub(out_addr, ("health",))
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            if poller.poll(max(0, int((deadline - time.monotonic()) * 1000))):
                _, h = P.parse_out(sub.recv_multipart())
                if not loaded or h.get("models_loaded"):
                    return h
    finally:
        sub.close()
    return None
