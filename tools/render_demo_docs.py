#!/usr/bin/env python
"""write the demo blocks of README.md ("The demo") and docs/design.md (1.8) from
results/demo_log.jsonl: demo.sh's own summary, run on the log, an excerpt of the
final state with the response, the paragraph that explains them, and the whole
final state for design 1.8. every number comes from the log or heads/manifest.json.

the three-input demo block (demo_tav.sh's summary run on results/demo_tav_log.jsonl,
the final state's vision, commit and discrepancy fields, the response with its
grounding and checks, and a paragraph) goes between the demo_tav markers in both
files. the markers stay where the docs put them.

usage: python tools/render_demo_docs.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_START, README_END = "The committed log came from", "## Setup on Ubuntu 24.04"
DESIGN_BLOCK = re.compile(r"(This is the final state of the demo in [^\n]*\n\n```json\n)(.*?)(\n```\n)", re.DOTALL)
MONITOR_FIELDS = ("topic", "t_recv_ns", "latency_client_ms")
TAV_START = "<!-- demo_tav: start. tools/render_demo_docs.py writes this block from results/demo_tav_log.jsonl -->"
TAV_END = "<!-- demo_tav: end -->"
DIRECTION_WORDS = {"tone_more_negative": "the voice is less positive than the words",
                   "tone_more_positive": "the voice is more positive than the words",
                   "tone_surprised": "the voice is surprised and the words are not",
                   "mixed": "the valences do not differ enough for a direction"}


def summary_code(root: Path, script: str = "demo.sh") -> str:
    """the python block of demo.sh (or demo_tav.sh) that prints the summary. demo.py runs the same block."""
    demo = (root / script).read_text(encoding="utf-8")
    return demo.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]


def utf8_flags(platform: str | None = None) -> list[str]:
    """windows: -X utf8 for a child python, whose stdout to a pipe would otherwise use the ansi code page."""
    return ["-X", "utf8"] if (platform or sys.platform) == "win32" else []


def demo_summary(root: Path, log: Path, script: str = "demo.sh") -> str:
    """the summary demo.sh (or demo_tav.sh) prints, from the python block of the script itself."""
    code = summary_code(root, script)
    return subprocess.run([sys.executable, "-I", *utf8_flags(), "-c", code, str(log)], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout.rstrip("\n")


def blocks(root: Path) -> tuple[str, str, str]:
    """(readme block, final state for design 1.8, summary) from the log."""
    log = root / "results/demo_log.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    finals = [r for r in rows if r["topic"] == "state.final"]
    assert len(finals) == 1, f"the demo log has {len(finals)} finals, the blocks describe one"
    f = finals[0]
    resp = next(r for r in rows if r["topic"] == "response" and r["utt_id"] == f["utt_id"])
    thr = json.loads((root / "heads/manifest.json").read_text(encoding="utf-8"))["thresholds"]
    summary = demo_summary(root, log)

    excerpt = "\n".join([
        "{" + ", ".join(f"{json.dumps(k)}: {json.dumps(f[k])}" for k in ("kind", "stream_id", "utt_id", "t_media_start", "t_media_end")) + ",",
        f' "asr": {json.dumps(f["asr"])},',
        ' "text": ' + json.dumps({k: f["text"][k] for k in ("partial", "context_k", "keyphrase")}) + ",",
        ' "fused": ' + json.dumps({k: f["fused"][k] for k in ("label", "confidence", "sentiment", "mode", "calibrated")}) + ",",
        f' "discrepancy": {json.dumps(f["discrepancy"])},',
        f' "timing_ms": {json.dumps(f["timing_ms"])},',
        f' "degraded": {json.dumps(f["degraded"])}' + "}",
    ])
    response = json.dumps({k: resp[k] for k in ("stream_id", "utt_id", "text", "template_id", "style", "grounding", "t_emitted_ns")})

    # the paragraph is written for a flagged, calibrated, mid-bucket final; another log needs another paragraph
    txt, aud = f["text"]["emotion"], f["audio"]["emotion"]
    text_top = max(txt, key=txt.get)
    a1, a2 = sorted(aud, key=aud.get, reverse=True)[:2]
    fu, d, tm = f["fused"], f["discrepancy"], f["timing_ms"]
    assert fu["mode"] == "stacked" and fu["calibrated"] and d["flag"], (fu, d)
    conf = fu["confidence"]
    bucket = "low" if conf < thr["low_conf"] else ("high" if conf >= thr["high_conf"] else "mid")
    assert bucket == "mid" and resp["style"] == "ask", (bucket, resp["style"])

    def tok(e: str) -> str:
        return f"`{e}`" if e in ("unknown", "other") else e

    para = (f"The text head's top class for the words is {text_top}. SenseVoice puts most of its mass on {tok(a1)} and most of the rest on {tok(a2)}. "
            f"The JSD between the text posterior and the raw SenseVoice posterior, {d['jsd']}, is above the dev threshold of {thr['jsd_flag']} "
            f"(`jsd_flag` in [heads/manifest.json](heads/manifest.json)), and {DIRECTION_WORDS[d['direction']]}, so the direction is `{d['direction']}`. "
            f"The service runs in `{fu['mode']}` mode, so the fusion head gives the label, {fu['label']}, with a calibrated confidence of {conf}. "
            f"That lies between `low_conf` ({thr['low_conf']}) and `high_conf` ({thr['high_conf']}), so the selector is in the mid bucket, "
            f"prefers an `{resp['style']}` line and takes one from the {fu['label']} and `{d['direction']}` cell. "
            "`en.wav` is a read sentence, so the run shows the mechanism and is not an emotion result. "
            f"The final state left the service {tm['service_total']} ms after the endpoint frame arrived, {tm['provisional_in_flight']} ms of which "
            f"it spent waiting for a provisional that was still running. The {tm['vad_tail']:g} ms of silence that ends the utterance comes before that.")

    intro = ("The committed log came from `./demo.sh models/sensevoice/test_wavs/en.wav` at the default endpoint silence of 500 ms, with the "
             "heads trained on MELD audio in `heads/`. `tools/render_demo_docs.py` writes the rest of this section from that log. "
             "`demo.sh` ends with this summary:")
    fields = ("These are some of the fields of that final state, and the response that followed it. "
              "[docs/design.md](docs/design.md) (section 1.8) shows the whole state.")
    readme_block = "\n".join([intro, "", "```", summary, "```", "", fields, "", "```json", excerpt, "```", "",
                              "```json", response, "```", "", para, "", ""])
    state = json.dumps({k: v for k, v in f.items() if k not in MONITOR_FIELDS}, indent=1)
    return readme_block, state, summary


def tav_block(root: Path, up: str = "") -> str:
    """the three-input demo block from results/demo_tav_log.jsonl and heads/manifest.json. up is the
    path from the document to the repository root, for its links ("../" in docs/)."""
    log = root / "results/demo_tav_log.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    finals = [r for r in rows if r["topic"] == "state.final"]
    assert len(finals) == 1, f"the three-input demo log has {len(finals)} finals, the block describes one"
    f = finals[0]
    resp = next(r for r in rows if r["topic"] == "response" and r["utt_id"] == f["utt_id"])
    deltas = [r for r in rows if r["topic"] == "response.delta" and r["utt_id"] == f["utt_id"]]
    man = json.loads((root / "heads/manifest.json").read_text(encoding="utf-8"))
    vis, core_thr = man["vision"], man["thresholds"]
    vthr = vis["thresholds"]
    summary = demo_summary(root, log, "demo_tav.sh")

    def line(key: str, keys=None, last: bool = False) -> str:
        v = f[key] if keys is None else {k: f[key][k] for k in keys}
        return f' "{key}": {json.dumps(v)}' + ("}" if last else ",")

    v = f["vision"]
    excerpt = "\n".join([
        "{" + ", ".join(f"{json.dumps(k)}: {json.dumps(f[k])}" for k in ("kind", "stream_id", "utt_id", "t_media_start", "t_media_end")) + ",",
        line("asr"),
        line("fused", ("label", "confidence", "sentiment", "mode", "calibrated")),
        line("discrepancy"),
        ' "vision": {"emotion": ' + json.dumps(v["emotion"]) + ",",
        '            "raw": ' + json.dumps(v["raw"]) + ",",
        "            " + ", ".join(f"{json.dumps(k)}: {json.dumps(v[k])}" for k in v if k not in ("emotion", "raw")) + "},",
        line("commit"),
        line("timing_ms"),
        line("degraded", last=True),
    ])
    response = "\n".join([
        "{" + ", ".join(f"{json.dumps(k)}: {json.dumps(resp[k])}" for k in ("stream_id", "utt_id", "text", "template_id", "style", "source", "model",
                                                                          "fallback")) + ",",
        f' "grounding": {json.dumps(resp["grounding"])},',
        f' "checks": {json.dumps(resp["checks"])},',
        " " + ", ".join(f"{json.dumps(k)}: {json.dumps(resp[k])}" for k in ("n_tokens", "stop_reason", "first_delta_ms", "gen_ms", "t_emitted_ns")) + "}",
    ])

    # the paragraph is written for this kind of run: a face in the window, a flagged words-voice gap, an unflagged
    # words-face gap, a fusion3 label below low_conf (the reply asks and names no feeling) and an early policy commit
    txt, aud, fu, d, c, tm = f["text"]["emotion"], f["audio"]["emotion"], f["fused"], f["discrepancy"], f["commit"], f["timing_ms"]
    named = {k: p for k, p in aud.items() if k not in ("other", "unknown")}
    assert v["present"] and d["flag"] and not d["face_flag"] and fu["mode"] == "stacked_tav" and fu["calibrated"], (v, d, fu)
    assert fu["confidence"] < vthr["low_conf"] and resp["source"] == "llm" and resp["checks"]["ok"] and c["source"] == "policy", (fu, resp, c)
    assert max(aud, key=aud.get) == "unknown"
    top = lambda dd: max(dd, key=dd.get)  # noqa: E731
    para = " ".join([
        (f"The face was found in {v['n_frames']} frames of the utterance (presence {v['presence']}, {v['n_tracks']} track, rule `{v['rule']}`). "
         f"Its calibrated top class is {top(v['emotion'])}, and the top class of the raw FER posterior is {top(v['raw'])}."),
        (f"The text head's top class for the words is {top(txt)}. SenseVoice puts most of its mass on `unknown`, and its top named class is "
         f"{top(named)}."),
        (f"The words-voice JSD, {d['jsd']}, is above the dev threshold of {core_thr['jsd_flag']}, so the direction `{d['direction']}` is flagged. "
         f"The words-face JSD, {d['face_jsd']}, is below the threshold of {vis['jsd_face_flag']} (`vision.jsd_face_flag` in "
         f"[heads/manifest.json]({up}heads/manifest.json)), so the face raises no flag."),
        (f"Fusion3 (`{fu['mode']}`) gives {fu['label']} with a calibrated confidence of {fu['confidence']}. That is below `low_conf` "
         f"({vthr['low_conf']}) of the vision mode, so the reply names no feeling and asks a question."),
        (f"The commit policy emitted {c['label']} at {c['t_commit_s']} s after the segment start, before the endpoint, and that label "
         + ("matches the final's." if c["matches_final"] else "does not match the final's.")),
        (f"The final state left the service {tm['service_total']} ms after the endpoint frame arrived. The reply came from "
         f"`{resp['model']}` in {len(deltas)} deltas. The first left the service {resp['first_delta_ms']} ms after the endpoint frame arrived, "
         f"and the generation took {resp['gen_ms']} ms. The reply passes every check of `check_reply`."),
    ])
    intro = (f"The committed log [results/demo_tav_log.jsonl]({up}results/demo_tav_log.jsonl) came from `./demo_tav.sh` on one MELD test clip, "
             "with the vision heads, the commit policy and the deployed reply generator. `tools/render_demo_docs.py` writes this block from "
             "that log. `demo_tav.sh` ends with this summary:")
    fields = "These are the main fields of the final state, with its vision, commit and discrepancy blocks, and the response that followed it."
    return "\n".join([intro, "", "```", summary, "```", "", fields, "", "```json", excerpt, "```", "", "```json", response, "```", "", para])


def put_tav(text: str, block: str, name: str) -> str:
    """the block between the demo_tav markers, which the file must have."""
    assert TAV_START in text and TAV_END in text, f"{name} has no demo_tav markers"
    i, j = text.index(TAV_START) + len(TAV_START), text.index(TAV_END)
    assert i <= j, f"{name}: the demo_tav markers are out of order"
    return text[:i] + "\n" + block + "\n" + text[j:]


def render(root: Path) -> tuple[str, str]:
    """README.md and docs/design.md with their demo blocks replaced."""
    readme_block, state, _ = blocks(root)
    readme = (root / "README.md").read_text(encoding="utf-8")
    start, end = readme.index(README_START), readme.index(README_END)
    if TAV_START in readme[start:end]:        # the three-input block sits at the end of the demo section
        end = readme.index(TAV_START, start)
    readme = readme[:start] + readme_block + readme[end:]
    design = (root / "docs/design.md").read_text(encoding="utf-8")
    m = DESIGN_BLOCK.search(design)
    assert m, "design 1.8 has no final-state block"
    design = design[:m.start(2)] + state + design[m.end(2):]
    return put_tav(readme, tav_block(root), "README.md"), put_tav(design, tav_block(root, "../"), "docs/design.md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="write the demo blocks of README.md and docs/design.md from results/demo_log.jsonl and results/demo_tav_log.jsonl")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true", help="only report whether the files already match the log")
    a = ap.parse_args(argv)
    readme, design = render(a.root)
    paths = {a.root / "README.md": readme, a.root / "docs/design.md": design}
    stale = [p for p, text in paths.items() if p.read_text(encoding="utf-8") != text]
    if a.check:
        print("up to date" if not stale else "out of date: " + ", ".join(str(p) for p in stale))
        return 1 if stale else 0
    for p in stale:
        p.write_text(paths[p], encoding="utf-8", newline="\n")
    print(f"rewrote {', '.join(str(p) for p in stale) or 'nothing, both files match the log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
