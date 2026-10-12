"""report renderers: RESULTS.md (render_results.py) and results/01_text_only.md
(report_text_only.py) from the committed metrics and manifest. every number in
the output comes from those files, and the prose has no semicolons."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_results as rr  # noqa: E402
import report_text_only as rto  # noqa: E402

METRICS, MANIFEST = ROOT / "results/metrics.json", ROOT / "heads/manifest.json"
pytestmark = pytest.mark.skipif(not (METRICS.exists() and MANIFEST.exists()), reason="evaluate.py or train_heads.py has not run")


def prose(md: str) -> list[str]:
    """lines outside tables and code blocks, with inline code removed."""
    out, fence = [], False
    for line in md.splitlines():
        if line.startswith("```"):
            fence = not fence
            continue
        if not fence and not line.startswith("|"):
            out.append(re.sub(r"`[^`]*`", "", line))
    return out


def test_render_results_from_the_committed_files():
    met, man = json.loads(METRICS.read_text()), json.loads(MANIFEST.read_text())
    md = rr.render(met, man, "python tools/render_results.py", "results/metrics.json", "heads/manifest.json")
    for r in met["conditions"].values():
        assert f"`{r['condition']}`" in md
        if r["status"] == "done":
            assert rr.pct(r["weighted_f1"]) in md
    obs = rr.observations(met)
    assert 3 <= len(obs) <= 6 and all(s.endswith(".") and s.count(". ") == 0 for s in obs)
    assert not any(";" in line for line in prose(md))


def test_report_text_only_from_the_committed_files():
    met, man = json.loads(METRICS.read_text()), json.loads(MANIFEST.read_text())
    md = rto.render(man, met, "python tools/report_text_only.py")
    sel = man["selected"]["text_ctx"]
    assert rto.pct(sel["dev"]["weighted_f1"]) in md and sel["grid_point"] in md
    assert not any(";" in line for line in prose(md))
