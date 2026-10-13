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

import render_demo_docs as rdd  # noqa: E402
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
    met, man = json.loads(METRICS.read_text(encoding="utf-8")), json.loads(MANIFEST.read_text(encoding="utf-8"))
    md = rr.render(met, man, "python tools/render_results.py", "results/metrics.json", "heads/manifest.json")
    for r in met["conditions"].values():
        assert f"`{r['condition']}`" in md
        if r["status"] == "done":
            assert rr.pct(r["weighted_f1"]) in md
    obs = rr.observations(met)
    assert 3 <= len(obs) <= 6 and all(s.endswith(".") and s.count(". ") == 0 for s in obs)
    assert not any(";" in line for line in prose(md)) and ";" not in md


def test_render_results_appends_the_extension_sections():
    """the three-input, commit-policy and streamed-reply sections come after the core report, which they leave as it was."""
    files = {k: ROOT / "results" / f for k, f in (("vision", "metrics_vision.json"), ("commit", "commit_policy.json"),
                                                   ("replay", "replay_bench_vm_tav.json"))}
    gens = sorted((ROOT / "results").glob("generator_bench_*.json"))
    if not all(p.exists() for p in files.values()) or not gens:
        pytest.skip("the extension result files are not there")
    met, man = json.loads(METRICS.read_text(encoding="utf-8")), json.loads(MANIFEST.read_text(encoding="utf-8"))
    ext = {k: (f"results/{p.name}", json.loads(p.read_text(encoding="utf-8"))) for k, p in files.items()}
    ext["generator"] = [(f"results/{p.name}", json.loads(p.read_text(encoding="utf-8"))) for p in gens]
    core = rr.render(met, man, "python tools/render_results.py", "results/metrics.json", "heads/manifest.json")
    md = rr.render(met, man, "python tools/render_results.py", "results/metrics.json", "heads/manifest.json", ext)
    assert md.startswith(core + "\n## Text, voice and face on MELD test")
    assert ";" not in md and not any(ch in md for ch in "\u2013\u2014!")
    vm, cp = ext["vision"][1], ext["commit"][1]
    for r in vm["conditions"].values():
        assert f"`{r['condition']}`" in md and rr.wci(r) in md
    g = vm["comparisons"]["gold_all"]
    assert f"[{rr.pct(g['ci95']['weighted_f1'][0])}, {rr.pct(g['ci95']['weighted_f1'][1])}]" in md
    assert ("It does, so by this rule the face adds measured value here." in md) == vm["claim"]["face_adds"]
    f = vm["label_flips"]["gold"]
    assert f"| {f['n']} | {f['changed']} | {f['wrong_to_right']} | {f['right_to_wrong']} |" in md
    for c, t in cp["test"].items():
        for b, d in t["paired_fqi_minus"].items():
            assert f"| {c} | {b} | {d['diff']:.4f} | [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | {'yes' if t['rl_adds_value_over'][b] else 'no'} |" in md
    dc = str(cp["deploy_c"])
    assert all(f"over {n}" in md for n in ("the endpoint", "the fixed time", "the threshold")) and f"c = {dc}" in md
    ch = rr.deployed_choice([g for _, g in ext["generator"]])
    assert ch is not None and f"The deployed file is `{ch['deployed']}`." in md
    from meld_rt.response import generator
    assert ch["deployed"] == f"{generator.DEFAULT_MODEL}/{generator.DEFAULT_ONNX}"


def test_pending_rows_section_only_while_a_row_is_pending():
    """RESULTS.md has a "Pending rows" section only while a condition is pending. with both sentiment rows
    done, the sentiment section compares the two heads with each other and with the MELD paper rows."""
    met, man = json.loads(METRICS.read_text(encoding="utf-8")), json.loads(MANIFEST.read_text(encoding="utf-8"))
    md = rr.render(met, man, "python tools/render_results.py", "results/metrics.json", "heads/manifest.json")
    waiting = [n for n, r in met["conditions"].items() if r["status"] != "done"]
    assert ("## Pending rows" in md) == bool(waiting) and ("is not scored in this run" in md) == bool(waiting)
    c = met["conditions"]
    if not waiting:
        t, f = c["sentiment_text_only"], c["sentiment_fused"]
        assert (f"The text-only sentiment head scores {rr.pct(t['weighted_f1'])} and the fused one {rr.pct(f['weighted_f1'])}, "
                "next to ") in md
        assert f"The fused head scores {rr.pts(f['weighted_f1'], t['weighted_f1'])} points {rr.side(f['weighted_f1'], t['weighted_f1'])}" in md
    met2 = json.loads(json.dumps(met))
    row = met2["conditions"]["sentiment_text_only"]
    met2["conditions"]["sentiment_text_only"] = {"condition": row["condition"], "task": row["task"], "description": row["description"],
                                                 "status": "pending (run tools/train_heads.py --sentiment-text-only first)"}
    md2 = rr.render(met2, man, "python tools/render_results.py", "results/metrics.json", "heads/manifest.json")
    assert "## Pending rows" in md2 and "| `sentiment_text_only` |" in md2.split("## Pending rows")[1]
    assert "`sentiment_text_only` is not scored in this run (see the pending rows)." in md2
    assert md2.count("## Pending rows") == 1 and md2.index("## Pending rows") < md2.index("## Choices made on dev")


def test_gaps_come_from_the_printed_values():
    """60.9149 and 60.7651 print as 60.91 and 60.77, so the gap reads 0.14, not 0.15."""
    assert rr.pts(0.609149, 0.607651) == "0.14" and rr.side(0.609149, 0.607651) == "above"
    assert rr.side(0.123401, 0.123399) == "level with" and rr.pts(0.123401, 0.123399) == "0.00"


def test_table_cells_with_semicolons_become_sentences():
    assert rr.sentences("Table 8; no WF1 given") == "Table 8. No WF1 given"
    assert rr.sentences("row (WF1; UA 50.8)") == "row (WF1, UA 50.8)"


def test_report_text_only_from_the_committed_files():
    met, man = json.loads(METRICS.read_text(encoding="utf-8")), json.loads(MANIFEST.read_text(encoding="utf-8"))
    md = rto.render(man, met, "python tools/report_text_only.py")
    sel = man["selected"]["text_ctx"]
    assert rto.pct(sel["dev"]["weighted_f1"]) in md and sel["grid_point"] in md
    assert not any(";" in line for line in prose(md))


@pytest.mark.skipif(not (ROOT / "results/demo_tav_log.jsonl").exists(), reason="demo_tav.sh has not run")
def test_demo_tav_block_follows_its_markers(tmp_path):
    """the three-input block goes between the demo_tav markers wherever the docs put them, and the core blocks stay as they were."""
    import shutil
    for rel in ("README.md", "docs/design.md", "results/demo_log.jsonl", "results/demo_tav_log.jsonl", "heads/manifest.json", "demo.sh", "demo_tav.sh"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    readme, design = rdd.render(ROOT)
    marked = readme[readme.index(rdd.TAV_START):readme.index(rdd.TAV_END) + len(rdd.TAV_END)]
    moved = readme.replace(marked + "\n", "") + "\n" + rdd.TAV_START + "\nold text\n" + rdd.TAV_END + "\n"
    (tmp_path / "README.md").write_text(moved, encoding="utf-8")
    new_readme, new_design = rdd.render(tmp_path)
    assert new_readme.endswith("\n" + marked + "\n") and rdd.blocks(ROOT)[0] in new_readme and new_design == design
    block = new_readme.split(rdd.TAV_START)[1].split(rdd.TAV_END)[0]
    log = [json.loads(line) for line in (ROOT / "results/demo_tav_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    final = next(r for r in log if r["topic"] == "state.final")
    resp = next(r for r in log if r["topic"] == "response")
    assert json.dumps(final["commit"]) in block and json.dumps(final["discrepancy"]) in block and json.dumps(resp["grounding"]) in block
    assert json.dumps(resp["checks"]) in block and "](results/demo_tav_log.jsonl)" in block
    assert "](../results/demo_tav_log.jsonl)" in new_design and not any(";" in line for line in prose(block))
    (tmp_path / "README.md").write_text(moved.replace(rdd.TAV_END, ""), encoding="utf-8")
    with pytest.raises(AssertionError, match="no demo_tav markers"):
        rdd.render(tmp_path)


def test_demo_blocks_match_the_committed_log():
    """README.md ("The demo") and design 1.8 are what tools/render_demo_docs.py writes from results/demo_log.jsonl."""
    readme, design = rdd.render(ROOT)
    assert readme == (ROOT / "README.md").read_text(encoding="utf-8")
    assert design == (ROOT / "docs/design.md").read_text(encoding="utf-8")
