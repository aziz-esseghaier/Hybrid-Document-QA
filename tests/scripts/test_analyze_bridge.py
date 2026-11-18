"""Tests for the bridge gap analysis."""

import importlib.util
import json
from pathlib import Path

import pytest

from doc_qa.core.parallel import write_csv, write_jsonl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_bridge.py"
spec = importlib.util.spec_from_file_location("analyze_bridge", SCRIPT)
analyze_bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_bridge)


def page(page_id: str, layer: int, ocr: int, fallback: bool = False) -> dict:
    return {
        "page_id": page_id, "url": "https://example.org/a.pdf", "pdf_page": 1, "correlation": 0.99, "runner_up": 0.2,
        "layer_word_chars": layer, "ocr_word_chars": ocr, "fallback": fallback, "token_f1": 0.5,
    }


def write_run(root: Path, source: str, rankings: dict[str, list[str]], metrics: dict) -> None:
    run_dir = root / "text_eval" / f"d-bridge-{source}"
    for method in analyze_bridge.METHODS:
        records = [{"query_id": q, "pages": [[p, 1.0] for p in pages]} for q, pages in rankings.items()]
        write_jsonl(run_dir / f"rankings_{method}.jsonl", records)
    (run_dir / "metrics.json").write_text(json.dumps({m: metrics for m in analyze_bridge.METHODS}), encoding="utf-8")


def group_metrics(ndcg: float, recall: float) -> dict:
    return {"ndcg@10": ndcg, "recall@100": recall}


def test_text_group():
    def group(layer, ocr, fallback="False"):
        return analyze_bridge.text_group({"layer_word_chars": layer, "ocr_word_chars": ocr, "fallback": fallback})

    assert group(10, 500, "True") == "same_text"
    assert group(200, 100) == "ocr_lacks_most"
    assert group(130, 100) == "ocr_lacks_some"
    assert group(100, 100) == "similar"
    assert group(100, 130) == "pdf_lacks_some"
    assert group(60, 0) == "ocr_lacks_most"


def test_analyze_finds_pages_only_pdf_retrieves(tmp_path):
    write_csv(tmp_path / "bridge" / "d" / "pages.csv", [page("p1", 400, 100), page("p2", 100, 100), page("p3", 5, 90, True)])
    qrels = {"en1": {"p1": 1}, "fr1": {"p1": 1}, "en2": {"p2": 1, "outside": 1}, "en3": {"outside": 1}}
    languages = {"en1": "en", "fr1": "fr", "en2": "en", "en3": "en"}

    write_run(tmp_path, "ocr", {"en1": ["p2"], "fr1": ["p2"], "en2": ["p2"]}, {
        "all": group_metrics(1 / 3, 1 / 3), "en": group_metrics(0.5, 0.5), "fr": group_metrics(0.0, 0.0),
    })
    write_run(tmp_path, "pdf", {"en1": ["p1"], "fr1": ["p1"], "en2": ["p2"]}, {
        "all": group_metrics(1.0, 1.0), "en": group_metrics(1.0, 1.0), "fr": group_metrics(1.0, 1.0),
    })
    summary, rows = analyze_bridge.analyze(tmp_path, "d", qrels, languages)

    assert summary["queries"] == 3 and summary["question_groups"] == 2  # en1 and fr1 are one question
    rrf = summary["methods"]["rrf"]
    assert rrf["scores"]["all"]["recall@100"]["delta"] == pytest.approx(2 / 3, abs=1e-4)
    assert rrf["by_text_group"]["ocr_lacks_most"] == {
        "pairs": 2, "ocr_hit@10": 0.0, "pdf_hit@10": 1.0, "ocr_hit@100": 0.0, "pdf_hit@100": 1.0,
        "only_pdf@100": 2, "only_ocr@100": 0,
    }
    assert rrf["by_text_group"]["similar"]["pairs"] == 1
    assert len(rows) == 9 and {r["page_id"] for r in rows} == {"p1", "p2"}
    assert not any("text" == key for row in rows for key in row)


def test_analyze_rejects_rankings_that_disagree_with_metrics(tmp_path):
    write_csv(tmp_path / "bridge" / "d" / "pages.csv", [page("p1", 100, 100)])
    for source in ("ocr", "pdf"):
        write_run(tmp_path, source, {"q": ["p1"]}, {"all": group_metrics(0.1, 0.1), "en": group_metrics(0.1, 0.1)})
    with pytest.raises(SystemExit, match="metrics.json"):
        analyze_bridge.analyze(tmp_path, "d", {"q": {"p1": 1}}, {"q": "en"})
