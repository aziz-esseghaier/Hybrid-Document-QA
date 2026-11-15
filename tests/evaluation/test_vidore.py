"""Tests for the ViDoRe loader's parsing and filtering."""

from pathlib import Path

from doc_qa.core.schemas import Benchmark, Document, Page, Qrel, Query
from doc_qa.evaluation.vidore import apply_exclusions, clean_qrels, filter_languages, parse_answers, v1_ids


def toy_benchmark() -> Benchmark:
    """Two documents; q1 needs a page of each, q2 only doc B, q3 only doc A."""
    pages = {
        "d/a1": Page("d/a1", "d/A", 1),
        "d/a2": Page("d/a2", "d/A", 2),
        "d/b1": Page("d/b1", "d/B", 1),
    }
    return Benchmark(
        name="d",
        role="dev",
        documents={"d/A": Document("d/A", "A"), "d/B": Document("d/B", "B")},
        pages=pages,
        queries={
            "d/q1": Query("d/q1", "?", "en"),
            "d/q2": Query("d/q2", "?", "fr"),
            "d/q3": Query("d/q3", "?", "de"),
        },
        qrels=[Qrel("d/q1", "d/a1", 1), Qrel("d/q1", "d/b1", 2), Qrel("d/q2", "d/b1", 1), Qrel("d/q3", "d/a2", 1)],
        page_locations={pid: (Path("x.parquet"), i) for i, pid in enumerate(pages)},
    )


def test_apply_exclusions_removes_document_pages_and_queries():
    result = apply_exclusions(toy_benchmark(), ["d/B"])
    assert set(result.documents) == {"d/A"}
    assert set(result.pages) == {"d/a1", "d/a2"} == set(result.page_locations)
    # q1 had one relevant page in B: dropped entirely, not kept with fewer pages.
    assert set(result.queries) == {"d/q3"}
    assert result.qrels == [Qrel("d/q3", "d/a2", 1)]


def test_filter_languages_keeps_pages():
    result = filter_languages(toy_benchmark(), ["en", "fr"])
    assert set(result.queries) == {"d/q1", "d/q2"}
    assert {q.query_id for q in result.qrels} == {"d/q1", "d/q2"}
    assert len(result.pages) == 3


def test_clean_qrels_drops_unusable_judgments():
    bench = toy_benchmark()
    bench.qrels += [Qrel("d/q2", "d/a1", 0), Qrel("d/q2", "d/missing", 1)]
    bench.queries["d/q4"] = Query("d/q4", "?", "en")  # no judgment at all
    result = clean_qrels(bench)
    assert len(result.qrels) == 4
    assert "d/q4" not in result.queries
    assert len(result.warnings) == 3


def test_parse_answers():
    assert parse_answers(None) == ()
    assert parse_answers("['Augmentation', 'Augmentation', ' ']") == ("Augmentation",)
    assert parse_answers("Paris") == ("Paris",)
    assert parse_answers("[not a list") == ("[not a list",)
    assert parse_answers(["a", None, "b"]) == ("a", "b")


def test_v1_ids():
    shift = "data/pages_extracted/shift_project_test/Rapport_2021.pdf/page_130.jpg"
    assert v1_ids(shift) == ("Rapport_2021.pdf", "Rapport_2021.pdf/page_130", 130)
    assert v1_ids("/home/user/Downloads/test/2315.png") == ("2315", "2315", None)
