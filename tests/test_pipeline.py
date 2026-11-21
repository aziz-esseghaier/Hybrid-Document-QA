"""Tests for the demo pipeline: helpers, config consistency, and the flow with fake models."""

import io
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
import yaml
from PIL import Image

import doc_qa.answer.model as answer_model
import doc_qa.retrieval.rerank as rerank
import doc_qa.retrieval.visual as visual
from doc_qa import pipeline
from doc_qa.answer.generate import AnswerResult
from doc_qa.answer.output import Answer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load(name):
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_demo_uses_the_evaluated_models_and_settings():
    demo = load("demo.yaml")
    for part, config in (("visual", "visual_tomoro.yaml"), ("reranker", "rerank.yaml"), ("answer", "answer.yaml")):
        evaluated = load(config)
        assert (demo[part]["model"], demo[part]["revision"]) == (evaluated["model"], evaluated["revision"])
    reranker, fusion = load("rerank.yaml"), load("fusion_rerank_tomoro_top20.yaml")
    assert demo["reranker"]["instruction"] == reranker["instruction"]
    assert demo["reranker"]["max_tokens"] == reranker["max_tokens"]
    assert demo["reranker"]["candidates"] == fusion["restrict"]["rerank"][1]
    assert demo["fusion"]["rrf_k"] == fusion["rrf_k"]
    answer = load("answer.yaml")
    for key in ("max_text_chars", "max_new_tokens", "retries", "attn_implementation", "sdpa_backend"):
        assert demo["answer"][key] == answer[key]


def test_helpers():
    assert pipeline.document_id(b"abc") == pipeline.document_id(b"abc") != pipeline.document_id(b"abd")
    assert pipeline.page_text("Hello  world\r\n" * 10, 50).startswith("Hello world")
    assert pipeline.page_text("p. 3", 50) == ""
    jpeg = pipeline.jpeg_bytes(Image.new("RGB", (2000, 1000)), 500)
    assert Image.open(io.BytesIO(jpeg)).size == (500, 250)


def test_blend_reorders_the_candidates_and_keeps_the_rest_in_visual_order():
    visual_order = ["a", "b", "c", "d"]
    ranked = pipeline.blend(visual_order, {"a": 0.0, "b": 5.0, "c": 1.0}, 1, 1, 60)
    assert ranked[0] == "b" and ranked[-1] == "d"
    assert pipeline.blend(visual_order, {"a": 0.0, "b": 5.0, "c": 1.0}, 1, 0, 60) == visual_order


def make_pdf(pages):
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(300, 400)
    buffer = io.BytesIO()
    pdf.save(buffer)
    pdf.close()
    return buffer.getvalue()


@pytest.fixture
def qa(monkeypatch):
    """A DocumentQA with fake models: page i gets vector e_(i mod 4)."""
    cfg = load("demo.yaml")
    cfg["max_pages_per_pdf"], cfg["render_scale"] = 3, 0.5
    instance = object.__new__(pipeline.DocumentQA)
    instance.cfg, instance.documents = cfg, {}
    instance.visual = instance.visual_processor = instance.reranker = instance.rerank_tokenizer = None
    instance.answerer = instance.answer_processor = None

    def embed_images(model, processor, images, batch_size):
        return [np.eye(4)[[i % 4]] for i in range(len(images))]

    monkeypatch.setattr(visual, "embed_images", embed_images)
    monkeypatch.setattr(visual, "embed_queries", lambda *args: [np.eye(4)[[2]]])  # matches page 3
    monkeypatch.setattr(rerank, "pair_ids", lambda tokenizer, instruction, question, text, limit: [1])
    monkeypatch.setattr(rerank, "score_pairs", lambda model, tokenizer, prompts, batch: np.arange(len(prompts), 0, -1.0))
    monkeypatch.setattr(
        answer_model, "answer_question",
        lambda *args: (AnswerResult(Answer("Forty-two.", (1,), True), 1, ()), []),
    )
    return instance


def test_flow_indexes_searches_and_cites(qa):
    info = qa.add_pdf(make_pdf(5), "report.pdf")
    assert (info["pages"], info["total_pages"], info["name"]) == (3, 5, "report.pdf")  # page limit
    assert qa.add_pdf(make_pdf(5), "report.pdf")["seconds"] == 0.0  # already indexed
    result = qa.ask("What?", [info["doc_id"]])
    assert result["answer"] == "Forty-two." and result["found"]
    assert result["citations"] == [{"document": "report.pdf", "page": 3}]  # best visual page, label 1
    assert [p["cited"] for p in result["pages"]] == [True, False, False]
    assert all(Image.open(io.BytesIO(p["jpeg"])) for p in result["pages"])


def test_oldest_document_is_dropped(qa):
    ids = [qa.add_pdf(make_pdf(n), f"{n}.pdf")["doc_id"] for n in range(1, 5)]
    assert list(qa.documents) == ids[1:]  # max_documents is 3
    with pytest.raises(ValueError, match="upload a PDF"):
        qa.ask("What?", ids[:1])
