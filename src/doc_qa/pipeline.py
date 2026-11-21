"""Question answering over uploaded PDFs: visual search, text reranking, and an answer that cites its pages.

Documents stay in memory, which suits a demo. Loading the models needs `torch` and `transformers`.
"""

from __future__ import annotations

import hashlib
import io
import tempfile
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from doc_qa.ingest.pdf import clean_pdf_text, read_pdf, word_chars
from doc_qa.retrieval.fusion import rrf


@dataclass(frozen=True)
class IndexedPage:
    """A page held in memory, with what search and answering need."""

    page_id: str
    document: str
    number: int  # from 1
    image: Image.Image
    text: str
    vectors: np.ndarray


def document_id(data: bytes) -> str:
    """Name a document by its content, so the same file is indexed once.

    Args:
        data (bytes): PDF file.

    Returns:
        str: First 16 hex digits of its SHA-256.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def page_text(raw: str, min_word_chars: int) -> str:
    """Clean a page's text layer, or drop it when there is too little to be useful.

    Args:
        raw (str): Raw text layer.
        min_word_chars (int): Letters and digits needed to keep the text.

    Returns:
        str: Cleaned text, or "" for image-only pages.
    """
    text = clean_pdf_text(raw)
    return text if word_chars(text) >= min_word_chars else ""


def blend(
    visual: list[str], rerank_scores: dict[str, float], visual_weight: float, rerank_weight: float, k: int
) -> list[str]:
    """Order the reranked pages by score (ties keep the visual order), then merge with the visual ranking by RRF.

    Args:
        visual (list[str]): Page ids in visual order.
        rerank_scores (dict[str, float]): Reranker score of the top visual pages.
        visual_weight (float): RRF weight of the visual ranking.
        rerank_weight (float): RRF weight of the reranker order.
        k (int): RRF damping constant.

    Returns:
        list[str]: Page ids, best first.
    """
    reranked = sorted((p for p in visual if p in rerank_scores), key=lambda p: -rerank_scores[p])
    return [page for page, _ in rrf([visual, reranked], k, None, [visual_weight, rerank_weight])]


def jpeg_bytes(image: Image.Image, max_side: int) -> bytes:
    """Encode a smaller copy of a page image for display.

    Args:
        image (Image.Image): Page image.
        max_side (int): Longest side in pixels.

    Returns:
        bytes: JPEG data.
    """
    small = image.copy()
    small.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    small.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class DocumentQA:
    """The three models on one GPU, and the uploaded documents in memory."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Load the visual retriever, the reranker and the answer model.

        Args:
            cfg (dict[str, Any]): Demo config (configs/demo.yaml).
        """
        from doc_qa.answer.model import load_answer_model
        from doc_qa.retrieval.rerank import load_reranker
        from doc_qa.retrieval.visual import load_visual_model

        self.cfg = cfg
        visual, reranker, answer = cfg["visual"], cfg["reranker"], cfg["answer"]
        self.visual, self.visual_processor, _ = load_visual_model(
            visual["model"], visual["revision"], cfg["dtype"], cfg["device"], visual["attn_implementation"]
        )
        self.reranker, self.rerank_tokenizer = load_reranker(
            reranker["model"], reranker["revision"], cfg["dtype"], cfg["device"], reranker["attn_implementation"]
        )
        self.answerer, self.answer_processor = load_answer_model(
            answer["model"], answer["revision"], cfg["dtype"], answer["attn_implementation"], answer.get("max_memory")
        )
        self.documents: dict[str, list[IndexedPage]] = {}  # oldest first

    def add_pdf(self, data: bytes, name: str) -> dict[str, Any]:
        """Render, read and embed a PDF's pages; a file already in memory is not redone.

        Args:
            data (bytes): PDF file.
            name (str): File name, shown in citations.

        Returns:
            dict[str, Any]: {doc_id, name, pages (indexed), total_pages, seconds}.
        """
        import pypdfium2 as pdfium

        from doc_qa.retrieval.visual import embed_images

        doc_id = document_id(data)
        if doc_id in self.documents:
            pages = self.documents[doc_id]
            return {"doc_id": doc_id, "name": name, "pages": len(pages), "total_pages": None, "seconds": 0.0}
        if len(data) > self.cfg["max_file_mb"] * 1e6:
            raise ValueError(f"{name} is larger than {self.cfg['max_file_mb']} MB.")

        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "upload.pdf"
            path.write_bytes(data)
            pdf = pdfium.PdfDocument(str(path))
            total = len(pdf)
            pdf.close()
            reader = read_pdf(path, self.cfg["render_scale"], color=True)
            read = list(islice(reader, self.cfg["max_pages_per_pdf"]))
            reader.close()  # closes the PDF when the page limit stopped the reading early
        images = [image for _, image, _ in read]
        vectors = embed_images(self.visual, self.visual_processor, images, self.cfg["visual"]["batch_size"])
        min_chars = self.cfg["min_word_chars"]
        self.documents[doc_id] = [
            IndexedPage(f"{doc_id}/{index + 1}", name, index + 1, image, page_text(raw, min_chars), page_vectors)
            for (index, image, raw), page_vectors in zip(read, vectors)
        ]
        while len(self.documents) > self.cfg["max_documents"]:
            self.documents.pop(next(iter(self.documents)))
        seconds = round(time.perf_counter() - start, 1)
        return {"doc_id": doc_id, "name": name, "pages": len(read), "total_pages": total, "seconds": seconds}

    def ask(self, question: str, doc_ids: list[str]) -> dict[str, Any]:
        """Find the pages that answer a question and answer it from them.

        Args:
            question (str): Question, in English or French.
            doc_ids (list[str]): Documents to search, from `add_pdf`.

        Returns:
            dict[str, Any]: {answer, found, citations, pages (given to the model, with JPEG thumbnails), attempts,
                seconds}; answer is None when the model gave no usable reply.
        """
        from doc_qa.answer.model import answer_question, fit_pixels
        from doc_qa.answer.output import cited_page_ids
        from doc_qa.answer.prompt import PageInput
        from doc_qa.retrieval.rerank import pair_ids, score_pairs
        from doc_qa.retrieval.visual import embed_queries, maxsim_matrix

        pages = {page.page_id: page for doc_id in doc_ids for page in self.documents.get(doc_id, [])}
        if not pages:
            raise ValueError("No document to search: upload a PDF first.")
        reranker, fusion, answer = self.cfg["reranker"], self.cfg["fusion"], self.cfg["answer"]
        timings: dict[str, float] = {}

        start = time.perf_counter()
        [query] = embed_queries(self.visual, self.visual_processor, [question], 1)
        ids = list(pages)
        scores = maxsim_matrix([query], [pages[p].vectors for p in ids])[0]
        visual = [ids[i] for i in np.argsort(-scores, kind="stable")]
        timings["search"] = round(time.perf_counter() - start, 2)

        start = time.perf_counter()
        candidates = visual[: reranker["candidates"]]
        prompts = [
            pair_ids(self.rerank_tokenizer, reranker["instruction"], question, pages[p].text, reranker["max_tokens"])
            for p in candidates
        ]
        rerank_scores = score_pairs(self.reranker, self.rerank_tokenizer, prompts, reranker["batch_size"])
        ranked = blend(
            visual, dict(zip(candidates, rerank_scores.tolist())), fusion["visual_weight"], fusion["rerank_weight"],
            fusion["rrf_k"],
        )
        timings["rerank"] = round(time.perf_counter() - start, 2)

        start = time.perf_counter()
        given = [pages[p] for p in ranked[: answer["pages"]]]
        inputs = [PageInput(p.page_id, fit_pixels(p.image, answer["page_tokens"]), p.text) for p in given]
        result, _ = answer_question(
            self.answerer, self.answer_processor, question, inputs, answer["max_text_chars"], answer["max_new_tokens"],
            answer["retries"], answer.get("sdpa_backend"),
        )
        timings["answer"] = round(time.perf_counter() - start, 2)

        cited = cited_page_ids(result.answer, [p.page_id for p in given]) if result.answer else []
        return {
            "answer": result.answer.text if result.answer else None,
            "found": result.answer.found if result.answer else None,
            "citations": [{"document": pages[p].document, "page": pages[p].number} for p in cited],
            "pages": [
                {
                    "document": p.document,
                    "page": p.number,
                    "cited": p.page_id in cited,
                    "jpeg": jpeg_bytes(p.image, self.cfg["thumbnail_side"]),
                }
                for p in given
            ],
            "attempts": result.attempts,
            "seconds": timings,
        }
