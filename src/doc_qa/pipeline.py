"""Question answering over uploaded PDFs: visual search, text reranking, and an answer that cites its pages.

The steps are plain functions over `Models` and `IndexedPage` lists, so a host can keep the documents wherever suits it
(a web session, a server's memory). `DocumentQA` keeps them in memory. Loading the models needs `torch` and
`transformers`.
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
    """A searchable page: lossless image, text layer and visual vectors."""

    page_id: str
    document: str
    number: int  # from 1
    png: bytes
    text: str
    vectors: np.ndarray  # float16 (tokens, dim)

    def image(self) -> Image.Image:
        """Decode the page image.

        Returns:
            Image.Image: RGB image.
        """
        return Image.open(io.BytesIO(self.png)).convert("RGB")


@dataclass
class Models:
    """The three loaded models and their processors."""

    visual: Any
    visual_processor: Any
    reranker: Any
    rerank_tokenizer: Any
    answerer: Any
    answer_processor: Any


def load_models(cfg: dict[str, Any]) -> Models:
    """Load the visual retriever, the reranker and the answer model on `cfg["device"]`.

    Args:
        cfg (dict[str, Any]): Demo config (configs/demo.yaml).

    Returns:
        Models: Loaded models.
    """
    from doc_qa.answer.model import load_answer_model
    from doc_qa.retrieval.rerank import load_reranker
    from doc_qa.retrieval.visual import load_visual_model

    visual, reranker, answer = cfg["visual"], cfg["reranker"], cfg["answer"]
    visual_model, visual_processor, _ = load_visual_model(
        visual["model"], visual["revision"], cfg["dtype"], cfg["device"], visual["attn_implementation"]
    )
    rerank_model, rerank_tokenizer = load_reranker(
        reranker["model"], reranker["revision"], cfg["dtype"], cfg["device"], reranker["attn_implementation"]
    )
    answer_model, answer_processor = load_answer_model(
        answer["model"], answer["revision"], cfg["dtype"], answer["attn_implementation"], device=cfg["device"]
    )
    return Models(visual_model, visual_processor, rerank_model, rerank_tokenizer, answer_model, answer_processor)


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


def png_bytes(image: Image.Image) -> bytes:
    """Encode a page image losslessly.

    Args:
        image (Image.Image): Page image.

    Returns:
        bytes: PNG data.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def read_document(data: bytes, cfg: dict[str, Any]) -> tuple[list[tuple[int, Image.Image, str]], int]:
    """Render a PDF's first pages and read their text layers (CPU only).

    Args:
        data (bytes): PDF file.
        cfg (dict[str, Any]): Demo config: `max_file_mb`, `max_pages_per_pdf`, `render_scale`, `min_word_chars`.

    Returns:
        tuple[list[tuple[int, Image.Image, str]], int]: (page number from 1, image, text) for the pages kept, and the
            PDF's page count.
    """
    import pypdfium2 as pdfium

    if len(data) > cfg["max_file_mb"] * 1e6:
        raise ValueError(f"The file is larger than {cfg['max_file_mb']} MB.")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "upload.pdf"
        path.write_bytes(data)
        pdf = pdfium.PdfDocument(str(path))
        total = len(pdf)
        pdf.close()
        reader = read_pdf(path, cfg["render_scale"], color=True)
        read = list(islice(reader, cfg["max_pages_per_pdf"]))
        reader.close()  # closes the PDF when the page limit stopped the reading early
    return [(index + 1, image, page_text(raw, cfg["min_word_chars"])) for index, image, raw in read], total


def embed_pages(models: Models, cfg: dict[str, Any], images: list[Image.Image]) -> list[np.ndarray]:
    """Embed page images with the visual retriever (GPU).

    Args:
        models (Models): Loaded models.
        cfg (dict[str, Any]): Demo config.
        images (list[Image.Image]): Page images.

    Returns:
        list[np.ndarray]: float16 (tokens, dim) vectors per page.
    """
    from doc_qa.retrieval.visual import embed_images

    vectors = embed_images(models.visual, models.visual_processor, images, cfg["visual"]["batch_size"])
    return [v.astype(np.float16) for v in vectors]


def index_pages(
    doc_id: str, name: str, pages: list[tuple[int, Image.Image, str]], vectors: list[np.ndarray]
) -> list[IndexedPage]:
    """Pack read and embedded pages for search.

    Args:
        doc_id (str): Document id.
        name (str): File name, shown in citations.
        pages (list[tuple[int, Image.Image, str]]): Output of `read_document`.
        vectors (list[np.ndarray]): Output of `embed_pages`, same order.

    Returns:
        list[IndexedPage]: Pages.
    """
    return [
        IndexedPage(f"{doc_id}/{number}", name, number, png_bytes(image), text, page_vectors)
        for (number, image, text), page_vectors in zip(pages, vectors)
    ]


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


def answer(models: Models, cfg: dict[str, Any], question: str, pages: list[IndexedPage]) -> dict[str, Any]:
    """Find the pages that answer a question, and answer it from them (GPU).

    Args:
        models (Models): Loaded models.
        cfg (dict[str, Any]): Demo config.
        question (str): Question, in English or French.
        pages (list[IndexedPage]): Pages to search.

    Returns:
        dict[str, Any]: {answer, found, citations, pages (given to the model, with JPEG thumbnails), attempts,
            seconds}; answer is None when the model gave no usable reply.
    """
    from doc_qa.answer.model import answer_question, fit_pixels
    from doc_qa.answer.output import cited_page_ids
    from doc_qa.answer.prompt import PageInput
    from doc_qa.retrieval.rerank import pair_ids, score_pairs
    from doc_qa.retrieval.visual import embed_queries, maxsim_matrix

    if not pages:
        raise ValueError("No document to search: upload a PDF first.")
    by_id = {page.page_id: page for page in pages}
    reranker, fusion, settings = cfg["reranker"], cfg["fusion"], cfg["answer"]
    timings: dict[str, float] = {}

    start = time.perf_counter()
    [query] = embed_queries(models.visual, models.visual_processor, [question], 1)
    ids = list(by_id)
    scores = maxsim_matrix([query], [by_id[p].vectors for p in ids])[0]
    visual = [ids[i] for i in np.argsort(-scores, kind="stable")]
    timings["search"] = round(time.perf_counter() - start, 2)

    start = time.perf_counter()
    candidates = visual[: reranker["candidates"]]
    prompts = [
        pair_ids(models.rerank_tokenizer, reranker["instruction"], question, by_id[p].text, reranker["max_tokens"])
        for p in candidates
    ]
    rerank_scores = score_pairs(models.reranker, models.rerank_tokenizer, prompts, reranker["batch_size"])
    ranked = blend(
        visual, dict(zip(candidates, rerank_scores.tolist())), fusion["visual_weight"], fusion["rerank_weight"],
        fusion["rrf_k"],
    )
    timings["rerank"] = round(time.perf_counter() - start, 2)

    start = time.perf_counter()
    given = [by_id[p] for p in ranked[: settings["pages"]]]
    images = {p.page_id: p.image() for p in given}
    inputs = [PageInput(p.page_id, fit_pixels(images[p.page_id], settings["page_tokens"]), p.text) for p in given]
    result, _ = answer_question(
        models.answerer, models.answer_processor, question, inputs, settings["max_text_chars"],
        settings["max_new_tokens"], settings["retries"], settings.get("sdpa_backend"),
    )
    timings["answer"] = round(time.perf_counter() - start, 2)

    cited = cited_page_ids(result.answer, [p.page_id for p in given]) if result.answer else []
    return {
        "answer": result.answer.text if result.answer else None,
        "found": result.answer.found if result.answer else None,
        "citations": [{"document": by_id[p].document, "page": by_id[p].number} for p in cited],
        "pages": [
            {
                "document": p.document,
                "page": p.number,
                "cited": p.page_id in cited,
                "jpeg": jpeg_bytes(images[p.page_id], cfg["thumbnail_side"]),
            }
            for p in given
        ],
        "attempts": result.attempts,
        "seconds": timings,
    }


class DocumentQA:
    """The models plus the uploaded documents in this process's memory (one server)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Load the models.

        Args:
            cfg (dict[str, Any]): Demo config (configs/demo.yaml).
        """
        self.cfg = cfg
        self.models = load_models(cfg)
        self.documents: dict[str, list[IndexedPage]] = {}  # oldest first

    def add_pdf(self, data: bytes, name: str) -> dict[str, Any]:
        """Read and embed a PDF's pages; a file already in memory is not redone.

        Args:
            data (bytes): PDF file.
            name (str): File name, shown in citations.

        Returns:
            dict[str, Any]: {doc_id, name, pages (indexed), total_pages, seconds}.
        """
        doc_id = document_id(data)
        if doc_id in self.documents:
            pages = self.documents[doc_id]
            return {"doc_id": doc_id, "name": name, "pages": len(pages), "total_pages": None, "seconds": 0.0}
        start = time.perf_counter()
        read, total = read_document(data, self.cfg)
        vectors = embed_pages(self.models, self.cfg, [image for _, image, _ in read])
        self.documents[doc_id] = index_pages(doc_id, name, read, vectors)
        while len(self.documents) > self.cfg["max_documents"]:
            self.documents.pop(next(iter(self.documents)))
        seconds = round(time.perf_counter() - start, 1)
        return {"doc_id": doc_id, "name": name, "pages": len(read), "total_pages": total, "seconds": seconds}

    def ask(self, question: str, doc_ids: list[str]) -> dict[str, Any]:
        """Answer a question from documents in memory.

        Args:
            question (str): Question.
            doc_ids (list[str]): Documents to search, from `add_pdf`.

        Returns:
            dict[str, Any]: See `answer`.
        """
        pages = [page for doc_id in doc_ids for page in self.documents.get(doc_id, [])]
        return answer(self.models, self.cfg, question, pages)
