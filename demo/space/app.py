"""Document Q&A demo on a Hugging Face ZeroGPU Space: upload PDFs, ask a question, get an answer citing its pages.

ZeroGPU attaches a GPU only while a `@spaces.GPU` function runs, so those functions take everything they need as
arguments and return their results; each visitor's documents live in their own Gradio session state.
"""

from __future__ import annotations

import html
import inspect
import io
from pathlib import Path

import gradio as gr
import spaces
import yaml
from PIL import Image

from doc_qa.pipeline import answer, document_id, embed_pages, index_pages, load_models, read_document

CFG = yaml.safe_load((Path(__file__).parent / "demo.yaml").read_text(encoding="utf-8"))
MODELS = load_models(CFG)  # moved to "cuda" at start-up; ZeroGPU emulates CUDA until a GPU function runs

INTRO = """# Document Q&A
Ask questions about your PDFs, in English or French. Every answer cites the pages it comes from."""
PLACEHOLDER = "*The answer will appear here, with the pages it comes from.*"

HOW_IT_WORKS = """1. **Search by image.** Each page is read as an image by a visual retriever
(tomoro-colqwen3-embed-4b), so charts, tables and scans count as much as text.
2. **Rerank by text.** The 20 best pages are reread with their text (Qwen3-Reranker-4B), and both orders are merged.
3. **Answer.** Qwen3-VL-8B reads the top 5 pages, image and text, answers only from them, and names the pages it used.

All models are open (Apache 2.0)."""

CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto !important;}
#answer {min-height: 140px; padding: 18px 20px; border-radius: 12px;
         border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);}
#answer p {font-size: 1.05rem; line-height: 1.6;}
.source {display: inline-block; margin: 2px 6px 2px 0; padding: 2px 10px; border-radius: 999px;
         background: var(--color-accent-soft); font-size: 0.9rem; white-space: nowrap;}
.muted {opacity: 0.65; font-size: 0.85rem;}
"""
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
STYLE = {"theme": THEME, "css": CSS}
# Gradio 6 takes the theme and CSS in launch(), earlier versions in Blocks().
STYLE_IN_LAUNCH = "theme" in inspect.signature(gr.Blocks.launch).parameters


def index_seconds(images: list[Image.Image]) -> int:
    """Request GPU time for embedding pages, in proportion to their number.

    Args:
        images (list[Image.Image]): Pages to embed.

    Returns:
        int: Seconds.
    """
    return 20 + len(images)


@spaces.GPU(duration=index_seconds)
def embed(images: list[Image.Image]) -> list:
    """Embed page images on the GPU."""
    return embed_pages(MODELS, CFG, images)


@spaces.GPU(duration=60)
def respond(question: str, pages: list) -> dict:
    """Search, rerank and answer on the GPU."""
    return answer(MODELS, CFG, question, pages)


def documents_markdown(documents: dict, notes: list[str] | None = None) -> str:
    """List the documents in this session, with any upload notes.

    Args:
        documents (dict): Document id -> its indexed pages.
        notes (list[str] | None): Messages from the last upload.

    Returns:
        str: Markdown.
    """
    lines = [f"**{html.escape(pages[0].document)}** · {len(pages)} pages" for pages in documents.values() if pages]
    if not lines:
        lines = ["*No document yet. Upload one or more PDFs.*"]
    return "\n\n".join([*lines, *(f"<span class='muted'>{html.escape(n)}</span>" for n in notes or [])])


def answer_markdown(result: dict) -> str:
    """Format an answer with its sources.

    Args:
        result (dict): Output of `doc_qa.pipeline.answer`.

    Returns:
        str: Markdown (with a little HTML for the source labels).
    """
    if result["answer"] is None:
        return "The model gave no usable answer. Try rephrasing the question."
    text = html.escape(result["answer"])
    s = result["seconds"]
    timing = f"<span class='muted'>search {s['search']} s · rerank {s['rerank']} s · answer {s['answer']} s</span>"
    if not result["found"]:
        return f"{text}\n\n<span class='muted'>Not found in the retrieved pages.</span> {timing}"
    sources = "".join(
        f"<span class='source'>{html.escape(c['document'])} · p. {c['page']}</span>" for c in result["citations"]
    )
    return f"{text}\n\n**Sources** {sources}\n\n{timing}"


def gallery_items(result: dict) -> list[tuple[Image.Image, str]]:
    """List the pages given to the model, cited pages first.

    Args:
        result (dict): Output of `doc_qa.pipeline.answer`.

    Returns:
        list[tuple[Image.Image, str]]: (thumbnail, caption) pairs.
    """
    ordered = sorted(result["pages"], key=lambda p: not p["cited"])  # stable: keeps the retrieval order within groups
    return [
        (Image.open(io.BytesIO(p["jpeg"])), f"{'Cited · ' if p['cited'] else ''}{p['document']} · p. {p['page']}")
        for p in ordered
    ]


def upload(paths: list[str] | None, documents: dict):
    """Read the uploaded PDFs on the CPU, then embed their pages on the GPU."""
    documents = dict(documents or {})
    notes: list[str] = []
    for path in paths or []:
        data, name = Path(path).read_bytes(), Path(path).name
        doc_id = document_id(data)
        if doc_id in documents:
            notes.append(f"{name}: already loaded")
            continue
        yield documents, documents_markdown(documents, [*notes, f"Reading {name}..."])
        try:
            read, total = read_document(data, CFG)
            progress = f"Indexing {len(read)} pages of {name} on the GPU..."
            yield documents, documents_markdown(documents, [*notes, progress])
            vectors = embed([image for _, image, _ in read])
        except Exception as error:  # e.g. not a PDF, too large, GPU quota used up
            notes.append(f"{name}: {error}")
            continue
        documents[doc_id] = index_pages(doc_id, name, read, vectors)
        while len(documents) > CFG["max_documents"]:
            dropped = documents.pop(next(iter(documents)))
            notes.append(f"{dropped[0].document} removed (at most {CFG['max_documents']} documents)")
        if len(read) < total:
            notes.append(f"{name}: only the first {len(read)} of {total} pages were read")
    yield documents, documents_markdown(documents, notes)


def ask(question: str, documents: dict):
    """Answer the question from this session's documents."""
    if not documents:
        yield "Upload a PDF first.", []
        return
    if not question.strip():
        yield "Type a question.", []
        return
    yield "Searching your documents and writing the answer...", []
    pages = [page for doc_pages in documents.values() for page in doc_pages]
    try:
        result = respond(question.strip(), pages)
    except Exception as error:  # e.g. GPU quota used up
        yield f"Something went wrong: {html.escape(str(error))}", []
        return
    yield answer_markdown(result), gallery_items(result)


def clear() -> tuple[dict, str, None, str, list]:
    """Forget this session's documents and answer."""
    return {}, documents_markdown({}), None, PLACEHOLDER, []


with gr.Blocks(title="Document Q&A", **({} if STYLE_IN_LAUNCH else STYLE)) as demo:
    documents = gr.State({})
    gr.Markdown(INTRO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=300):
            files = gr.File(label="1 · Upload PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
            status = gr.Markdown(documents_markdown({}))
            remove = gr.Button("Remove all documents", size="sm", variant="secondary")
            with gr.Accordion("How it works", open=False):
                gr.Markdown(HOW_IT_WORKS)
        with gr.Column(scale=3):
            with gr.Row(equal_height=True):
                question = gr.Textbox(
                    label="2 · Ask a question", placeholder="e.g. What were the total emissions in 2023?", scale=5
                )
                button = gr.Button("Ask", variant="primary", scale=1, min_width=100)
            reply = gr.Markdown(PLACEHOLDER, elem_id="answer")
            pages = gr.Gallery(
                label="Pages read to answer (cited first)", columns=5, height="auto", object_fit="contain"
            )

    files.upload(upload, inputs=[files, documents], outputs=[documents, status])
    remove.click(clear, outputs=[documents, status, files, reply, pages])
    button.click(ask, inputs=[question, documents], outputs=[reply, pages])
    question.submit(ask, inputs=[question, documents], outputs=[reply, pages])

# Server-side rendering left the page unstyled on this Space (its styles and scripts never loaded), so it is off.
demo.queue().launch(ssr_mode=False, **(STYLE if STYLE_IN_LAUNCH else {}))
