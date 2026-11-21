"""Document Q&A demo on a Hugging Face ZeroGPU Space: upload PDFs, ask a question, get an answer citing its pages.

ZeroGPU attaches a GPU only while a `@spaces.GPU` function runs, so those functions take everything they need as
arguments and return their results; each visitor's documents live in their own Gradio session state.
"""

from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
import spaces
import yaml
from PIL import Image

from doc_qa.pipeline import answer, document_id, embed_pages, index_pages, load_models, read_document

CFG = yaml.safe_load((Path(__file__).parent / "demo.yaml").read_text(encoding="utf-8"))
MODELS = load_models(CFG)  # moved to "cuda" at start-up; ZeroGPU emulates CUDA until a GPU function runs


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


def answer_markdown(result: dict) -> str:
    """Format an answer and its citations.

    Args:
        result (dict): Output of `doc_qa.pipeline.answer`.

    Returns:
        str: Markdown.
    """
    if result["answer"] is None:
        return "The model gave no usable answer. Try rephrasing the question."
    lines = [result["answer"], ""]
    if result["found"]:
        lines.append("**Sources:** " + ", ".join(f"{c['document']} p. {c['page']}" for c in result["citations"]))
    else:
        lines.append("*Not found in the pages retrieved.*")
    s = result["seconds"]
    lines.append(f"\n<sub>search {s['search']} s, rerank {s['rerank']} s, answer {s['answer']} s</sub>")
    return "\n".join(lines)


def upload(paths: list[str] | None, documents: dict):
    """Read the uploaded PDFs on the CPU, then embed their pages on the GPU."""
    documents = dict(documents or {})
    if not paths:
        yield documents, "No PDF uploaded."
        return
    lines = []
    for path in paths:
        data, name = Path(path).read_bytes(), Path(path).name
        doc_id = document_id(data)
        if doc_id in documents:
            lines.append(f"{name}: already indexed")
            continue
        yield documents, "\n".join([*lines, f"{name}: reading..."])
        try:
            read, total = read_document(data, CFG)
            yield documents, "\n".join([*lines, f"{name}: embedding {len(read)} pages on the GPU..."])
            vectors = embed([image for _, image, _ in read])
        except Exception as error:  # e.g. not a PDF, too large, GPU quota used up
            lines.append(f"{name}: {error}")
            continue
        documents[doc_id] = index_pages(doc_id, name, read, vectors)
        while len(documents) > CFG["max_documents"]:
            documents.pop(next(iter(documents)))
        extra = f" (first {len(read)} of {total})" if len(read) < total else ""
        lines.append(f"{name}: {len(read)} pages indexed{extra}")
    yield documents, "\n".join(lines)


def ask(question: str, documents: dict):
    """Answer the question from this session's documents."""
    if not documents:
        return "Upload a PDF first.", []
    if not question.strip():
        return "Type a question.", []
    pages = [page for doc_pages in documents.values() for page in doc_pages]
    try:
        result = respond(question.strip(), pages)
    except Exception as error:
        return f"Something went wrong: {error}", []
    gallery = [
        (Image.open(io.BytesIO(p["jpeg"])), f"{p['document']} p. {p['page']}{' (cited)' if p['cited'] else ''}")
        for p in result["pages"]
    ]
    return answer_markdown(result), gallery


with gr.Blocks(title="Document Q&A") as demo:
    gr.Markdown(
        "# Document Q&A\nUpload English or French PDFs and ask a question. The answer cites its pages.\n\n"
        f"Preliminary demo: answers are not checked yet, and only the first {CFG['max_pages_per_pdf']} pages of each "
        "PDF are read. Logged-in Hugging Face users get more GPU time per day. Don't upload confidential files."
    )
    documents = gr.State({})
    files = gr.File(label="PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
    status = gr.Markdown()
    question = gr.Textbox(label="Question", placeholder="e.g. What were the total emissions in 2023?")
    button = gr.Button("Ask", variant="primary")
    reply = gr.Markdown()
    pages = gr.Gallery(label="Pages given to the model", columns=5)

    files.change(upload, inputs=[files, documents], outputs=[documents, status])
    button.click(ask, inputs=[question, documents], outputs=[reply, pages])
    question.submit(ask, inputs=[question, documents], outputs=[reply, pages])

demo.launch()
