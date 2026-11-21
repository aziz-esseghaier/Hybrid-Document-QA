"""Web demo on Modal: upload PDFs, ask a question, get an answer with the pages it comes from.

From the repo root, in the environment where `doc_qa` is installed (plus `pip install modal`, then `modal setup`):
    modal run demo/modal_app.py::download_weights   # once: models into a Volume, on a CPU machine
    modal serve demo/modal_app.py    # temporary URL while the command runs, reloads on code changes
    modal deploy demo/modal_app.py   # lasting URL

One L40S GPU holds the three models (about 35 GB in fp16). It scales to zero after two idle minutes, so it costs only
while used, and at most one GPU runs at a time. Model weights are kept in a Modal Volume, so they download once.
Uploaded PDFs stay in the GPU container's memory only; the web page sends them again if the container restarted.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "/root/demo.yaml"
WEIGHTS = "/weights"

GPU_PACKAGES = [  # versions of the Kaggle runs the models were checked with
    "torch==2.11.0",
    "torchvision==0.26.0",
    "transformers==5.20.2",
    "accelerate==1.15.0",
    "safetensors==0.8.0",
    "tokenizers==0.23.3",
    "huggingface_hub==1.33.0",
    "hf-xet==1.7.0",
    "einops==0.8.2",
    "numpy==2.1.3",
    "pillow==13.0.0",
    "scipy==1.16.3",
    "pypdfium2==5.13.0",
    "PyYAML==6.0.3",
    "python-dotenv==1.2.4",
]
UI_PACKAGES = ["gradio==6.30.0", "fastapi==0.141.1", "pillow==13.0.0"]

weights = modal.Volume.from_name("doc-qa-weights", create_if_missing=True)
gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(*GPU_PACKAGES)
    .env({"HF_HOME": f"{WEIGHTS}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(ROOT / "configs" / "demo.yaml", CONFIG)
    .add_local_python_source("doc_qa")
)
ui_image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(*UI_PACKAGES)
app = modal.App("doc-qa-demo")


@app.function(image=gpu_image, volumes={WEIGHTS: weights}, timeout=60 * 60)
def download_weights() -> None:
    """Download the three models into the Volume on a CPU machine, so no GPU time is spent downloading.

    Run once: modal run demo/modal_app.py::download_weights
    """
    import yaml
    from huggingface_hub import snapshot_download

    cfg = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    for part in ("visual", "reranker", "answer"):
        print(snapshot_download(cfg[part]["model"], revision=cfg[part]["revision"]), flush=True)
    weights.commit()


@app.cls(
    gpu="L40S",
    image=gpu_image,
    volumes={WEIGHTS: weights},
    timeout=15 * 60,
    scaledown_window=120,
    max_containers=1,
)
class DocQA:
    """The question-answering pipeline, loaded once per GPU container."""

    @modal.enter()
    def load(self) -> None:
        """Load the models (downloading them into the Volume the first time)."""
        import yaml

        from doc_qa.pipeline import DocumentQA

        self.qa = DocumentQA(yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8")))
        weights.commit()

    @modal.method()
    def add_pdf(self, data: bytes, name: str) -> dict:
        """Index a PDF.

        Args:
            data (bytes): PDF file.
            name (str): File name.

        Returns:
            dict: See `DocumentQA.add_pdf`.
        """
        return self.qa.add_pdf(data, name)

    @modal.method()
    def ask(self, question: str, documents: list[tuple[bytes, str]]) -> dict:
        """Answer a question from the given PDFs, indexing any this container does not hold yet.

        Args:
            question (str): Question.
            documents (list[tuple[bytes, str]]): (PDF file, name) pairs.

        Returns:
            dict: See `DocumentQA.ask`.
        """
        doc_ids = [self.qa.add_pdf(data, name)["doc_id"] for data, name in documents]
        return self.qa.ask(question, doc_ids)


def answer_markdown(result: dict) -> str:
    """Format an answer and its citations for the page.

    Args:
        result (dict): Output of `DocQA.ask`.

    Returns:
        str: Markdown.
    """
    if result["answer"] is None:
        return "The model gave no usable answer. Try rephrasing the question."
    lines = [result["answer"], ""]
    if result["found"]:
        sources = ", ".join(f"{c['document']} p. {c['page']}" for c in result["citations"])
        lines.append(f"**Sources:** {sources}")
    else:
        lines.append("*Not found in the pages retrieved.*")
    s = result["seconds"]
    lines.append(f"\n<sub>search {s['search']} s, rerank {s['rerank']} s, answer {s['answer']} s</sub>")
    return "\n".join(lines)


@app.function(image=ui_image, max_containers=1)  # one container: Gradio keeps session state in memory
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Serve the Gradio page."""
    import io

    import gradio as gr
    from fastapi import FastAPI
    from PIL import Image

    qa = DocQA()

    def upload(paths: list[str] | None):
        """Read and index the uploaded PDFs, reporting progress."""
        documents, lines = [], []
        if not paths:
            yield [], "No PDF uploaded."
            return
        yield [], "Indexing... The first request starts the GPU, which can take a few minutes."
        for path in paths:
            data, name = Path(path).read_bytes(), Path(path).name
            try:
                info = qa.add_pdf.remote(data, name)
            except Exception as error:  # shown to the user rather than as a server error
                lines.append(f"{name}: {error}")
                continue
            documents.append((data, name))
            total = info["total_pages"]
            extra = f" (first {info['pages']} of {total})" if total and info["pages"] < total else ""
            lines.append(f"{name}: {info['pages']} pages indexed{extra} in {info['seconds']} s")
        yield documents, "\n".join(lines)

    def ask(question: str, documents: list):
        """Answer the question from the uploaded PDFs."""
        if not documents:
            return "Upload a PDF first.", []
        if not question.strip():
            return "Type a question.", []
        try:
            result = qa.ask.remote(question.strip(), documents)
        except Exception as error:
            return f"Something went wrong: {error}", []
        gallery = [
            (Image.open(io.BytesIO(p["jpeg"])), f"{p['document']} p. {p['page']}{' (cited)' if p['cited'] else ''}")
            for p in result["pages"]
        ]
        return answer_markdown(result), gallery

    with gr.Blocks(title="Document Q&A") as blocks:
        gr.Markdown(
            "# Document Q&A\nUpload English or French PDFs and ask a question. The answer cites its pages.\n\n"
            "Preliminary demo: answers are not checked yet, uploads are kept only while the server is awake, and "
            "only the first 50 pages of each PDF are read. Don't upload confidential files."
        )
        documents = gr.State([])
        files = gr.File(label="PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
        status = gr.Markdown()
        question = gr.Textbox(label="Question", placeholder="e.g. What were the total emissions in 2023?")
        button = gr.Button("Ask", variant="primary")
        answer = gr.Markdown()
        pages = gr.Gallery(label="Pages given to the model", columns=5)

        files.change(upload, inputs=files, outputs=[documents, status])
        button.click(ask, inputs=[question, documents], outputs=[answer, pages])
        question.submit(ask, inputs=[question, documents], outputs=[answer, pages])

    return gr.mount_gradio_app(FastAPI(), blocks, path="/")
