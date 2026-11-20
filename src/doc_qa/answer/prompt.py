"""Chat messages for the answer model: labelled page images with their text, then the question.

Messages use the content-list format of `transformers` chat templates ({"type": "text" | "image", ...}).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = """You answer questions about documents, using only the pages you are given.
Each page has a label such as [Page 2], an image of the page, and text extracted from it. The text can contain \
errors; when the text and the image disagree, trust the image.

Rules:
- Use only what these pages show. Do not use outside knowledge.
- Write the answer in the language of the question. Keep it short and specific.
- List the label number of every page you used.
- If the pages do not contain the answer, set "found" to false, leave "pages" empty, and say in the answer that \
the pages do not contain it.

Reply with only this JSON object:
{"answer": "<your answer>", "pages": [<label numbers>], "found": <true or false>}"""

RETRY_PROMPT = "Your reply could not be used: {error}. Reply again with only the JSON object."
NO_TEXT = "(no text)"


@dataclass(frozen=True)
class PageInput:
    """Page given to the answer model."""

    page_id: str
    image: Any
    text: str


def build_messages(question: str, pages: list[PageInput], max_text_chars: int) -> list[dict[str, Any]]:
    """Build the chat for one question: system rules, then each page as [Page i], image and text, then the question.

    Args:
        question (str): Question.
        pages (list[PageInput]): Retrieved pages, best first; their labels are their positions, from 1.
        max_text_chars (int): Characters of text kept per page; the end of longer texts is cut.

    Returns:
        list[dict[str, Any]]: Chat messages.
    """
    content: list[dict[str, Any]] = []
    for label, page in enumerate(pages, 1):
        text = page.text.strip()[:max_text_chars] or NO_TEXT
        content += [
            {"type": "text", "text": f"[Page {label}]"},
            {"type": "image", "image": page.image},
            {"type": "text", "text": f"Text of page {label}:\n{text}"},
        ]
    content.append({"type": "text", "text": f"Question: {question}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def retry_messages(messages: list[dict[str, Any]], reply: str, error: str) -> list[dict[str, Any]]:
    """Add a rejected reply and the reason it was rejected, so a greedy model does not repeat it.

    Args:
        messages (list[dict[str, Any]]): Chat so far.
        reply (str): The model's rejected reply.
        error (str): Why it was rejected.

    Returns:
        list[dict[str, Any]]: The chat with the reply and a request to correct it.
    """
    return [
        *messages,
        {"role": "assistant", "content": [{"type": "text", "text": reply}]},
        {"role": "user", "content": [{"type": "text", "text": RETRY_PROMPT.format(error=error)}]},
    ]
