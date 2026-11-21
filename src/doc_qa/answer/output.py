"""Parse and check the answer model's JSON reply."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_LABEL = re.compile(r"\s*\[?\s*(?:page\s*)?(\d+)\s*\]?\s*", re.IGNORECASE)  # 3, "3", "Page 3", "[Page 3]"


def page_label(value: object) -> int | None:
    """Read one cited page label, as a number or as text such as "Page 3".

    Args:
        value (object): One item of the reply's "pages" list.

    Returns:
        int | None: The label, or None if the item is not a label.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and (match := _LABEL.fullmatch(value)):
        return int(match.group(1))
    return None


class InvalidAnswer(ValueError):
    """A reply that cannot be used; the message says why, for the retry prompt."""


@dataclass(frozen=True)
class Answer:
    """Checked answer. `pages` are the cited page labels (positions in the prompt, from 1)."""

    text: str
    pages: tuple[int, ...]
    found: bool


def extract_json(reply: str) -> str:
    """Return the JSON object in a reply, skipping reasoning blocks, code fences and text around it.

    Args:
        reply (str): Raw model output.

    Returns:
        str: Text from the first "{" to the last "}".
    """
    reply = _THINK.sub("", reply)
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise InvalidAnswer("it contains no JSON object")
    return reply[start : end + 1]


def parse_answer(reply: str, page_count: int) -> Answer:
    """Parse a reply and check it against the rules of the prompt.

    Args:
        reply (str): Raw model output.
        page_count (int): Pages in the prompt; valid labels are 1 to page_count.

    Returns:
        Answer: The answer, with duplicate labels removed (first citation order kept).
    """
    try:
        data = json.loads(extract_json(reply))
    except json.JSONDecodeError as error:
        raise InvalidAnswer(f"the JSON is malformed ({error.msg})") from None
    if not isinstance(data, dict):
        raise InvalidAnswer("the JSON is not an object")
    missing = [key for key in ("answer", "pages", "found") if key not in data]
    if missing:
        raise InvalidAnswer(f"keys missing: {', '.join(missing)}")
    text, pages, found = data["answer"], data["pages"], data["found"]
    if not isinstance(text, str):
        raise InvalidAnswer('"answer" must be a string')
    if not isinstance(found, bool):
        raise InvalidAnswer('"found" must be true or false')
    labels = [page_label(p) for p in pages] if isinstance(pages, list) else [None]
    if None in labels:
        raise InvalidAnswer('"pages" must be a list of label numbers, e.g. [1, 3]')
    pages = labels
    unknown = [p for p in pages if not 1 <= p <= page_count]
    if unknown:
        raise InvalidAnswer(f"pages {unknown} do not exist; labels go from 1 to {page_count}")
    if found and not text.strip():
        raise InvalidAnswer('"answer" is empty')
    if found and not pages:
        raise InvalidAnswer("an answer that was found must cite at least one page")
    if not found and pages:
        raise InvalidAnswer('"pages" must be empty when "found" is false')
    return Answer(text.strip(), tuple(dict.fromkeys(pages)), found)


def cited_page_ids(answer: Answer, page_ids: list[str]) -> list[str]:
    """Map an answer's page labels back to page ids.

    Args:
        answer (Answer): Checked answer.
        page_ids (list[str]): Page ids in prompt order.

    Returns:
        list[str]: Cited page ids, in citation order.
    """
    return [page_ids[label - 1] for label in answer.pages]
