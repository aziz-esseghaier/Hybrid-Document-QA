"""Ask the answer model, check its reply, and retry with the reason until the reply is usable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from doc_qa.answer.output import Answer, InvalidAnswer, parse_answer
from doc_qa.answer.prompt import retry_messages

Generate = Callable[[list[dict[str, Any]]], str]  # chat messages -> raw reply


@dataclass(frozen=True)
class AnswerResult:
    """Outcome for one question. `answer` is None when no attempt gave a usable reply."""

    answer: Answer | None
    attempts: int
    errors: tuple[str, ...]


def answer_with_retries(
    generate: Generate, messages: list[dict[str, Any]], page_count: int, retries: int
) -> AnswerResult:
    """Generate a reply; while it breaks the rules, show the model its reply and the reason, and ask again.

    Args:
        generate (Generate): Model call, chat messages to raw reply (greedy, so the same chat gives the same reply).
        messages (list[dict[str, Any]]): Chat from `build_messages`.
        page_count (int): Pages in the prompt.
        retries (int): Extra attempts after the first.

    Returns:
        AnswerResult: The first usable answer, the number of attempts, and why earlier replies were rejected.
    """
    errors: list[str] = []
    for attempt in range(1, retries + 2):
        reply = generate(messages)
        try:
            return AnswerResult(parse_answer(reply, page_count), attempt, tuple(errors))
        except InvalidAnswer as error:
            errors.append(str(error))
            messages = retry_messages(messages, reply, str(error))
    return AnswerResult(None, retries + 1, tuple(errors))
