"""Answer grading by an LLM judge: prompt, reply checks with retries, and judge loading.

The judge compares the model's answer with the reference answers (not with the pages) and returns a short explanation
and one of three labels. Loading needs `torch`, `transformers` and `bitsandbytes`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from doc_qa.answer.output import InvalidAnswer, extract_json
from doc_qa.answer.prompt import retry_messages

LABELS = ("correct", "partially correct", "incorrect")

JUDGE_PROMPT = """You grade an answer to a question about a document by comparing it with reference answers written by \
people who read the document. Each reference answer was written from one page; together they give the core \
information the answer needs. The references may be in another language than the question or the answer: compare \
the meaning, not the wording.

Labels:
- Correct: the answer contains the core information of the references and nothing that contradicts it. Extra \
details are fine.
- Partially correct: the answer contains part of the core information, or the core information together with a \
claim that contradicts the references.
- Incorrect: the answer misses or contradicts the core information, or says the information is not available.

Question: {question}

Reference answers:
{references}

Answer to grade:
{answer}

Reply with only this JSON object:
{{"explanation": "<one or two short sentences>", "judgment": "Correct" or "Partially correct" or "Incorrect"}}"""


@dataclass(frozen=True)
class Verdict:
    """A judge's grade. `label` is None when no attempt gave a valid reply."""

    label: str | None
    explanation: str
    attempts: int
    errors: tuple[str, ...]


def reference_block(references: list[str], max_chars: int) -> str:
    """List reference answers in order until a length budget, noting how many were left out.

    Args:
        references (list[str]): Reference answers.
        max_chars (int): Characters of references kept; the first one is always kept whole.

    Returns:
        str: Bulleted references.
    """
    kept, used = [], 0
    for reference in references:
        if kept and used + len(reference) > max_chars:
            break
        kept.append(f"- {reference}")
        used += len(reference)
    left_out = len(references) - len(kept)
    return "\n".join(kept) + (f"\n({left_out} more reference answers not shown)" if left_out else "")


def judge_messages(question: str, references: list[str], answer: str, max_reference_chars: int) -> list[dict[str, Any]]:
    """Build the judge's chat: one user turn, since some chat templates have no system role.

    Args:
        question (str): Question.
        references (list[str]): Reference answers.
        answer (str): Answer to grade.
        max_reference_chars (int): Characters of references kept, so long reference lists don't exhaust memory.

    Returns:
        list[dict[str, Any]]: Chat messages.
    """
    text = JUDGE_PROMPT.format(
        question=question, references=reference_block(references, max_reference_chars), answer=answer
    )
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def parse_verdict(reply: str) -> tuple[str, str]:
    """Read the label and explanation from a judge reply.

    Args:
        reply (str): Raw judge output.

    Returns:
        tuple[str, str]: Label (one of LABELS) and explanation.
    """
    try:
        data = json.loads(extract_json(reply))
    except json.JSONDecodeError as error:
        raise InvalidAnswer(f"the JSON is malformed ({error.msg})") from None
    if not isinstance(data, dict) or not isinstance(data.get("judgment"), str):
        raise InvalidAnswer('"judgment" is missing')
    label = " ".join(data["judgment"].replace("_", " ").lower().split())
    if label not in LABELS:
        raise InvalidAnswer('"judgment" must be "Correct", "Partially correct" or "Incorrect"')
    return label, str(data.get("explanation", "")).strip()


def judge_with_retries(
    generate: Callable[[list[dict[str, Any]]], str], messages: list[dict[str, Any]], retries: int
) -> Verdict:
    """Ask the judge; while the reply is invalid, show it the reply and the reason, and ask again.

    Args:
        generate (Callable[[list[dict[str, Any]]], str]): Judge call, chat messages to raw reply (greedy).
        messages (list[dict[str, Any]]): Chat from `judge_messages`.
        retries (int): Extra attempts after the first.

    Returns:
        Verdict: The first valid grade, with the number of attempts and why earlier replies were rejected.
    """
    errors: list[str] = []
    for attempt in range(1, retries + 2):
        reply = generate(messages)
        try:
            label, explanation = parse_verdict(reply)
            return Verdict(label, explanation, attempt, tuple(errors))
        except InvalidAnswer as error:
            errors.append(str(error))
            messages = retry_messages(messages, reply, str(error))
    return Verdict(None, "", retries + 1, tuple(errors))


@dataclass(frozen=True)
class JudgeReply:
    """One judge reply and what it cost."""

    text: str
    prompt_tokens: int
    seconds: float
    finite: bool  # False if any next-token scores were NaN or +inf (fp16 overflow)


def with_system_prompt(messages: list[dict[str, Any]], system_prompt: str | None) -> list[dict[str, Any]]:
    """Put a fixed system turn first, so a chat template doesn't insert its own (some add today's date).

    Args:
        messages (list[dict[str, Any]]): Chat messages.
        system_prompt (str | None): System text, or None to leave the chat as it is.

    Returns:
        list[dict[str, Any]]: Chat messages.
    """
    if not system_prompt:
        return messages
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, *messages]


def load_judge(spec: dict[str, Any], max_new_tokens: int) -> Callable[[list[dict[str, Any]]], JudgeReply]:
    """Load a pre-quantized 4-bit judge over all GPUs, computing in fp16, and return a greedy chat function.

    Args:
        spec (dict[str, Any]): Judge entry of configs/judges.yaml (model, revision, max_memory, template_kwargs,
            system_prompt).
        max_new_tokens (int): Reply length limit.

    Returns:
        Callable[[list[dict[str, Any]]], JudgeReply]: Chat messages to reply.
    """
    import bitsandbytes as bnb
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM as AutoModel
    except ImportError:  # older transformers
        from transformers import AutoModelForImageTextToText as AutoModel

    processor = AutoProcessor.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModel.from_pretrained(
        spec["model"], revision=spec["revision"], dtype=torch.float16, device_map="auto",
        max_memory=spec.get("max_memory"),
    ).eval()
    for module in model.modules():  # the checkpoints ask for bf16 computation, which the T4 lacks
        if isinstance(module, bnb.nn.Linear4bit):
            module.compute_dtype = torch.float16
    template_kwargs = spec.get("template_kwargs", {})

    def encode(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return processor.apply_chat_template(
            with_system_prompt(messages, spec.get("system_prompt")), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **template_kwargs,
        )

    def chat(messages: list[dict[str, Any]]) -> JudgeReply:
        inputs = {k: v.to(model.device) for k, v in encode(messages).items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                return_dict_in_generate=True,
            )
        seconds = time.perf_counter() - start
        finite = all(not (torch.isnan(s).any() or torch.isposinf(s).any()) for s in output.scores)
        text = processor.decode(output.sequences[0, prompt_tokens:], skip_special_tokens=True)
        return JudgeReply(text, prompt_tokens, round(seconds, 2), bool(finite))

    return chat
