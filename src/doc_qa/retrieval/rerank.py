"""Text reranking with a Qwen3 reranker: the model answers "yes" or "no" to whether a page fits a question.

Loading and scoring require `torch` and `transformers` (not base dependencies).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from doc_qa.core.parallel import batched

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_reranker(name: str, revision: str, dtype: str, device: str, attn_implementation: str) -> tuple[Any, Any]:
    """Load the reranker and its tokenizer (left padding, so every prompt ends at the last position).

    Args:
        name (str): Hugging Face repo id.
        revision (str): Commit hash.
        dtype (str): Torch dtype name, e.g. "float16".
        device (str): Torch device, or "auto" to spread the model over all GPUs.
        attn_implementation (str): Attention backend, e.g. "sdpa".

    Returns:
        tuple[Any, Any]: Model, tokenizer.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
        **({"device_map": "auto"} if device == "auto" else {}),
    )
    return (model if device == "auto" else model.to(device)).eval(), tokenizer


def pair_ids(tokenizer: Any, instruction: str, query: str, document: str, max_tokens: int) -> list[int]:
    """Build the token ids of one (question, page) prompt, cutting the end of the page if it is too long.

    Args:
        tokenizer (Any): Reranker tokenizer.
        instruction (str): Task description, in English.
        query (str): Question.
        document (str): Page text.
        max_tokens (int): Prompt limit.

    Returns:
        list[int]: Token ids.
    """
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body = tokenizer.encode(
        f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}", add_special_tokens=False
    )
    return prefix + body[: max_tokens - len(prefix) - len(suffix)] + suffix


def score_pairs(model: Any, tokenizer: Any, prompts: list[list[int]], batch_size: int) -> np.ndarray:
    """Score prompts by how much the model prefers "yes" over "no". Batches group prompts of similar length.

    Args:
        model (Any): Reranker.
        tokenizer (Any): Its tokenizer.
        prompts (list[list[int]]): Token ids from `pair_ids`.
        batch_size (int): Prompts per forward pass.

    Returns:
        np.ndarray: logit("yes") - logit("no") per prompt, in input order; higher means more relevant.
    """
    import torch

    yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
    device = model.get_input_embeddings().weight.device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    scores = np.empty(len(prompts), dtype=np.float32)
    for batch in batched(order, batch_size):
        padded = tokenizer.pad({"input_ids": [prompts[i] for i in batch]}, padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(
                input_ids=padded["input_ids"].to(device),
                attention_mask=padded["attention_mask"].to(device),
                logits_to_keep=1,  # only the next-token logits are needed
            ).logits[:, -1, :]
        scores[batch] = (logits[:, yes] - logits[:, no]).float().cpu().numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite reranker scores (fp16 overflow?)")
    return scores
