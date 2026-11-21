"""Answer model (Qwen3-VL): loading, page image sizing and greedy generation.

Loading and generation require `torch` and `transformers` (not base dependencies).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

PIXELS_PER_TOKEN = 32 * 32  # Qwen3-VL: 16-pixel patches, merged 2 x 2 into one token


@dataclass(frozen=True)
class Generation:
    """One model reply and what it cost."""

    reply: str
    prompt_tokens: int
    new_tokens: int
    seconds: float
    finite: bool  # False if any next-token scores were NaN or +inf (fp16 overflow)


def fit_pixels(image: Image.Image, max_tokens: int) -> Image.Image:
    """Shrink a page image so it costs at most about `max_tokens` image tokens; smaller images are kept.

    Args:
        image (Image.Image): Page image.
        max_tokens (int): Image token budget per page.

    Returns:
        Image.Image: The image, resized with its aspect ratio kept.
    """
    max_pixels = max_tokens * PIXELS_PER_TOKEN
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)


def load_answer_model(
    name: str, revision: str, dtype: str, attn_implementation: str, max_memory: dict[int, str] | None = None
) -> tuple[Any, Any]:
    """Load the answer model spread over all GPUs, and its processor.

    Args:
        name (str): Hugging Face repo id.
        revision (str): Commit hash.
        dtype (str): Torch dtype name, e.g. "float16".
        attn_implementation (str): Attention backend, e.g. "sdpa".
        max_memory (dict[int, str] | None): Weight budget per GPU (e.g. {0: "11GiB"}), leaving room for activations.

    Returns:
        tuple[Any, Any]: Model, processor.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(name, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        name,
        revision=revision,
        dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
        device_map="auto",
        max_memory=max_memory,
    )
    return model.eval(), processor


def chat_inputs(processor: Any, messages: list[dict[str, Any]]) -> Any:
    """Turn chat messages into model inputs; images are taken from the messages in order.

    Args:
        processor (Any): Model processor.
        messages (list[dict[str, Any]]): Chat from `build_messages` (and `retry_messages`).

    Returns:
        Any: Processor output (input ids, attention mask, pixel values, image grid sizes).
    """
    images = [item["image"] for message in messages for item in message["content"] if item["type"] == "image"]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=images or None, return_tensors="pt")


def generate_reply(model: Any, processor: Any, messages: list[dict[str, Any]], max_new_tokens: int) -> Generation:
    """Generate one reply greedily.

    Args:
        model (Any): Answer model.
        processor (Any): Its processor.
        messages (list[dict[str, Any]]): Chat messages.
        max_new_tokens (int): Reply length limit.

    Returns:
        Generation: Reply text, token counts, seconds and whether the scores stayed finite.
    """
    import torch

    inputs = chat_inputs(processor, messages).to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    seconds = time.perf_counter() - start
    new_ids = output.sequences[0, prompt_tokens:]
    finite = all(not (torch.isnan(s).any() or torch.isposinf(s).any()) for s in output.scores)
    reply = processor.decode(new_ids, skip_special_tokens=True)
    return Generation(reply, prompt_tokens, int(new_ids.shape[0]), seconds, bool(finite))
