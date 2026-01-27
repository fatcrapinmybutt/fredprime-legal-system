"""Simple AI adapter that prefers local open-source models when `AI_BACKEND=local`.

This is an optional shim: it will use `transformers` if available; otherwise
it provides a safe fallback that raises a clear error. It does not change
existing code automatically — import and use `ai_adapter.generate()` where
you previously called remote APIs to opt into local models.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Mapping, Sequence

AI_BACKEND = os.getenv("AI_BACKEND", "local")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gpt2")

if AI_BACKEND == "local" and importlib.util.find_spec("transformers"):
    from transformers import pipeline

    _gen = pipeline("text-generation", model=LOCAL_MODEL)
else:
    _gen = None


def supported_backends() -> list[str]:
    return ["local"]


Message = Mapping[str, str]


def _format_role_content(message: Message) -> str:
    return f"{message.get('role', 'user')}: {message.get('content', '')}".rstrip()


def _render_messages(messages: Sequence[Message], gen: Any) -> str:
    tokenizer = getattr(gen, "tokenizer", None)
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(_format_role_content(message) for message in messages)


def _apply_stop_sequences(text: str, stop: Sequence[str] | None) -> str:
    if not stop:
        return text
    earliest = None
    for marker in stop:
        if not marker:
            continue
        index = text.find(marker)
        if index != -1 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text
    return text[:earliest]


def _extract_text(output: Any) -> str:
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            generated = first.get("generated_text", "")
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict) and "content" in last:
                    return str(last["content"])
                return str(last)
            if isinstance(generated, str):
                return generated
        return str(first)
    return str(output)


def _generation_kwargs(
    *,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    do_sample: bool | None,
    return_full_text: bool,
) -> dict[str, Any]:
    if do_sample is None:
        do_sample = temperature is not None or top_p is not None
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "return_full_text": return_full_text,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    return kwargs


def generate(
    prompt: str | Sequence[Message],
    max_tokens: int = 128,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    stop: Sequence[str] | None = None,
    return_full_text: bool = False,
) -> str:
    """Generate a short text continuation from the chosen backend.

    - If `AI_BACKEND=local` and `transformers` is available, use it.
    - Otherwise raise RuntimeError explaining how to enable local models.
    """
    if AI_BACKEND == "local":
        if _gen is None:
            raise RuntimeError(
                "Local backend requested but `transformers` pipeline not available.\n"
                "Install `transformers` and a model (or set AI_BACKEND to another backend)."
            )
        kwargs = _generation_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_full_text=return_full_text,
        )
        if isinstance(prompt, (list, tuple)):
            rendered = _render_messages(prompt, _gen)
            out = _gen(rendered, **kwargs)
        else:
            out = _gen(prompt, **kwargs)
        return _apply_stop_sequences(_extract_text(out), stop)

    raise RuntimeError(f"Unsupported AI_BACKEND: {AI_BACKEND}. Supported: {supported_backends()}")
