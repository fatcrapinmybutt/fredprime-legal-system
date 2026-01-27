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


def _render_messages(messages: Sequence[Message]) -> str:
    return "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}".rstrip()
        for message in messages
    )


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


def generate(prompt: str | Sequence[Message], max_tokens: int = 128) -> str:
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
        if isinstance(prompt, (list, tuple)):
            try:
                out = _gen(prompt, max_new_tokens=max_tokens, do_sample=True)
            except Exception:
                out = _gen(_render_messages(prompt), max_new_tokens=max_tokens, do_sample=True)
        else:
            out = _gen(prompt, max_new_tokens=max_tokens, do_sample=True)
        return _extract_text(out)

    raise RuntimeError(f"Unsupported AI_BACKEND: {AI_BACKEND}. Supported: {supported_backends()}")
