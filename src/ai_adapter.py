"""Simple AI adapter that prefers local open-source models when `AI_BACKEND=local`.

This is an optional shim: it will use `transformers` if available; otherwise
it provides a safe fallback that raises a clear error. It does not change
existing code automatically — import and use `ai_adapter.generate()` where
you previously called remote APIs to opt into local models.
"""

from __future__ import annotations

import os

AI_BACKEND = os.getenv("AI_BACKEND", "local")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gpt2")

try:
    if AI_BACKEND == "local":
        from transformers import pipeline

        _gen = pipeline("text-generation", model=LOCAL_MODEL)
    else:
        _gen = None
except Exception:
    _gen = None


def supported_backends() -> list[str]:
    return ["local"]


def generate(prompt: str, max_tokens: int = 128) -> str:
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
        out = _gen(prompt, max_new_tokens=max_tokens, do_sample=True)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", "")
        return str(out)

    raise RuntimeError(f"Unsupported AI_BACKEND: {AI_BACKEND}. Supported: {supported_backends()}")
