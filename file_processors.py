"""File extraction utilities for Golden Litigator OS."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple


def process_txt(path: str) -> str:
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return p.read_text(encoding="latin-1", errors="ignore")
        except Exception as exc:  # pragma: no cover - log and return empty
            logging.error("TXT read failed: %s // %s", p, exc)
            return ""


def process_docx(path: str) -> str:
    import docx

    try:
        doc = docx.Document(path)
        return "\n".join(par.text for par in doc.paragraphs)
    except Exception as exc:  # pragma: no cover - log and return empty
        logging.error("DOCX read failed: %s // %s", path, exc)
        return ""


def process_pdf(path: str) -> str:
    import fitz  # type: ignore[import-not-found]

    try:
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text() or "")
        return "\n".join(text)
    except Exception as exc:  # pragma: no cover - log and return empty
        logging.error("PDF read failed: %s // %s", path, exc)
        return ""


def process_image(path: str, tesseract_cmd: str = "") -> str:
    import pytesseract  # type: ignore[import-untyped]
    from PIL import Image, ImageOps

    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        img = Image.open(path)
        img = ImageOps.grayscale(img)  # type: ignore[assignment]
        return pytesseract.image_to_string(img) or ""
    except Exception as exc:  # pragma: no cover
        logging.error("IMG OCR failed: %s // %s", path, exc)
        return ""


def process_audio(
    path: str, backend: str = "faster-whisper", model: str = "medium"
) -> str:
    try:
        if backend == "faster-whisper":
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            model_obj = WhisperModel(model, compute_type="int8_float16")
            segments, _ = model_obj.transcribe(path, vad_filter=True)
            return " ".join(seg.text for seg in segments if getattr(seg, "text", None))
        import whisper  # type: ignore[import-not-found]

        model_obj = whisper.load_model(model)
        res = model_obj.transcribe(path)
        return str(res.get("text", ""))
    except Exception as exc:  # pragma: no cover
        logging.error("Audio transcription failed: %s // %s", path, exc)
        return ""


def process_video(
    path: str, backend: str = "faster-whisper", model: str = "medium"
) -> str:
    try:
        with TemporaryDirectory() as td:
            audio_path = str(Path(td) / f"{Path(path).stem}.wav")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                audio_path,
            ]
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            if Path(audio_path).exists():
                return process_audio(audio_path, backend=backend, model=model)
            return ""
    except Exception as exc:  # pragma: no cover
        logging.error("Video extraction failed: %s // %s", path, exc)
        return ""


def dispatch_get_text(
    path: str,
    *,
    tesseract_cmd: str = "",
    whisper_backend: str = "faster-whisper",
    whisper_model: str = "medium",
) -> Tuple[str, str]:
    """Return extracted text and source type for ``path``."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".txt", ".json", ".csv", ".md"}:
        return process_txt(path), "txt"
    if ext in {".docx"}:
        return process_docx(path), "docx"
    if ext in {".pdf"}:
        return process_pdf(path), "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return process_image(path, tesseract_cmd=tesseract_cmd), "img"
    if ext in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}:
        return (
            process_audio(path, backend=whisper_backend, model=whisper_model),
            "audio",
        )
    if ext in {".mp4", ".mkv", ".mov", ".m4v", ".wmv", ".avi"}:
        return (
            process_video(path, backend=whisper_backend, model=whisper_model),
            "video",
        )
    return "", "unknown"
