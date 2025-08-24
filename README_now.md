# Golden Litigator OS Quickstart

1. `python -m venv .venv && .\.venv\Scripts\activate`
2. `pip install --upgrade pip`
3. `pip install pymupdf python-docx pytesseract pillow opencv-python pydub faster-whisper openai anthropic chromadb sentence-transformers pypdf`
4. Set environment variables `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TESSERACT_OCR_PATH`
5. (Optional) `python case_meta_seed.py --state` or `--federal`
6. `python app_modular.py`
7. `python litigator_upgrade_suite.py`
8. (Optional) `python forms_overlay.py`
9. (Optional) `./install_task.ps1`
