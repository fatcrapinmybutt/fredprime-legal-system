# Run the modular Golden Litigator OS
$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install pymupdf python-docx pytesseract pillow opencv-python pydub faster-whisper openai anthropic pypdf
$env:TESSERACT_OCR_PATH = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python .\app_modular.py
