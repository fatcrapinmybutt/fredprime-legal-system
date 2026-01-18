# Bootstrap Environment Requirements

This project requires the following components beyond the provided scripts. Install them before running `golden_god_mode_bootstrap.py`.

## 1. Runtime

- Windows 10 or 11 x64
- Python 3.10+ with `pip`
- Offline builds: supply a `.wheels/` wheelhouse matching `requirements.txt`

## 2. Models

- Local GGUF model at `models/llm/model.gguf` or enable Ollama in `config.yaml`
- SBERT embeddings at `models/emb/sentence-transformers/all-MiniLM-L6-v2`

## 3. External Binaries

- Tesseract OCR for PNG/JPG/TIFF processing
- Ghostscript and qpdf if PDF/A conversion or OCR pipelines are enabled

## 4. Service Wrapper

- NSSM or similar to run the backend as a Windows service instead of using `.bat` files

## 5. Optional Vector/ANN Extras

- `faiss-cpu` or `hnswlib` for accelerated search
- `sentence-transformers` for SBERT embeddings

## 6. Packaging & Ops

- `pyinstaller` and a `constraints.txt` file with hashed pins for reproducible one-file builds
- `minisign` or `cosign` to sign `evidence_manifest.json`

## 7. GPU Acceleration (Optional)

- Install the matching CUDA runtime and use a CUDA-built `llama.cpp` binary when GPU support is desired

---

## Single-Pass Setup Commands

Run these commands in PowerShell, adjusting the `F:\LAWFORGE_SUPREMACY` path if needed.

```powershell
# Verify Python
python --version

# Install external tools (via winget)
winget install -e --id TesseractOCR.Tesseract
winget install -e --id ArtifexSoftware.Ghostscript   # optional
winget install -e --id QPDF.QPDF                     # optional
winget install -e --id NSSM.NSSM                     # service wrapper
# Optional LLM provider
# winget install -e --id Ollama.Ollama

# Create workspace and bootstrap
python golden_god_mode_bootstrap.py --root "F:\LAWFORGE_SUPREMACY"  # add --offline if using .wheels

# Activate venv and install extras
F:\LAWFORGE_SUPREMACY\.venv\Scripts\activate
pip install faiss-cpu sentence-transformers hnswlib watchdog pyinstaller

# Place models
# - GGUF model at: F:\LAWFORGE_SUPREMACY\models\llm\model.gguf
# - SBERT model folder at: F:\LAWFORGE_SUPREMACY\models\emb\sentence-transformers\all-MiniLM-L6-v2

# Edit config.yaml with concrete paths and LLM settings

# Install API as Windows service
nssm install LAWFORGE_API "F:\LAWFORGE_SUPREMACY\.venv\Scripts\python.exe" "-m uvicorn backend.app:app --host 127.0.0.1 --port 8000"
nssm set LAWFORGE_API AppDirectory "F:\LAWFORGE_SUPREMACY"
nssm set LAWFORGE_API Start SERVICE_AUTO_START
nssm start LAWFORGE_API

# Optional: package to single EXE
pyinstaller -F -n LAWFORGE_API --add-data "backend;backend" --add-data "frontend;frontend" --add-data "config.yaml;." golden_god_mode_bootstrap.py
```
