# Environment & Secrets Audit (Repo‑Visible References)

## Scope & Limits
- This audit **only** covers variables referenced in the repository files. It **cannot** read GitHub Secrets or external environment configurations.  
- Items not referenced in repo code/docs are marked **PINPOINT_MISSING** for verification in your CI/Codex/GitHub settings.  

---

## Observed References in Repo

### Core LLM Access
- **OPENAI_API_KEY** is referenced for runtime configuration.  
  - `golden_litigator_os.py` uses `OPENAI_API_KEY` when selecting an OpenAI provider.  
  - `litigation_core_engine_v_9999_full.py` enforces that `OPENAI_API_KEY` is set at startup.  

### Hugging Face (Conditional)
- **HF_TOKEN** is referenced for Hugging Face inference client access.  

### Logging / Audit
- **LOG_LEVEL** is set in CLI usage for logging verbosity.  

### CI / Pipeline Secrets
- **GITHUB_TOKEN** is referenced in CI documentation as a secret to store.  

---

## PINPOINT_MISSING (Not Found in Repo References)
These names are **not** referenced in repo files; confirm in CI/Codex configuration if required.

- **GH_PAT** (if you require PAT beyond GITHUB_TOKEN).  
- **NEO4J_*** (e.g., `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`).  
- **RCLONE_*** / **GDRIVE_*** (Drive sync).  
- **HF_TOKEN** scoping requirements (referenced, but no env‑scoped guidance).  
- **Build/signing tokens** (registry, package publish, signing).  
- **Audit/telemetry flags** (e.g., `AUDIT_LOG_ENABLED`, OTEL variables).  

---

## Required Verification (Dev/Stage/Prod)
Treat all of the following as **missing until proven present and environment‑scoped**:
- **OPENAI_API_KEY** (distinct per environment).  
- **GITHUB_TOKEN** or **GH_PAT** (only if workflows require write access beyond GITHUB_TOKEN).  
- **HF_TOKEN** (if HF inference is used).  
- **NEO4J_*** and **RCLONE_***/**GDRIVE_*** (only if those services are enabled).  
- **Build/signing secrets** (only if publish/sign workflows are active).  

---

## Evidence (Repo‑Visible Lines)
- `OPENAI_API_KEY` usage in OpenAI provider selection.  
- `OPENAI_API_KEY` required at runtime.  
- `HF_TOKEN` usage for Hugging Face.  
- `LOG_LEVEL` set for CLI verbosity.  
- `GITHUB_TOKEN` referenced in CI secret configuration.  
