# LITIGATION_CORE_ENGINE_v9999.py (FULL SYSTEM – LEVEL 9999, *GUI-WRAPPED EXECUTABLE + CONFIG-DRIVEN THREAD POOL + PREFECT + GPT + RETRIES + LOGGING + DAG + ALERTS + AUDIT + METRICS + SMS/EMAIL + SHA256 + GUI.EXE*)

import os
import sys
import json
import argparse
import logging
import hashlib
import smtplib
import threading
from email.message import EmailMessage
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from pythonjsonlogger import jsonlogger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from gui_wrapper import launch_gui
from config_manager import ConfigManager, ConfigSchema
from ocr_engine import OCRFallback
from memory_crawler import MemoryCrawler
from gpt_tools import extract_text, classify_step, determine_filings
from db_repo import PostgresDBRepo, DBRepository
from metrics import doc_build_counter
from behavior_manager import BehaviorManager
from alerts import send_sms, send_email

# === Logging Configuration ===
logger = logging.getLogger("LitigationEngine")
logger.setLevel(logging.INFO)
json_handler = logging.FileHandler("litigation_engine.log")
json_handler.setFormatter(jsonlogger.JsonFormatter())
console_handler = logging.StreamHandler()
console_handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(json_handler)
logger.addHandler(console_handler)

# === Structured GPT Logging ===
gpt_log_handler = logging.FileHandler("gpt_traces.jsonl")
gpt_log_handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(gpt_log_handler)

# === FastAPI Prefect Integration ===
app = FastAPI()
FastAPIInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()


# === Prefect Flow ===
@task(name="Run Litigation Pipeline", retries=3, retry_delay_seconds=10)
def litigation_task(case_num: str, cfg_dict: dict):
    cfg = ConfigSchema(**cfg_dict)
    db = PostgresDBRepo(cfg.db_config)
    run_evidence_pipeline(case_num, cfg, db)


@flow(name="Litigation Engine Flow", task_runner=ConcurrentTaskRunner())
def litigation_flow(case_num: str, cfg_dict: dict):
    litigation_task.submit(case_num, cfg_dict)


# === GPT Draft Wrapper with Retry + Logging ===
class GPTClient:
    @staticmethod
    @BehaviorManager.retry(times=4, backoff=3)
    def draft(cfg: ConfigSchema, content: str, case_num: str, file_path: str):
        response = openai.ChatCompletion.create(
            model=cfg.gpt_model,
            messages=[
                {"role": "system", "content": "Extract facts for litigation."},
                {"role": "user", "content": content},
            ],
            timeout=cfg.gpt_timeout,
        )
        logger.info(
            "GPT draft success",
            extra={"file": file_path, "case": case_num, "response": response},
        )
        return response


# === SHA256 Hasher ===
def compute_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# === Evidence Pipeline with GPTClient, Prefect, and Logging ===
def run_evidence_pipeline(case_num: str, cfg: ConfigSchema, db: DBRepository):
    items = MemoryCrawler().scan(cfg.source_paths, case_num)
    db.save_files(items)

    ocr_results = []
    with ThreadPoolExecutor(max_workers=cfg.ocr_threads) as executor:
        future_map = {
            executor.submit(OCRFallback().parse, itm["path"], case_num): itm
            for itm in items
        }
        for future in as_completed(future_map, timeout=cfg.ocr_timeout):
            itm = future_map[future]
            try:
                success = future.result(timeout=cfg.ocr_timeout)
                ocr_results.append((itm, success))
            except Exception as e:
                logger.error(
                    "OCR error for file",
                    extra={"file": itm["path"], "error": str(e), "case": case_num},
                )
                send_sms(f"OCR failed: {itm['path']} - {str(e)}")

    classification = classify_step(ocr_results, cfg, case_num)
    filings = determine_filings(classification)

    for doc, success in ocr_results:
        if not success:
            continue
        try:
            content = extract_text(doc["path"])[: cfg.snippet_chars]
            GPTClient.draft(cfg, content, case_num, doc["path"])
        except Exception as e:
            logger.error(
                "GPT error",
                extra={"file": doc["path"], "error": str(e), "case": case_num},
            )
            send_email(f"GPT Error on {doc['path']}", str(e))

    for filing in filings:
        filename = f"{case_num}_{filing.replace(' ', '_')}.docx"
        file_path = os.path.join(cfg.output_dir, filename)
        db.save_document(case_num, file_path)
        sha = compute_sha256(file_path)
        logger.info(
            "Document created",
            extra={"filename": filename, "sha256": sha, "case": case_num},
        )
        doc_build_counter.labels(case=case_num, document=filename).inc()


# === CLI / GUI Entrypoint ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Case number")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--gui", action="store_true", help="Launch GUI instead")
    args = parser.parse_args()

    if args.gui:
        launch_gui()
    elif args.case:
        cfg_obj = ConfigManager.load(args.config)
        litigation_flow(args.case, cfg_obj.dict())
    else:
        print("Either --case or --gui must be provided")
