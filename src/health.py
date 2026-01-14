"""
Lightweight health endpoint using FastAPI. Exposes /health and /ready and mounts metrics when available.

Usage:
    from fastapi import FastAPI
    from src.health import create_health_app

    app = create_health_app()

Run locally:
    uvicorn src.health:app --reload
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import logging
from typing import Optional

from .observability import instrument_fastapi, start_metrics_server


def _default_checks() -> dict:
    """Run quick local health checks. Non-blocking and inexpensive."""
    # Add more checks as needed (disk space, DB connection, caches, etc.)
    checks = {
        "disk_ok": True,
        "env": True,
    }
    try:
        # Example: warn if free space low
        statvfs = os.statvfs('.')
        free_bytes = statvfs.f_bavail * statvfs.f_frsize
        checks["free_bytes"] = free_bytes
        checks["disk_ok"] = free_bytes > 50 * 1024 * 1024  # 50MB
    except Exception:
        checks["disk_ok"] = False
    return checks


def create_health_app(metrics_port: Optional[int] = None) -> FastAPI:
    """Create and return a FastAPI app with health endpoints and metrics mounted if available."""
    app = FastAPI(title="FRED Health")

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    @app.get("/ready")
    def ready():
        checks = _default_checks()
        status = "ok" if checks.get("disk_ok") and checks.get("env") else "degraded"
        return JSONResponse({"status": status, "checks": checks})

    # Instrument app for metrics and tracing
    try:
        instrument_fastapi(app)
    except Exception:
        logging.getLogger(__name__).exception("Failed to instrument app with observability")

    # Optionally start a standalone metrics server for Prometheus scraping
    if metrics_port:
        try:
            start_metrics_server(metrics_port)
        except Exception:
            logging.getLogger(__name__).exception("Failed to start metrics server")

    return app


# Expose `app` for easy mounting / uvicorn runs
app = create_health_app(metrics_port=int(os.getenv("METRICS_PORT", "0")) or None)
