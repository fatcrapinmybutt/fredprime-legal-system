"""
Observability utilities: structured logging, Prometheus metrics, and OpenTelemetry starter.

Usage:
    from src.observability import setup_logging, instrument_fastapi, start_metrics_server

    settings = get_settings()
    setup_logging(settings)
    app = FastAPI()
    instrument_fastapi(app)

    # Optional: start external Prometheus metrics server
    start_metrics_server(port=8000)

This module avoids hard failures if optional dependencies are missing.
"""

import logging
import os
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None

# Prometheus
try:
    from prometheus_client import start_http_server, Counter, Histogram, Summary
    from prometheus_client import make_asgi_app
except Exception:
    start_http_server = None
    Counter = None
    Histogram = None
    Summary = None
    make_asgi_app = None


# Provide lightweight fallbacks to avoid import-time failures
class _DummyMetric:
    def __init__(self, *a, **k):
        pass

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass


if Counter is None:
    Counter = _DummyMetric


if Histogram is None:
    Histogram = _DummyMetric


if Summary is None:
    Summary = _DummyMetric

# OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
except Exception:
    trace = None
    TracerProvider = None
    OTLPSpanExporter = None
    BatchSpanProcessor = None
    FastAPIInstrumentor = None
    RequestsInstrumentor = None


DEFAULT_METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))

# Default application-wide metrics (if prometheus_client available)
request_counter = None
request_latency = None


def setup_logging(settings: Optional[object] = None) -> None:
    """Configure structured logging.

    Accepts either the AppSettings instance or a simple object with `.logging` attr.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = None
    if settings is not None:
        try:
            fmt = settings.logging.format
            level = settings.logging.level
            log_file = settings.logging.file
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        except Exception:
            pass

    # Console handler
    ch = logging.StreamHandler()
    if jsonlogger:
        formatter = jsonlogger.JsonFormatter(fmt)
    else:
        formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def init_metrics(app=None) -> None:
    """Initialize Prometheus metrics objects and optionally attach to FastAPI app."""
    global request_counter, request_latency
    try:
        if Counter is None:
            return
        request_counter = Counter("fred_requests_total", "Total HTTP requests")
        request_latency = Histogram("fred_request_latency_seconds", "Request latency (s)")

        if app is not None and make_asgi_app is not None:
            # mount /metrics
            metrics_app = make_asgi_app()
            app.mount("/metrics", metrics_app)
    except Exception:
        logging.getLogger(__name__).exception("Failed to init metrics")


def start_metrics_server(port: int = DEFAULT_METRICS_PORT) -> None:
    """Start the Prometheus metrics HTTP server (background)."""
    if start_http_server is None:
        logging.getLogger(__name__).warning("prometheus_client not installed; metrics disabled")
        return
    try:
        start_http_server(port)
        logging.getLogger(__name__).info("Prometheus metrics server started on port %s", port)
    except Exception:
        logging.getLogger(__name__).exception("Failed to start Prometheus metrics server")


def record_request(duration_seconds: float = 0.0) -> None:
    """Record a single request (helper)."""
    if request_counter is not None:
        try:
            request_counter.inc()
            if request_latency is not None:
                request_latency.observe(duration_seconds)
        except Exception:
            pass


def init_tracing(service_name: str = "fredprime") -> None:
    """Initialize OpenTelemetry tracing with OTLP exporter if available.

    No-op if opentelemetry packages are missing or env not configured.
    """
    if TracerProvider is None or OTLPSpanExporter is None:
        logging.getLogger(__name__).warning("OpenTelemetry not available; tracing disabled")
        return

    try:
        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)
        logging.getLogger(__name__).info("OpenTelemetry tracing initialized for %s", service_name)
    except Exception:
        logging.getLogger(__name__).exception("Failed to initialize OpenTelemetry")


def instrument_fastapi(app) -> None:
    """Instrument a FastAPI app with metrics and optional tracing.

    - mounts /metrics (prometheus_client) if available
    - instruments FastAPI for tracing if opentelemetry available
    """
    init_metrics(app)

    if FastAPIInstrumentor is not None:
        try:
            FastAPIInstrumentor().instrument_app(app)
            logging.getLogger(__name__).info("FastAPI instrumented for OpenTelemetry")
        except Exception:
            logging.getLogger(__name__).exception("Failed to instrument FastAPI")

    if RequestsInstrumentor is not None:
        try:
            RequestsInstrumentor().instrument()
        except Exception:
            logging.getLogger(__name__).exception("Failed to instrument requests")
