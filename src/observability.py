"""
Observability utilities: structured logging, Prometheus metrics, and OpenTelemetry starter.

This module provides safe fallbacks when optional packages are missing and exposes
helpers for setting up logging, Prometheus metrics, and tracing.
"""

import logging
import os
from typing import Any, Callable, Optional

jsonlogger: Optional[Any] = None
try:
    from pythonjsonlogger import jsonlogger as _jsonlogger

    jsonlogger = _jsonlogger
except Exception:
    pass

# Prometheus (optional)
start_http_server: Optional[Callable[..., None]] = None
Counter: Any = None
Histogram: Any = None
Summary: Any = None
make_asgi_app: Optional[Callable[..., Any]] = None
try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Histogram as _Histogram
    from prometheus_client import Summary as _Summary
    from prometheus_client import make_asgi_app as _make_asgi_app
    from prometheus_client import start_http_server as _start_http_server

    start_http_server = _start_http_server
    Counter = _Counter
    Histogram = _Histogram
    Summary = _Summary
    make_asgi_app = _make_asgi_app
except Exception:
    pass


class _DummyMetric:
    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def inc(self, *a: Any, **k: Any) -> None:
        pass

    def observe(self, *a: Any, **k: Any) -> None:
        pass


if Counter is None:
    Counter = _DummyMetric

if Histogram is None:
    Histogram = _DummyMetric

if Summary is None:
    Summary = _DummyMetric


# OpenTelemetry (optional)
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:
    trace = None
    TracerProvider = None
    OTLPSpanExporter = None
    BatchSpanProcessor = None
    FastAPIInstrumentor = None
    RequestsInstrumentor = None


DEFAULT_METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))

# Default application-wide metrics (if prometheus_client available)
request_counter: Optional[Any] = None
request_latency: Optional[Any] = None


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
