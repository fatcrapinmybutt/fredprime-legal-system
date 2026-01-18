"""
Comprehensive Testing Framework with Fixtures, Parameterization & Benchmarks

Provides:
- Shared pytest fixtures for common testing needs
- Database fixtures with automatic rollback
- Mock/stub factories
- Parameterized test examples
- Performance benchmarking utilities
- Integration test helpers
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import from config
from src.config import AppSettings, get_settings


@pytest.fixture(scope="session")
def settings() -> AppSettings:
    """Session-scoped settings fixture."""
    return get_settings()


@pytest.fixture(scope="function")
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture(scope="function")
def test_settings(tmp_path: Path) -> AppSettings:
    """Settings fixture with temporary directories."""
    return AppSettings(
        environment="testing",
        debug=True,
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        documents_dir=tmp_path / "documents",
        output_dir=tmp_path / "output",
        logs_dir=tmp_path / "logs",
    )


@pytest.fixture(scope="function")
def sample_form_data() -> Dict[str, Any]:
    """Sample form data for testing."""
    return {
        "id": "MC-12",
        "title": "Motion to Adjourn",
        "category": "motions",
        "fields": [
            {"name": "case_number", "type": "text", "required": True},
            {"name": "court", "type": "select", "required": True},
            {"name": "reason", "type": "textarea", "required": True},
        ],
    }


@pytest.fixture(scope="function")
def sample_document_data() -> Dict[str, Any]:
    """Sample document data for testing."""
    return {
        "filename": "motion_adjourn.docx",
        "case_id": "2025-001234-CZ",
        "type": "motion",
        "created_at": "2025-01-14T10:30:00",
        "author": "test@example.com",
        "content_hash": "abc123def456",
    }


@pytest.fixture(scope="function")
def mock_file_system(tmp_path: Path) -> Dict[str, Path]:
    """Mock file system structure for tests."""
    structure = {
        "forms": tmp_path / "forms",
        "documents": tmp_path / "documents",
        "output": tmp_path / "output",
        "evidence": tmp_path / "evidence",
    }

    for directory in structure.values():
        directory.mkdir(parents=True, exist_ok=True)

    return structure


@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer fixture."""

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self) -> float:
            if self.start_time is None:
                raise RuntimeError("Timer not started")
            self.elapsed = time.perf_counter() - self.start_time
            return self.elapsed

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()

    return Timer()


class MockDatabaseSession:
    """Mock database session for testing."""

    def __init__(self):
        self.data = {}
        self.committed = False
        self.rolled_back = False

    def add(self, obj):
        self.data[obj.id] = obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True
        self.data.clear()

    def query(self, model_class):
        return MockQuery(model_class, self.data)

    def close(self):
        pass


class MockQuery:
    """Mock query builder for testing."""

    def __init__(self, model_class, data):
        self.model_class = model_class
        self.data = data
        self.filters = {}

    def filter(self, condition):
        # Simple mock filter
        self.filters.update(condition)
        return self

    def all(self):
        return list(self.data.values())

    def first(self):
        results = self.all()
        return results[0] if results else None


@pytest.fixture
def mock_db_session():
    """Mock database session fixture."""
    return MockDatabaseSession()


# Parametrized test example
@pytest.mark.parametrize(
    "input_value,expected_output",
    [
        ("MC-12", "Motion to Adjourn"),
        ("FOC-87", "Motion Regarding Parenting Time"),
        ("MC-97", "Complaint for Injunctive Relief"),
    ],
)
def test_form_lookup(input_value, expected_output):
    """Example parametrized test."""
    # This demonstrates the pattern - actual implementation would use real form database
    pass


# Performance test example
def test_form_loading_performance(benchmark_timer, sample_form_data):
    """Example performance test."""

    def load_forms():
        return json.dumps(sample_form_data)

    with benchmark_timer:
        result = load_forms()

    assert result is not None
    # Could assert performance threshold: assert benchmark_timer.elapsed < 0.1


# Integration test helper class
class IntegrationTestHelper:
    """Helper for integration tests."""

    @staticmethod
    def create_test_case(case_id: str, case_type: str) -> Dict[str, Any]:
        """Create test case data."""
        return {
            "case_id": case_id,
            "type": case_type,
            "created_at": "2025-01-14T10:30:00",
            "status": "open",
            "documents": [],
            "timeline": [],
        }

    @staticmethod
    def create_test_document(filename: str, content: str = "") -> Dict[str, Any]:
        """Create test document."""
        return {
            "filename": filename,
            "content": content,
            "size": len(content),
            "created_at": "2025-01-14T10:30:00",
        }

    @staticmethod
    def create_test_evidence() -> List[Dict[str, Any]]:
        """Create test evidence items."""
        return [
            {"id": "EXH-A", "description": "Document A", "type": "document"},
            {"id": "EXH-B", "description": "Photo B", "type": "photo"},
            {"id": "EXH-C", "description": "Email C", "type": "email"},
        ]


@pytest.fixture
def integration_helper() -> IntegrationTestHelper:
    """Integration test helper fixture."""
    return IntegrationTestHelper()
