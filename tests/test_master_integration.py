"""
Master Workflow Integration Tests

Comprehensive test suite for:
- Workflow engine orchestration
- CLI interface
- State management
- Stage handlers
- End-to-end workflow execution
- Error handling and recovery
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Test fixtures and helpers
@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "evidence").mkdir()
        (workspace / "config").mkdir()
        (workspace / "output").mkdir()
        (workspace / "state").mkdir()
        yield workspace


@pytest.fixture
def sample_evidence_files(temp_workspace):
    """Create sample evidence files for testing."""
    evidence_dir = temp_workspace / "evidence"
    files = []

    # Create sample text files
    for i in range(5):
        file_path = evidence_dir / f"document_{i:03d}.txt"
        content = f"Sample evidence file {i}\nDate: {datetime.now()}\nContent: {i * 100} words"
        file_path.write_text(content)
        files.append(file_path)

    return files


@pytest.fixture
def case_context(temp_workspace, sample_evidence_files):
    """Create sample case context."""
    from src.master_integration_bridge import CaseContext

    return CaseContext(
        case_id="TEST2025001",
        case_type="custody",
        case_number="2025-001234-CZ",
        root_directories=[temp_workspace / "evidence"],
        parties={
            "plaintiff": "Test Plaintiff",
            "defendant": "Test Defendant",
        },
    )


# ============================================================================
# Stage Handler Tests
# ============================================================================

class TestStageHandlers:
    """Test individual stage handlers."""

    @pytest.mark.asyncio
    async def test_intake_handler_ingests_files(self, case_context, sample_evidence_files):
        """Test INTAKE stage ingests evidence files."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()
        result = await registry.handle_intake_stage(case_context, {})

        assert result['status'] == 'completed'
        assert result['files_ingested'] == len(sample_evidence_files)
        assert len(case_context.evidence_files) == len(sample_evidence_files)

    @pytest.mark.asyncio
    async def test_analysis_handler_scores_files(self, case_context, sample_evidence_files):
        """Test ANALYSIS stage scores files."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # First ingest
        await registry.handle_intake_stage(case_context, {})

        # Then analyze
        result = await registry.handle_analysis_stage(case_context, {})

        assert result['status'] == 'completed'
        assert 'avg_relevance_score' in result
        assert all('relevance_score' in f for f in case_context.evidence_files)

    @pytest.mark.asyncio
    async def test_organization_handler_labels_exhibits(self, case_context, sample_evidence_files, temp_workspace):
        """Test ORGANIZATION stage labels exhibits."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # Ingest and analyze
        await registry.handle_intake_stage(case_context, {})
        await registry.handle_analysis_stage(case_context, {})

        # Organize
        output_dir = temp_workspace / "exhibits"
        result = await registry.handle_organization_stage(case_context, {'output_dir': output_dir})

        assert result['status'] == 'completed'
        assert result['exhibits_created'] == len(sample_evidence_files)
        assert output_dir.exists()
        assert all('exhibit_label' in f for f in case_context.evidence_files)

    @pytest.mark.asyncio
    async def test_generation_handler_creates_documents(self, case_context, temp_workspace):
        """Test GENERATION stage creates documents."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # Setup
        await registry.handle_intake_stage(case_context, {})
        await registry.handle_analysis_stage(case_context, {})

        # Generate
        output_dir = temp_workspace / "documents"
        result = await registry.handle_generation_stage(case_context, {'output_dir': output_dir})

        assert result['status'] == 'completed'
        assert result['documents_created'] > 0
        assert output_dir.exists()

    @pytest.mark.asyncio
    async def test_validation_handler_validates_documents(self, case_context, temp_workspace):
        """Test VALIDATION stage validates documents."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # Setup
        await registry.handle_intake_stage(case_context, {})
        await registry.handle_analysis_stage(case_context, {})
        await registry.handle_generation_stage(case_context, {'output_dir': temp_workspace / "documents"})

        # Validate
        result = await registry.handle_validation_stage(case_context, {})

        assert 'valid' in result
        assert isinstance(result['valid'], bool)

    @pytest.mark.asyncio
    async def test_warboarding_handler_creates_visualizations(self, case_context, temp_workspace):
        """Test WARBOARDING stage creates visualizations."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # Setup
        await registry.handle_intake_stage(case_context, {})

        # Warboard
        output_dir = temp_workspace / "warboards"
        result = await registry.handle_warboarding_stage(case_context, {'output_dir': output_dir})

        assert result['status'] == 'completed'
        assert output_dir.exists()

    @pytest.mark.asyncio
    async def test_discovery_handler_creates_requests(self, case_context, temp_workspace):
        """Test DISCOVERY stage creates discovery documents."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # Setup
        await registry.handle_intake_stage(case_context, {})

        # Discovery
        output_dir = temp_workspace / "discovery"
        result = await registry.handle_discovery_stage(case_context, {'output_dir': output_dir})

        assert result['status'] == 'completed'
        assert output_dir.exists()

    @pytest.mark.asyncio
    async def test_filing_handler_bundles_documents(self, case_context, temp_workspace):
        """Test FILING stage bundles documents."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # Setup
        await registry.handle_intake_stage(case_context, {})
        await registry.handle_generation_stage(case_context, {'output_dir': temp_workspace / "documents"})

        # Filing
        output_dir = temp_workspace / "filing"
        result = await registry.handle_filing_stage(case_context, {'output_dir': output_dir})

        assert result['status'] == 'completed'
        assert output_dir.exists()


# ============================================================================
# Handler Registry Tests
# ============================================================================

class TestHandlerRegistry:
    """Test handler registry functionality."""

    def test_registry_has_builtin_handlers(self):
        """Test registry loads built-in handlers."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()
        expected_handlers = [
            'intake', 'analysis', 'organization', 'generation',
            'validation', 'warboarding', 'discovery', 'filing'
        ]

        for handler_type in expected_handlers:
            assert handler_type in registry.handlers
            assert callable(registry.handlers[handler_type])

    def test_registry_can_register_custom_handler(self):
        """Test registering custom handlers."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        def custom_handler(context, config):
            return {'status': 'custom'}

        registry.register('custom', custom_handler)
        assert 'custom' in registry.handlers
        assert registry.handlers['custom'] == custom_handler

    @pytest.mark.asyncio
    async def test_registry_dispatches_to_handler(self, case_context):
        """Test dispatcher routes to correct handler."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()
        result = await registry.dispatch('intake', case_context, {})

        assert 'status' in result


# ============================================================================
# Case Context Tests
# ============================================================================

class TestCaseContext:
    """Test case context data structure."""

    def test_case_context_initialization(self):
        """Test CaseContext initializes correctly."""
        from src.master_integration_bridge import CaseContext

        context = CaseContext(
            case_id="TEST001",
            case_type="custody",
            case_number="2025-001234-CZ",
            root_directories=[Path(".")],
        )

        assert context.case_id == "TEST001"
        assert context.case_type == "custody"
        assert context.evidence_files == []
        assert context.file_hashes == {}

    def test_case_context_with_custom_parties(self):
        """Test CaseContext with custom parties."""
        from src.master_integration_bridge import CaseContext

        parties = {"plaintiff": "John Doe", "defendant": "Jane Doe"}
        context = CaseContext(
            case_id="TEST001",
            case_type="custody",
            case_number="2025-001234-CZ",
            root_directories=[Path(".")],
            parties=parties,
        )

        assert context.parties == parties


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationTests:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_custody_workflow_full_execution(self, case_context, sample_evidence_files, temp_workspace):
        """Test full custody workflow execution."""
        from src.master_integration_bridge import get_handler_registry

        registry = get_handler_registry()

        # Execute stages in order
        results = {}

        # 1. Intake
        results['intake'] = await registry.handle_intake_stage(case_context, {})
        assert results['intake']['status'] == 'completed'

        # 2. Analysis
        results['analysis'] = await registry.handle_analysis_stage(case_context, {})
        assert results['analysis']['status'] == 'completed'

        # 3. Organization
        results['organization'] = await registry.handle_organization_stage(
            case_context,
            {'output_dir': temp_workspace / "exhibits"}
        )
        assert results['organization']['status'] == 'completed'

        # 4. Generation
        results['generation'] = await registry.handle_generation_stage(
            case_context,
            {'output_dir': temp_workspace / "documents"}
        )
        assert results['generation']['status'] == 'completed'

        # 5. Validation
        results['validation'] = await registry.handle_validation_stage(case_context, {})
        assert 'valid' in results['validation']

        # 6. Warboarding
        results['warboarding'] = await registry.handle_warboarding_stage(
            case_context,
            {'output_dir': temp_workspace / "warboards"}
        )
        assert results['warboarding']['status'] == 'completed'

        # 7. Filing
        results['filing'] = await registry.handle_filing_stage(
            case_context,
            {'output_dir': temp_workspace / "filing"}
        )
        assert results['filing']['status'] == 'completed'

    @pytest.mark.asyncio
    async def test_workflow_with_multiple_case_types(self, temp_workspace, sample_evidence_files):
        """Test workflows with different case types."""
        from src.master_integration_bridge import CaseContext, get_handler_registry

        registry = get_handler_registry()
        case_types = ['custody', 'housing', 'ppo']

        for case_type in case_types:
            context = CaseContext(
                case_id=f"TEST_{case_type}",
                case_type=case_type,
                case_number=f"2025-{case_type[:3].upper()}-001",
                root_directories=[temp_workspace / "evidence"],
            )

            result = await registry.handle_intake_stage(context, {})
            assert result['status'] == 'completed'

    @pytest.mark.asyncio
    async def test_error_handling_on_missing_evidence(self, temp_workspace):
        """Test error handling when evidence directory missing."""
        from src.master_integration_bridge import CaseContext, get_handler_registry

        registry = get_handler_registry()
        context = CaseContext(
            case_id="TEST_ERROR",
            case_type="custody",
            case_number="2025-ERROR-001",
            root_directories=[temp_workspace / "nonexistent"],
        )

        # Should not raise, just log warning
        result = await registry.handle_intake_stage(context, {})
        assert result['files_ingested'] == 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and benchmark tests."""

    @pytest.mark.asyncio
    async def test_intake_performance_with_many_files(self, case_context, temp_workspace):
        """Benchmark intake handler performance."""
        from src.master_integration_bridge import get_handler_registry
        import time

        registry = get_handler_registry()

        # Create 100 sample files
        evidence_dir = temp_workspace / "perf_evidence"
        evidence_dir.mkdir()

        for i in range(100):
            (evidence_dir / f"file_{i:04d}.txt").write_text(f"Content {i}")

        context = CaseContext(
            case_id="PERF001",
            case_type="custody",
            case_number="2025-PERF-001",
            root_directories=[evidence_dir],
        )

        start = time.time()
        result = await registry.handle_intake_stage(context, {})
        elapsed = time.time() - start

        assert result['files_ingested'] == 100
        assert elapsed < 10.0  # Should complete within 10 seconds

    @pytest.mark.asyncio
    async def test_full_workflow_performance(self, case_context, sample_evidence_files, temp_workspace):
        """Benchmark full workflow execution."""
        from src.master_integration_bridge import get_handler_registry
        import time

        registry = get_handler_registry()

        start = time.time()

        # Execute all stages
        await registry.handle_intake_stage(case_context, {})
        await registry.handle_analysis_stage(case_context, {})
        await registry.handle_organization_stage(case_context, {'output_dir': temp_workspace / "exhibits"})
        await registry.handle_generation_stage(case_context, {'output_dir': temp_workspace / "documents"})
        await registry.handle_validation_stage(case_context, {})
        await registry.handle_warboarding_stage(case_context, {'output_dir': temp_workspace / "warboards"})
        await registry.handle_filing_stage(case_context, {'output_dir': temp_workspace / "filing"})

        elapsed = time.time() - start

        # Full workflow should complete quickly
        assert elapsed < 60.0  # Within 60 seconds


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
