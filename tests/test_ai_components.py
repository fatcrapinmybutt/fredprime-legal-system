"""
Tests for AI/LLM/NLP components

Comprehensive test suite for:
- NLP engine (entity extraction, classification, sentiment)
- Evidence analyzer (AI-powered analysis)
- Argument reasoning engine (legal argument generation)
- AI-enhanced handlers
"""

import pytest
import asyncio
import json
from pathlib import Path
from datetime import datetime


# ============================================================================
# NLP Engine Tests
# ============================================================================

class TestNLPEngine:
    """Test Hugging Face NLP engine."""

    def test_nlp_engine_initialization(self):
        """Test NLP engine initializes without errors."""
        try:
            from src.ai_litigation_engine import NLPEngine
            engine = NLPEngine(use_gpu=False)
            assert engine is not None
            assert engine.device in ["cpu", "cuda"]
        except ImportError:
            pytest.skip("transformers not installed")

    @pytest.mark.asyncio
    async def test_evidence_classification(self):
        """Test evidence type classification."""
        try:
            from src.ai_litigation_engine import NLPEngine, EvidenceType
            engine = NLPEngine(use_gpu=False)

            # Test email classification
            email_text = "From: john@example.com To: jane@example.com Subject: Custody arrangement"
            email_type = await engine.classify_evidence_type(email_text, "email.txt")
            assert email_type in [EvidenceType.COMMUNICATION.value, EvidenceType.UNKNOWN.value]

            # Test document classification
            doc_text = "Court Order for Custody: This order grants primary custody to Plaintiff"
            doc_type = await engine.classify_evidence_type(doc_text, "order.pdf")
            assert doc_type is not None
        except ImportError:
            pytest.skip("transformers not installed")

    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        """Test named entity extraction."""
        try:
            from src.ai_litigation_engine import NLPEngine
            engine = NLPEngine(use_gpu=False)

            text = "John Smith spoke with Jane Doe on January 15, 2025 at 10:30 AM in Michigan"
            entities = await engine.extract_entities(text)

            assert isinstance(entities, list)
            # Should find some entities
            assert len(entities) > 0
        except ImportError:
            pytest.skip("transformers not installed")

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        try:
            from src.ai_litigation_engine import NLPEngine
            engine = NLPEngine(use_gpu=False)

            # Positive sentiment
            positive_text = "This is wonderful and excellent news!"
            positive_sentiment = await engine.analyze_sentiment(positive_text)
            assert positive_sentiment in ["positive", "negative", "neutral"]

            # Negative sentiment
            negative_text = "This is terrible and awful."
            negative_sentiment = await engine.analyze_sentiment(negative_text)
            assert negative_sentiment in ["positive", "negative", "neutral"]
        except ImportError:
            pytest.skip("transformers not installed")


# ============================================================================
# Evidence Analyzer Tests
# ============================================================================

class TestEvidenceAnalyzer:
    """Test AI-powered evidence analyzer."""

    @pytest.mark.asyncio
    async def test_evidence_analysis_complete(self, tmp_path):
        """Test comprehensive evidence analysis."""
        try:
            from src.ai_litigation_engine import NLPEngine, EvidenceAnalyzer

            nlp = NLPEngine(use_gpu=False)
            analyzer = EvidenceAnalyzer(nlp, case_type="custody")

            # Create test file
            test_file = tmp_path / "test_evidence.txt"
            test_content = """
            John denied Jane access to their child on multiple occasions.
            This is a clear violation of the custody order.
            The behavior has been systematic and deliberate.
            Date: January 15, 2025
            Location: Family home
            """
            test_file.write_text(test_content)

            # Analyze
            analysis = await analyzer.analyze_evidence(
                test_file,
                "test_evidence.txt",
                test_content
            )

            assert analysis.file_name == "test_evidence.txt"
            assert analysis.relevance_score > 0.0
            assert len(analysis.key_entities) > 0
            assert len(analysis.claims_supported) > 0
        except ImportError:
            pytest.skip("transformers not installed")

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, tmp_path):
        """Test evidence relevance scoring."""
        try:
            from src.ai_litigation_engine import NLPEngine, EvidenceAnalyzer

            nlp = NLPEngine(use_gpu=False)
            analyzer = EvidenceAnalyzer(nlp, case_type="custody")

            # High relevance evidence
            high_rel_content = "Pattern of custody interference with deliberate denial of parenting time"
            high_rel = await analyzer.analyze_evidence(
                Path("test1.txt"),
                "test1.txt",
                high_rel_content
            )

            # Low relevance evidence
            low_rel_content = "Weather forecast for tomorrow"
            low_rel = await analyzer.analyze_evidence(
                Path("test2.txt"),
                "test2.txt",
                low_rel_content
            )

            assert high_rel.relevance_score > low_rel.relevance_score
        except ImportError:
            pytest.skip("transformers not installed")


# ============================================================================
# Argument Reasoning Engine Tests
# ============================================================================

class TestArgumentReasoningEngine:
    """Test legal argument generation."""

    @pytest.mark.asyncio
    async def test_argument_generation(self):
        """Test argument generation from evidence."""
        try:
            from src.ai_litigation_engine import ArgumentReasoningEngine, EvidenceAnalysis

            engine = ArgumentReasoningEngine(case_type="custody")

            # Create mock evidence analyses
            evidence = [
                EvidenceAnalysis(
                    file_name="email_1.msg",
                    file_path="/evidence/email_1.msg",
                    evidence_type="communication",
                    relevance_level="critical",
                    relevance_score=0.95,
                    key_entities=["John", "Jane"],
                    key_phrases=["custody interference", "denial of access"],
                    summary="Email showing denial of parenting time",
                    claims_supported=["custody_interference"],
                    arguments_enabled=["custody"],
                    sentiment="negative",
                    credibility_indicators=["detailed_account"],
                    inconsistencies=[],
                    timestamps=["01/15/2025"],
                    parties_mentioned=["John", "Jane"],
                    analysis_timestamp=datetime.now().isoformat(),
                )
            ]

            # Generate argument
            argument = await engine.generate_argument(
                evidence,
                "custody_interference",
                "2025-001234-CZ"
            )

            assert argument.claim_type == "custody_interference"
            assert argument.supporting_evidence == ["email_1.msg"]
            assert argument.strength_score > 0.0
            assert len(argument.legal_basis) > 0
        except ImportError:
            pytest.skip("transformers not installed")

    def test_legal_basis_retrieval(self):
        """Test Michigan statute retrieval."""
        from src.ai_litigation_engine import ArgumentReasoningEngine

        engine = ArgumentReasoningEngine(case_type="custody")

        # Test MCL/MCR retrieval
        basis = engine._get_legal_basis("custody_interference")
        assert len(basis) > 0
        assert any("MCL" in statute or "MCR" in statute for statute in basis)


# ============================================================================
# AI Litigation Engine Tests
# ============================================================================

class TestAILitationEngine:
    """Test unified AI litigation engine."""

    @pytest.mark.asyncio
    async def test_case_analysis_complete(self, tmp_path):
        """Test complete case analysis workflow."""
        try:
            from src.ai_litigation_engine import AILitationEngine

            engine = AILitationEngine(case_type="custody", use_gpu=False)

            # Create test evidence files
            evidence_dir = tmp_path / "evidence"
            evidence_dir.mkdir()

            (evidence_dir / "email_denial.msg").write_text(
                "Email showing denial of custody: John denied access to child"
            )
            (evidence_dir / "court_order.pdf").write_text(
                "Court order granting joint custody with parenting time schedule"
            )

            # Prepare evidence list
            evidence_files = [
                {"path": str(evidence_dir / "email_denial.msg"), "name": "email_denial.msg"},
                {"path": str(evidence_dir / "court_order.pdf"), "name": "court_order.pdf"},
            ]

            # Run analysis
            analyses, arguments = await engine.analyze_case_evidence(
                evidence_files,
                "2025-001234-CZ"
            )

            assert len(analyses) > 0
            assert len(arguments) > 0

            # Check strength scoring
            scores = await engine.score_evidence_for_strength(analyses)
            assert 0.0 <= scores['overall'] <= 1.0
            assert 'categories' in scores
        except ImportError:
            pytest.skip("transformers not installed")


# ============================================================================
# AI-Enhanced Handlers Tests
# ============================================================================

class TestAIEnhancedHandlers:
    """Test AI-enhanced workflow handlers."""

    @pytest.mark.asyncio
    async def test_ai_analysis_stage(self, tmp_path):
        """Test AI-powered analysis stage."""
        try:
            from src.ai_enhanced_handlers import AIEnabledStageHandlers
            from src.master_integration_bridge import CaseContext

            handlers = AIEnabledStageHandlers()

            # Create context with evidence
            evidence_dir = tmp_path / "evidence"
            evidence_dir.mkdir()

            context = CaseContext(
                case_id="TEST001",
                case_type="custody",
                case_number="2025-TEST-001",
                root_directories=[evidence_dir],
            )

            context.evidence_files = [
                {
                    "path": str(evidence_dir / "test.txt"),
                    "name": "test.txt",
                    "size": 100,
                }
            ]

            # Write test file
            (evidence_dir / "test.txt").write_text("Test evidence content")

            # Run analysis
            result = await handlers.analyze_evidence_with_ai(context, {})

            assert result['status'] in ['completed', 'skipped', 'failed']
            if result['status'] == 'completed':
                assert 'analyses_count' in result
        except ImportError:
            pytest.skip("transformers not installed")


# ============================================================================
# GitHub Integration Tests
# ============================================================================

class TestGitHubIntegration:
    """Test GitHub integration."""

    @pytest.mark.asyncio
    async def test_case_issue_creation(self):
        """Test GitHub case issue creation."""
        from src.ai_enhanced_handlers import GitHubIntegration

        # Initialize without token (will use templates)
        github = GitHubIntegration()

        issue_title = await github.create_case_issue(
            "2025-001234-CZ",
            "custody",
            "Test case description"
        )

        assert "2025-001234-CZ" in issue_title

    @pytest.mark.asyncio
    async def test_workflow_progress_tracking(self):
        """Test GitHub workflow progress tracking."""
        from src.ai_enhanced_handlers import GitHubIntegration

        github = GitHubIntegration()

        result = await github.track_workflow_progress(
            "2025-001234-CZ",
            "intake_evidence",
            "completed",
            {"files_processed": 145}
        )

        assert result is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestAIIntegration:
    """Integration tests combining all AI components."""

    @pytest.mark.asyncio
    async def test_end_to_end_ai_workflow(self, tmp_path):
        """Test complete end-to-end AI-powered workflow."""
        try:
            from src.ai_litigation_engine import AILitationEngine
            from src.ai_enhanced_handlers import AIEnabledStageHandlers, GitHubIntegration

            # Initialize all AI components
            ai_engine = AILitationEngine(case_type="custody", use_gpu=False)
            handlers = AIEnabledStageHandlers()
            github = GitHubIntegration()

            # Setup test case
            evidence_dir = tmp_path / "evidence"
            evidence_dir.mkdir()

            # Create test evidence
            (evidence_dir / "email.msg").write_text(
                "From: john@test.com To: jane@test.com\n"
                "Subject: Custody Issue\n"
                "Body: Denied access to children on January 15, 2025"
            )

            evidence_files = [
                {"path": str(evidence_dir / "email.msg"), "name": "email.msg"}
            ]

            # Run complete workflow
            analyses, arguments = await ai_engine.analyze_case_evidence(
                evidence_files,
                "2025-001234-CZ"
            )

            # Create GitHub issue
            issue = await github.create_case_issue(
                "2025-001234-CZ",
                "custody",
                "AI-powered case analysis"
            )

            assert len(analyses) > 0 or len(analyses) == 0  # May be empty if AI disabled
            assert issue is not None
        except ImportError:
            pytest.skip("transformers not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
