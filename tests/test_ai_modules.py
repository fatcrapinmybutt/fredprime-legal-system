"""
Comprehensive test suite for AI/ML modules
Tests for LLM, NLP, ARG systems and AI pipeline orchestrator
"""

import pytest
import json

# Import AI modules
from ai.evidence_llm_analyzer import (
    EvidenceLLMAnalyzer,
    EvidenceType,
    AnalyzedEvidence
)
from ai.nlp_document_processor import (
    NLPDocumentProcessor,
    DocumentType,
    SentimentType,
    DocumentMetadata
)
from ai.argument_reasoning import (
    ArgumentReasoningGraph,
    ArgumentType,
    RelationType
)
from ai.ai_pipeline_orchestrator import (
    AIPipelineOrchestrator,
    ProcessingStage,
    PipelineStatus
)
from integrations.github_integration import (
    GitHubAPIClient,
    IssueState,
    PRState
)


class TestEvidenceLLMAnalyzer:
    """Test evidence LLM analyzer"""

    @pytest.fixture
    def analyzer(self) -> EvidenceLLMAnalyzer:
        return EvidenceLLMAnalyzer()

    def test_initialization(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.model_name == "distilbert-base-uncased-finetuned-sst-2-english"

    def test_evidence_type_classification(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test evidence type classification"""
        # Documentary evidence
        result = analyzer.analyze_evidence(
            "ev001",
            "This is a document email containing important information"
        )
        assert result.evidence_type == EvidenceType.DOCUMENTARY

        # Testimonial evidence
        result = analyzer.analyze_evidence(
            "ev002",
            "The witness testified that they saw the defendant violate the order"
        )
        assert result.evidence_type == EvidenceType.TESTIMONIAL

    def test_semantic_summary_extraction(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test semantic summary extraction"""
        content = "This is the first sentence. This is the second sentence. More content here."
        result = analyzer.analyze_evidence("ev001", content)
        assert result.semantic_summary is not None
        assert len(result.semantic_summary) > 0

    def test_key_phrase_extraction(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test key phrase extraction"""
        content = "custody modification motion regarding parental fitness and child welfare"
        result = analyzer.analyze_evidence("ev001", content)
        assert result.key_phrases is not None
        assert len(result.key_phrases) > 0

    def test_entity_extraction(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test entity extraction"""
        content = "Judge Smith ordered that Mr. John Doe appear in Michigan Court on January 15, 2024"
        result = analyzer.analyze_evidence("ev001", content)
        assert len(result.extracted_entities) > 0
        # Check for person entities
        person_entities = [e for e in result.extracted_entities if e.entity_type == 'PERSON']
        assert len(person_entities) > 0

    def test_evidence_scoring(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test evidence scoring"""
        content = "Credible witness testimony with detailed evidence"
        result = analyzer.analyze_evidence("ev001", content, case_type="custody")
        assert result.scores is not None
        assert 0 <= result.scores.relevance_score <= 1
        assert 0 <= result.scores.reliability_score <= 1
        assert 0 <= result.scores.overall_strength <= 1

    def test_batch_analysis(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test batch evidence analysis"""
        evidence_list = [
            ("ev001", "First piece of evidence"),
            ("ev002", "Second piece of evidence"),
            ("ev003", "Third piece of evidence"),
        ]
        results = analyzer.batch_analyze_evidence(evidence_list)
        assert len(results) == 3
        assert all(isinstance(r, AnalyzedEvidence) for r in results)

    def test_evidence_comparison(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test evidence comparison"""
        ev1 = analyzer.analyze_evidence("ev001", "Testimony supporting the claim")
        ev2 = analyzer.analyze_evidence("ev002", "Documentation confirming the testimony")

        comparison = analyzer.compare_evidence(ev1, ev2)
        assert "evidence1_id" in comparison
        assert "evidence2_id" in comparison
        assert "recommendation" in comparison

    def test_export_json(self, analyzer: EvidenceLLMAnalyzer) -> None:
        """Test JSON export"""
        evidence_list = [
            analyzer.analyze_evidence("ev001", "Test evidence"),
            analyzer.analyze_evidence("ev002", "Another evidence"),
        ]
        json_output = analyzer.export_analysis_report(evidence_list, "json")
        assert json_output is not None
        data = json.loads(json_output)
        assert len(data) == 2


class TestNLPDocumentProcessor:
    """Test NLP document processor"""

    @pytest.fixture
    def processor(self) -> NLPDocumentProcessor:
        return NLPDocumentProcessor()

    def test_initialization(self, processor: NLPDocumentProcessor) -> None:
        """Test processor initialization"""
        assert processor is not None

    def test_document_type_classification(self, processor: NLPDocumentProcessor) -> None:
        """Test document type classification"""
        # Motion
        motion_text = "MOTION FOR MODIFICATION OF CUSTODY\n\nPlaintiff moves that the court modify..."
        result = processor.process_document(motion_text, "Motion1")
        assert result.document_type == DocumentType.MOTION

        # Affidavit
        affidavit_text = "AFFIDAVIT\n\nState of Michigan\nCounty of Wayne\nI, John Smith, being duly sworn..."
        result = processor.process_document(affidavit_text, "Affidavit1")
        assert result.document_type == DocumentType.AFFIDAVIT

    def test_entity_extraction(self, processor: NLPDocumentProcessor) -> None:
        """Test entity extraction from documents"""
        content = "Judge Smith ordered on January 15, 2024 that the defendant, John Doe, appear in court"
        result = processor.process_document(content, "Doc1")
        assert len(result.entities) > 0

    def test_party_extraction(self, processor: NLPDocumentProcessor) -> None:
        """Test party extraction"""
        content = "Plaintiff: Jane Smith v. Defendant: John Doe"
        result = processor.process_document(content, "Doc1")
        assert len(result.parties_involved) > 0

    def test_sentiment_analysis(self, processor: NLPDocumentProcessor) -> None:
        """Test sentiment analysis"""
        # Positive document
        positive_text = "The defendant demonstrated excellent compliance with the court order"
        result = processor.process_document(positive_text, "Doc1")
        assert result.sentiment in [SentimentType.POSITIVE, SentimentType.HIGHLY_POSITIVE]

        # Negative document
        negative_text = "The defendant violated the order and caused harm to the child"
        result = processor.process_document(negative_text, "Doc2")
        assert result.sentiment in [SentimentType.NEGATIVE, SentimentType.HIGHLY_NEGATIVE]

    def test_key_concept_extraction(self, processor: NLPDocumentProcessor) -> None:
        """Test key concept extraction"""
        content = "Custody modification motion regarding visitation and child support"
        result = processor.process_document(content, "Doc1")
        assert len(result.key_concepts) > 0
        assert any('custody' in c for c in result.key_concepts)

    def test_action_item_extraction(self, processor: NLPDocumentProcessor) -> None:
        """Test action item extraction"""
        content = "The defendant shall appear in court by January 15, 2024. The defendant must comply with visitation."
        result = processor.process_document(content, "Doc1")
        assert len(result.action_items) > 0

    def test_deadline_extraction(self, processor: NLPDocumentProcessor) -> None:
        """Test deadline extraction"""
        content = "The defendant must file a response by January 15, 2024, or within 21 days of service."
        result = processor.process_document(content, "Doc1")
        assert len(result.deadlines) > 0

    def test_batch_processing(self, processor: NLPDocumentProcessor) -> None:
        """Test batch document processing"""
        documents = [
            ("doc1", "MOTION FOR MODIFICATION\n\nPlaintiff seeks to modify..."),
            ("doc2", "AFFIDAVIT\n\nI, John Smith, state under oath..."),
            ("doc3", "ANSWER AND AFFIRMATIVE DEFENSES\n\nDefendant denies..."),
        ]
        results = processor.batch_process_documents(documents)
        assert len(results) == 3
        assert all(isinstance(r, DocumentMetadata) for r in results)

    def test_summary_report(self, processor: NLPDocumentProcessor) -> None:
        """Test summary report generation"""
        documents = [
            ("doc1", "MOTION FOR MODIFICATION"),
            ("doc2", "AFFIDAVIT"),
        ]
        results = processor.batch_process_documents(documents)
        report = processor.generate_summary_report(results)

        assert report["total_documents"] == 2
        assert "document_types" in report
        assert "sentiment_distribution" in report


class TestArgumentReasoningGraph:
    """Test Argument Reasoning Graph system"""

    @pytest.fixture
    def arg_system(self) -> ArgumentReasoningGraph:
        return ArgumentReasoningGraph()

    def test_initialization(self, arg_system: ArgumentReasoningGraph) -> None:
        """Test ARG system initialization"""
        assert arg_system is not None
        assert len(arg_system.nodes) == 0
        assert len(arg_system.edges) == 0

    def test_node_creation(self, arg_system: ArgumentReasoningGraph) -> None:
        """Test argument node creation"""
        node = arg_system.create_node(
            text="The defendant violated the custody order",
            arg_type=ArgumentType.CLAIM,
            source="case_facts",
            confidence=0.85
        )
        assert node is not None
        assert node.text == "The defendant violated the custody order"
        assert node.arg_type == ArgumentType.CLAIM
        assert node.confidence == 0.85

    def test_edge_creation(self, arg_system: ArgumentReasoningGraph) -> None:
        """Test argument edge creation"""
        node1 = arg_system.create_node("Evidence 1", ArgumentType.EVIDENCE, "source1", 0.8)
        node2 = arg_system.create_node("Claim 1", ArgumentType.CLAIM, "source2", 0.9)

        edge = arg_system.create_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            relation_type=RelationType.SUPPORTS,
            strength=0.85,
            reasoning="This evidence supports the claim"
        )
        assert edge is not None
        assert edge.relation_type == RelationType.SUPPORTS

    def test_case_analysis(self, arg_system: ArgumentReasoningGraph) -> None:
        """Test case analysis"""
        # Create nodes
        main_claim = arg_system.create_node(
            "Defendant violated custody order",
            ArgumentType.CLAIM,
            "facts",
            0.9
        )
        evidence = arg_system.create_node(
            "Witness testimony",
            ArgumentType.EVIDENCE,
            "witness",
            0.85
        )

        # Create edge
        arg_system.create_edge(
            evidence.node_id,
            main_claim.node_id,
            RelationType.SUPPORTS,
            0.85
        )

        # Analyze
        analysis = arg_system.analyze_case(
            case_id="case_001",
            case_title="Test Case",
            evidence_nodes=[evidence],
            argument_nodes=[main_claim],
            edges=list(arg_system.edges.values())
        )

        assert analysis is not None
        assert analysis.main_claim is not None
        assert len(analysis.supporting_arguments) >= 0

    def test_vulnerability_identification(self, arg_system: ArgumentReasoningGraph) -> None:
        """Test vulnerability identification"""
        # Create weak main claim
        weak_claim = arg_system.create_node(
            "Weak claim with little support",
            ArgumentType.CLAIM,
            "source",
            0.5
        )

        analysis = arg_system.analyze_case(
            case_id="case_001",
            case_title="Weak Case",
            evidence_nodes=[],
            argument_nodes=[weak_claim],
            edges=[]
        )

        assert len(analysis.vulnerabilities) > 0

    def test_export_text_format(self, arg_system: ArgumentReasoningGraph) -> None:
        """Test text format export"""
        claim = arg_system.create_node("Test claim", ArgumentType.CLAIM, "source", 0.8)

        analysis = arg_system.analyze_case(
            case_id="case_001",
            case_title="Test Case",
            evidence_nodes=[],
            argument_nodes=[claim],
            edges=[]
        )

        text_output = arg_system.export_analysis(analysis, "text")
        assert text_output is not None
        assert "ARGUMENT ANALYSIS REPORT" in text_output


class TestAIPipelineOrchestrator:
    """Test AI Pipeline Orchestrator"""

    @pytest.fixture
    def orchestrator(self) -> AIPipelineOrchestrator:
        return AIPipelineOrchestrator()

    def test_initialization(self, orchestrator: AIPipelineOrchestrator) -> None:
        """Test orchestrator initialization"""
        assert orchestrator is not None
        assert orchestrator.evidence_analyzer is not None
        assert orchestrator.document_processor is not None
        assert orchestrator.arg_system is not None

    def test_case_processing(self, orchestrator: AIPipelineOrchestrator) -> None:
        """Test complete case processing"""
        evidence = [
            ("ev001", "Witness testimony about custody violation"),
            ("ev002", "Documentation of missed visitation"),
        ]

        documents = [
            ("doc001", "MOTION FOR MODIFICATION\n\nPlaintiff seeks modification of custody."),
            ("doc002", "AFFIDAVIT\n\nI state under oath that defendant violated the order."),
        ]

        report = orchestrator.process_case(
            case_id="case_001",
            case_title="Custody Modification",
            evidence_items=evidence,
            documents=documents,
            case_type="custody"
        )

        assert report is not None
        assert report.case_id == "case_001"
        assert len(report.evidence_analyses) > 0
        assert len(report.document_analyses) > 0
        assert len(report.key_findings) > 0

    def test_stage_results(self, orchestrator: AIPipelineOrchestrator) -> None:
        """Test that all stages produce results"""
        evidence = [("ev001", "Test evidence")]
        documents = [("doc001", "Test document")]

        report = orchestrator.process_case(
            case_id="case_001",
            case_title="Test Case",
            evidence_items=evidence,
            documents=documents
        )

        assert len(report.stage_results) > 0
        stages_found = {r.stage for r in report.stage_results}
        assert ProcessingStage.EVIDENCE_ANALYSIS in stages_found
        assert ProcessingStage.DOCUMENT_PROCESSING in stages_found

    def test_export_json(self, orchestrator: AIPipelineOrchestrator) -> None:
        """Test JSON export"""
        evidence = [("ev001", "Test evidence")]
        documents = [("doc001", "Test document")]

        report = orchestrator.process_case(
            case_id="case_001",
            case_title="Test Case",
            evidence_items=evidence,
            documents=documents
        )

        json_output = orchestrator.export_report(report, "json")
        assert json_output is not None
        data = json.loads(json_output)
        assert data["case_id"] == "case_001"

    def test_export_text(self, orchestrator: AIPipelineOrchestrator) -> None:
        """Test text export"""
        evidence = [("ev001", "Test evidence")]
        documents = [("doc001", "Test document")]

        report = orchestrator.process_case(
            case_id="case_001",
            case_title="Test Case",
            evidence_items=evidence,
            documents=documents
        )

        text_output = orchestrator.export_report(report, "text")
        assert text_output is not None
        assert "AI LITIGATION ANALYSIS REPORT" in text_output


class TestGitHubIntegration:
    """Test GitHub integration"""

    @pytest.fixture
    def client(self) -> GitHubAPIClient:
        # Note: This will use unauthenticated client if no token
        return GitHubAPIClient(owner="test", repo="test-repo")

    def test_initialization(self, client: GitHubAPIClient) -> None:
        """Test client initialization"""
        assert client is not None
        assert client.owner == "test"
        assert client.repo == "test-repo"

    def test_issue_state_enum(self) -> None:
        """Test issue state enum"""
        assert IssueState.OPEN.value == "open"
        assert IssueState.CLOSED.value == "closed"

    def test_pr_state_enum(self) -> None:
        """Test PR state enum"""
        assert PRState.OPEN.value == "open"
        assert PRState.CLOSED.value == "closed"
        assert PRState.MERGED.value == "merged"


# Integration tests
class TestIntegration:
    """Integration tests for all components"""

    def test_end_to_end_analysis(self):
        """Test end-to-end case analysis"""
        orchestrator = AIPipelineOrchestrator()

        # Realistic case data
        evidence = [
            ("ev001", """
                Witness statement from January 15, 2024:
                On this date, I observed the defendant failing to return the child
                at the scheduled time. The child was returned at 8:30 PM instead of
                6:00 PM as ordered by the court.
            """),
            ("ev002", """
                Text message exchange showing missed pickup times
                on multiple occasions throughout 2024.
            """),
        ]

        documents = [
            ("doc001", """
                MOTION FOR ENFORCEMENT OF CUSTODY ORDER

                Plaintiff seeks enforcement of the custody order dated
                January 1, 2024, due to repeated violations by the defendant.

                The defendant has failed to comply with visitation schedules
                on multiple occasions.
            """),
            ("doc002", """
                AFFIDAVIT IN SUPPORT OF MOTION

                State of Michigan
                County of Wayne

                I, Jane Smith, being duly sworn, state that:

                1. I am the plaintiff in this action
                2. The defendant has repeatedly violated the custody order
                3. These violations have caused emotional harm to the child
            """),
        ]

        # Process complete case
        report = orchestrator.process_case(
            case_id="case_001",
            case_title="Custody Order Enforcement",
            evidence_items=evidence,
            documents=documents,
            case_type="custody"
        )

        # Verify results
        assert report.pipeline_status == PipelineStatus.COMPLETED
        assert len(report.evidence_analyses) == 2
        assert len(report.document_analyses) == 2
        assert len(report.key_findings) > 0
        assert report.confidence_score > 0

    def test_component_integration(self):
        """Test that all components work together"""
        # Create instances
        analyzer = EvidenceLLMAnalyzer()
        processor = NLPDocumentProcessor()
        arg_system = ArgumentReasoningGraph()

        # Process evidence
        evidence = analyzer.analyze_evidence(
            "ev001",
            "Testimony showing defendant non-compliance"
        )
        assert evidence is not None

        # Process document
        doc_metadata = processor.process_document(
            "MOTION FOR MODIFICATION",
            "Motion1"
        )
        assert doc_metadata is not None

        # Create argument
        claim_node = arg_system.create_node(
            evidence.semantic_summary,
            ArgumentType.CLAIM,
            "evidence",
            confidence=evidence.confidence
        )
        assert claim_node is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
