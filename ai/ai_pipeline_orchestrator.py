"""
AI Pipeline Orchestrator
Unified orchestration of all AI/ML components for litigation case analysis.
Coordinates LLM, NLP, and ARG systems for comprehensive case understanding.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import concurrent.futures
from pathlib import Path

from ai.evidence_llm_analyzer import EvidenceLLMAnalyzer, AnalyzedEvidence
from ai.nlp_document_processor import NLPDocumentProcessor, DocumentMetadata
from ai.argument_reasoning import ArgumentReasoningGraph, ArgumentAnalysis, ArgumentNode, ArgumentType, ArgumentEdge, RelationType

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Stages of AI pipeline processing"""
    INTAKE = "intake"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    ARGUMENT_CONSTRUCTION = "argument_construction"
    REASONING = "reasoning"
    VALIDATION = "validation"
    REPORTING = "reporting"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageResult:
    """Result of a pipeline stage"""
    stage: ProcessingStage
    status: str
    duration_ms: int
    items_processed: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    results: Any = None


@dataclass
class AIAnalysisReport:
    """Comprehensive AI analysis report"""
    case_id: str
    case_title: str
    analysis_timestamp: str
    pipeline_status: PipelineStatus
    stage_results: List[StageResult] = field(default_factory=list)
    evidence_analyses: List[AnalyzedEvidence] = field(default_factory=list)
    document_analyses: List[DocumentMetadata] = field(default_factory=list)
    argument_analysis: Optional[ArgumentAnalysis] = None
    key_findings: List[str] = field(default_factory=list)
    critical_issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_seconds: float = 0.0

    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['pipeline_status'] = self.pipeline_status.value
        data['stage_results'] = [
            {
                **asdict(r),
                'stage': r.stage.value
            }
            for r in self.stage_results
        ]
        data['evidence_analyses'] = [e.to_dict() for e in self.evidence_analyses]
        data['document_analyses'] = [d.to_dict() for d in self.document_analyses]
        data['argument_analysis'] = self.argument_analysis.to_dict() if self.argument_analysis else None
        return data


class AIPipelineOrchestrator:
    """
    Orchestrates all AI/ML components for litigation case analysis.
    Manages the entire analysis pipeline from evidence intake through final reporting.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize the AI pipeline orchestrator

        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.evidence_analyzer = EvidenceLLMAnalyzer()
        self.document_processor = NLPDocumentProcessor()
        self.arg_system = ArgumentReasoningGraph()
        self.pipeline_status = PipelineStatus.INITIALIZED

    def process_case(
        self,
        case_id: str,
        case_title: str,
        evidence_items: List[Tuple[str, str]],
        documents: List[Tuple[str, str]],
        case_type: str = "general",
        case_context: Optional[Dict[str, Any]] = None
    ) -> AIAnalysisReport:
        """
        Process a complete litigation case through the AI pipeline

        Args:
            case_id: Unique case identifier
            case_title: Title of the case
            evidence_items: List of (evidence_id, content) tuples
            documents: List of (doc_id, content) tuples
            case_type: Type of case (custody, ppo, etc.)
            case_context: Additional context about the case

        Returns:
            Comprehensive AI analysis report
        """
        logger.info(f"Starting pipeline for case: {case_id}")
        case_context = case_context or {}
        start_time = datetime.now()
        report = AIAnalysisReport(
            case_id=case_id,
            case_title=case_title,
            analysis_timestamp=datetime.now().isoformat(),
            pipeline_status=PipelineStatus.RUNNING
        )

        try:
            # Stage 1: Evidence Analysis
            report.stage_results.append(
                self._process_evidence_stage(evidence_items, case_type, case_context)
            )
            report.evidence_analyses = report.stage_results[-1].results or []

            # Stage 2: Document Processing
            report.stage_results.append(
                self._process_document_stage(documents, case_context)
            )
            report.document_analyses = report.stage_results[-1].results or []

            # Stage 3: Argument Construction
            report.stage_results.append(
                self._construct_arguments(
                    report.evidence_analyses,
                    report.document_analyses,
                    case_type
                )
            )

            # Stage 4: Reasoning
            report.stage_results.append(
                self._perform_reasoning(report.evidence_analyses, report.document_analyses)
            )

            # Stage 5: Argument Analysis
            if report.stage_results[-1].results:
                arg_data = report.stage_results[-1].results
                report.argument_analysis = self._analyze_arguments(
                    case_id, case_title, arg_data
                )

            # Stage 6: Validation
            report.stage_results.append(
                self._validate_analysis(
                    report.evidence_analyses,
                    report.document_analyses,
                    report.argument_analysis
                )
            )

            # Stage 7: Generate Findings and Recommendations
            report.key_findings = self._extract_key_findings(
                report.evidence_analyses,
                report.argument_analysis
            )
            report.critical_issues = self._identify_critical_issues(
                report.evidence_analyses,
                report.argument_analysis
            )
            report.recommendations = self._generate_recommendations(
                report.argument_analysis,
                report.critical_issues
            )

            # Calculate overall confidence
            report.confidence_score = self._calculate_overall_confidence(report)

            # Finalize report
            report.pipeline_status = PipelineStatus.COMPLETED
            report.processing_time_seconds = (datetime.now() - start_time).total_seconds()

            logger.info(f"Pipeline completed for case {case_id} in {report.processing_time_seconds:.2f}s")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            report.pipeline_status = PipelineStatus.FAILED
            report.critical_issues.append({
                "type": "pipeline_error",
                "severity": "critical",
                "description": str(e)
            })

        return report

    def _process_evidence_stage(
        self,
        evidence_items: List[Tuple[str, str]],
        case_type: str,
        context: Dict[str, Any]
    ) -> StageResult:
        """Process evidence intake and analysis stage"""
        logger.info(f"Processing {len(evidence_items)} evidence items")
        start_time = datetime.now()
        errors = []
        results = []

        try:
            # Process evidence in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.evidence_analyzer.analyze_evidence,
                        evi_id, content, case_type, context
                    )
                    for evi_id, content in evidence_items
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Evidence analysis failed: {e}")
                        errors.append(str(e))

        except Exception as e:
            logger.error(f"Evidence stage failed: {e}")
            errors.append(str(e))

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return StageResult(
            stage=ProcessingStage.EVIDENCE_ANALYSIS,
            status="completed" if not errors else "completed_with_errors",
            duration_ms=duration_ms,
            items_processed=len(results),
            errors=errors,
            results=results
        )

    def _process_document_stage(
        self,
        documents: List[Tuple[str, str]],
        context: Dict[str, Any]
    ) -> StageResult:
        """Process document intake and NLP analysis stage"""
        logger.info(f"Processing {len(documents)} documents")
        start_time = datetime.now()
        errors = []
        results = []

        try:
            # Process documents in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.document_processor.process_document,
                        content, doc_id, context
                    )
                    for doc_id, content in documents
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Document processing failed: {e}")
                        errors.append(str(e))

        except Exception as e:
            logger.error(f"Document stage failed: {e}")
            errors.append(str(e))

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return StageResult(
            stage=ProcessingStage.DOCUMENT_PROCESSING,
            status="completed" if not errors else "completed_with_errors",
            duration_ms=duration_ms,
            items_processed=len(results),
            errors=errors,
            results=results
        )

    def _construct_arguments(
        self,
        evidence: List[AnalyzedEvidence],
        documents: List[DocumentMetadata],
        case_type: str
    ) -> StageResult:
        """Construct argument nodes from evidence and documents"""
        logger.info("Constructing argument nodes")
        start_time = datetime.now()
        arg_nodes = []

        try:
            # Create evidence nodes
            for ev in evidence:
                node = self.arg_system.create_node(
                    text=ev.semantic_summary or ev.content[:200],
                    arg_type=ArgumentType.EVIDENCE,
                    source=ev.evidence_id,
                    confidence=ev.scores.overall_strength if ev.scores else 0.5
                )
                arg_nodes.append(node)

            # Create claim nodes from documents
            for doc in documents:
                if doc.document_type.value in ["motion", "complaint", "answer"]:
                    node = self.arg_system.create_node(
                        text=doc.summary,
                        arg_type=ArgumentType.CLAIM,
                        source=doc.title,
                        confidence=0.75
                    )
                    arg_nodes.append(node)

            logger.info(f"Created {len(arg_nodes)} argument nodes")

        except Exception as e:
            logger.error(f"Argument construction failed: {e}")

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return StageResult(
            stage=ProcessingStage.ARGUMENT_CONSTRUCTION,
            status="completed",
            duration_ms=duration_ms,
            items_processed=len(arg_nodes),
            results=arg_nodes
        )

    def _perform_reasoning(
        self,
        evidence: List[AnalyzedEvidence],
        documents: List[DocumentMetadata]
    ) -> StageResult:
        """Perform logical reasoning over evidence and documents"""
        logger.info("Performing reasoning stage")
        start_time = datetime.now()

        try:
            # Create edges between evidence items
            edges = []

            # Connect related evidence
            for i, ev1 in enumerate(evidence):
                for ev2 in evidence[i + 1:]:
                    # Check if evidence is complementary
                    if ev1.evidence_type == ev2.evidence_type:
                        edge = self.arg_system.create_edge(
                            source_id=f"node_{i}",
                            target_id=f"node_{i + 1}",
                            relation_type=RelationType.SUPPORTS,
                            strength=0.7,
                            reasoning="Related evidence items"
                        )
                        edges.append(edge)

            logger.info(f"Created {len(edges)} reasoning edges")

            return StageResult(
                stage=ProcessingStage.REASONING,
                status="completed",
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                items_processed=len(edges),
                results={"edges": edges}
            )

        except Exception as e:
            logger.error(f"Reasoning stage failed: {e}")
            return StageResult(
                stage=ProcessingStage.REASONING,
                status="failed",
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                items_processed=0,
                errors=[str(e)]
            )

    def _analyze_arguments(
        self,
        case_id: str,
        case_title: str,
        arg_data: Dict[str, Any]
    ) -> Optional[ArgumentAnalysis]:
        """Analyze arguments using ARG system"""
        logger.info("Analyzing arguments")

        try:
            nodes = list(self.arg_system.nodes.values())
            edges = list(self.arg_system.edges.values())

            if nodes:
                evidence_nodes = [n for n in nodes if n.arg_type == ArgumentType.EVIDENCE]
                argument_nodes = [n for n in nodes if n.arg_type != ArgumentType.EVIDENCE]

                return self.arg_system.analyze_case(
                    case_id=case_id,
                    case_title=case_title,
                    evidence_nodes=evidence_nodes,
                    argument_nodes=argument_nodes,
                    edges=edges
                )

        except Exception as e:
            logger.error(f"Argument analysis failed: {e}")

        return None

    def _validate_analysis(
        self,
        evidence: List[AnalyzedEvidence],
        documents: List[DocumentMetadata],
        analysis: Optional[ArgumentAnalysis]
    ) -> StageResult:
        """Validate the overall analysis"""
        logger.info("Validating analysis")
        start_time = datetime.now()
        errors = []
        warnings = []

        # Validation checks
        if not evidence:
            warnings.append("No evidence items were processed")

        if not documents:
            warnings.append("No documents were processed")

        if not analysis:
            warnings.append("Argument analysis could not be completed")

        # Check evidence quality
        low_confidence_evidence = [e for e in evidence if e.confidence < 0.5]
        if low_confidence_evidence:
            warnings.append(
                f"{len(low_confidence_evidence)} evidence items have low confidence"
            )

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return StageResult(
            stage=ProcessingStage.VALIDATION,
            status="completed",
            duration_ms=duration_ms,
            items_processed=len(evidence) + len(documents),
            errors=errors,
            warnings=warnings
        )

    def _extract_key_findings(
        self,
        evidence: List[AnalyzedEvidence],
        analysis: Optional[ArgumentAnalysis]
    ) -> List[str]:
        """Extract key findings from analysis"""
        findings = []

        # Key findings from evidence
        high_confidence_evidence = [e for e in evidence if e.scores and e.scores.overall_strength > 0.8]
        if high_confidence_evidence:
            findings.append(
                f"Identified {len(high_confidence_evidence)} high-confidence evidence items"
            )

        # Key findings from argument analysis
        if analysis:
            findings.append(f"Main claim: {analysis.main_claim.text if analysis.main_claim else 'N/A'}")
            findings.append(f"Supporting arguments: {len(analysis.supporting_arguments)}")
            findings.append(f"Counter-arguments: {len(analysis.counter_arguments)}")
            findings.append(f"Overall strength: {analysis.overall_strength.value}")

        return findings

    def _identify_critical_issues(
        self,
        evidence: List[AnalyzedEvidence],
        analysis: Optional[ArgumentAnalysis]
    ) -> List[Dict[str, Any]]:
        """Identify critical issues in the case"""
        issues = []

        # Missing evidence
        low_conf_evidence = [e for e in evidence if e.scores and e.scores.overall_strength < 0.5]
        if len(low_conf_evidence) > len(evidence) * 0.3:
            issues.append({
                "type": "weak_evidence",
                "severity": "high",
                "description": f"{len(low_conf_evidence)} evidence items have low credibility"
            })

        # Argument vulnerabilities
        if analysis and analysis.vulnerabilities:
            for vuln in analysis.vulnerabilities:
                if vuln.get("severity") == "high":
                    issues.append(vuln)

        return issues

    def _generate_recommendations(
        self,
        analysis: Optional[ArgumentAnalysis],
        issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []

        if analysis:
            recommendations.extend(analysis.recommendations)

        # Address critical issues
        for issue in issues:
            if issue.get("severity") == "high":
                recommendations.append(f"Address {issue.get('type', 'issue')}: {issue.get('description', '')}")

        return recommendations[:10]  # Limit to top 10

    def _calculate_overall_confidence(self, report: AIAnalysisReport) -> float:
        """Calculate overall confidence in the analysis"""
        confidence = 0.5

        # Weight evidence analysis
        if report.evidence_analyses:
            avg_ev_confidence = sum(e.confidence for e in report.evidence_analyses) / len(report.evidence_analyses)
            confidence += avg_ev_confidence * 0.3

        # Weight document analysis
        if report.document_analyses:
            avg_doc_confidence = sum(d.confidence_score for d in report.document_analyses) / len(report.document_analyses)
            confidence += avg_doc_confidence * 0.2

        # Weight argument analysis
        if report.argument_analysis:
            confidence += report.argument_analysis.overall_score * 0.3

        # Penalize validation issues
        if report.stage_results:
            last_validation = next(
                (r for r in report.stage_results if r.stage == ProcessingStage.VALIDATION),
                None
            )
            if last_validation:
                warning_count = len(last_validation.warnings)
                confidence -= warning_count * 0.05

        return max(0.0, min(confidence, 1.0))

    def export_report(
        self,
        report: AIAnalysisReport,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """Export analysis report in various formats"""
        if format == "json":
            report_str = json.dumps(report.to_dict(), indent=2, default=str)
        elif format == "text":
            report_str = self._generate_text_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Report exported to {output_path}")

        return report_str

    def _generate_text_report(self, report: AIAnalysisReport) -> str:
        """Generate text format report"""
        text = f"""
        AI LITIGATION ANALYSIS REPORT
        =============================
        Case ID: {report.case_id}
        Case Title: {report.case_title}
        Analysis Date: {report.analysis_timestamp}
        Processing Time: {report.processing_time_seconds:.2f} seconds
        Overall Confidence: {report.confidence_score:.0%}

        PIPELINE STATUS
        ---------------
        Status: {report.pipeline_status.value}
        Stages Completed: {len(report.stage_results)}

        EVIDENCE ANALYSIS
        -----------------
        Items Analyzed: {len(report.evidence_analyses)}
        """

        # Add evidence summary
        for ev in report.evidence_analyses[:5]:
            text += f"\n- {ev.evidence_type.value.upper()}: {ev.semantic_summary[:100]}"
            if ev.scores:
                text += f" (Strength: {ev.scores.overall_strength:.0%})"

        text += f"\n\nDOCUMENT ANALYSIS\n"
        text += "-----------------\n"
        text += f"Documents Processed: {len(report.document_analyses)}\n"

        # Add document summary
        for doc in report.document_analyses[:5]:
            text += f"\n- {doc.document_type.value.upper()}: {doc.summary[:100]}"

        text += f"\n\nKEY FINDINGS\n"
        text += "------------\n"
        for finding in report.key_findings:
            text += f"\n- {finding}"

        text += f"\n\nCRITICAL ISSUES\n"
        text += "---------------\n"
        for issue in report.critical_issues:
            text += f"\n- [{issue.get('severity', 'unknown').upper()}] {issue.get('description', '')}"

        text += f"\n\nRECOMMENDATIONS\n"
        text += "---------------\n"
        for rec in report.recommendations:
            text += f"\n- {rec}"

        return text


if __name__ == "__main__":
    # Example usage
    orchestrator = AIPipelineOrchestrator()

    # Sample case data
    evidence = [
        ("ev001", "Witness testimony regarding custody violations"),
        ("ev002", "Documentation of missed visitation dates"),
    ]

    documents = [
        ("doc001", "MOTION FOR MODIFICATION OF CUSTODY\n\nPlaintiff seeks modification..."),
        ("doc002", "AFFIDAVIT IN SUPPORT\n\nDeponent states under oath..."),
    ]

    # Process case
    report = orchestrator.process_case(
        case_id="case_001",
        case_title="Custody Modification Case",
        evidence_items=evidence,
        documents=documents,
        case_type="custody"
    )

    # Export report
    json_output = orchestrator.export_report(report, "json")
    print(json_output[:500])  # Print first 500 characters
