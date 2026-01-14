"""
AI Integration Wrapper for Master Integration Bridge
Connects AI/ML components to the master workflow system
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
import json

from ai.ai_pipeline_orchestrator import AIPipelineOrchestrator, AIAnalysisReport
from ai.evidence_llm_analyzer import EvidenceLLMAnalyzer
from ai.nlp_document_processor import NLPDocumentProcessor
from ai.argument_reasoning import ArgumentReasoningGraph

logger = logging.getLogger(__name__)


@dataclass
class AIIntegrationConfig:
    """Configuration for AI integration"""
    enable_ai_analysis: bool = True
    enable_llm_evidence: bool = True
    enable_nlp_documents: bool = True
    enable_arg_reasoning: bool = True
    max_workers: int = 4
    case_type: str = "general"
    export_formats: List[str] = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["json", "text"]


class AIIntegrationBridge:
    """
    Bridge between master integration and AI/ML components.
    Provides AI analysis capabilities integrated into the workflow.
    """

    def __init__(self, config: Optional[AIIntegrationConfig] = None):
        """Initialize AI integration bridge"""
        self.config = config or AIIntegrationConfig()
        self.orchestrator = AIPipelineOrchestrator(
            max_workers=self.config.max_workers
        ) if self.config.enable_ai_analysis else None
        self.evidence_analyzer = EvidenceLLMAnalyzer() if self.config.enable_llm_evidence else None
        self.document_processor = NLPDocumentProcessor() if self.config.enable_nlp_documents else None
        self.arg_system = ArgumentReasoningGraph() if self.config.enable_arg_reasoning else None

        logger.info("AI Integration Bridge initialized")
        logger.info(f"  LLM Analysis: {self.config.enable_llm_evidence}")
        logger.info(f"  NLP Processing: {self.config.enable_nlp_documents}")
        logger.info(f"  ARG Reasoning: {self.config.enable_arg_reasoning}")

    async def analyze_evidence_with_ai(
        self,
        evidence_items: List[Tuple[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze evidence using AI.

        Args:
            evidence_items: List of (evidence_id, content) tuples
            context: Additional context

        Returns:
            Analysis results
        """
        if not self.evidence_analyzer:
            logger.warning("LLM Evidence analyzer not enabled")
            return {"status": "disabled"}

        try:
            logger.info(f"Analyzing {len(evidence_items)} evidence items with AI")
            results = self.evidence_analyzer.batch_analyze_evidence(
                evidence_items,
                case_type=self.config.case_type
            )

            return {
                "status": "completed",
                "items_analyzed": len(results),
                "analyses": [r.to_dict() for r in results],
                "high_confidence_items": len([r for r in results if r.confidence > 0.8])
            }

        except Exception as e:
            logger.error(f"Evidence analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def analyze_documents_with_nlp(
        self,
        documents: List[Tuple[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process documents using NLP.

        Args:
            documents: List of (doc_id, content) tuples
            context: Additional context

        Returns:
            Processing results
        """
        if not self.document_processor:
            logger.warning("NLP Document processor not enabled")
            return {"status": "disabled"}

        try:
            logger.info(f"Processing {len(documents)} documents with NLP")
            results = self.document_processor.batch_process_documents(
                documents,
                case_context=context
            )

            summary = self.document_processor.generate_summary_report(results)

            return {
                "status": "completed",
                "items_processed": len(results),
                "summary": summary,
                "documents": [d.to_dict() for d in results]
            }

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def full_case_analysis(
        self,
        case_id: str,
        case_title: str,
        evidence_items: List[Tuple[str, str]],
        documents: List[Tuple[str, str]],
        case_context: Optional[Dict[str, Any]] = None
    ) -> AIAnalysisReport:
        """
        Perform complete case analysis through entire AI pipeline.

        Args:
            case_id: Case identifier
            case_title: Case title
            evidence_items: Evidence list
            documents: Document list
            case_context: Additional context

        Returns:
            Complete analysis report
        """
        if not self.orchestrator:
            logger.warning("AI orchestrator not enabled")
            return None

        try:
            logger.info(f"Starting full AI case analysis for {case_id}")
            report = self.orchestrator.process_case(
                case_id=case_id,
                case_title=case_title,
                evidence_items=evidence_items,
                documents=documents,
                case_type=self.config.case_type,
                case_context=case_context
            )

            logger.info(f"Case analysis completed with confidence: {report.confidence_score:.0%}")
            return report

        except Exception as e:
            logger.error(f"Full case analysis failed: {e}")
            return None

    def export_report(
        self,
        report: AIAnalysisReport,
        output_dir: Path,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export analysis report in specified formats.

        Args:
            report: Analysis report
            output_dir: Output directory
            formats: List of formats (json, text)

        Returns:
            Dictionary of format -> file_path
        """
        if not self.orchestrator or not report:
            return {}

        formats = formats or self.config.export_formats
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}

        for fmt in formats:
            try:
                content = self.orchestrator.export_report(report, fmt)
                filename = f"analysis_report_{report.case_id}.{fmt if fmt != 'text' else 'txt'}"
                filepath = output_dir / filename

                with open(filepath, 'w') as f:
                    f.write(content)

                outputs[fmt] = str(filepath)
                logger.info(f"Exported {fmt} report to {filepath}")

            except Exception as e:
                logger.error(f"Failed to export {fmt} report: {e}")

        return outputs

    async def continuous_case_monitoring(
        self,
        case_id: str,
        evidence_stream: List[Tuple[str, str]],
        document_stream: List[Tuple[str, str]],
        analysis_interval: float = 3600  # 1 hour
    ):
        """
        Continuously monitor and analyze case as new evidence arrives.

        Args:
            case_id: Case identifier
            evidence_stream: Stream of evidence items
            document_stream: Stream of documents
            analysis_interval: Seconds between analyses
        """
        logger.info(f"Starting continuous monitoring for case {case_id}")

        while True:
            try:
                # Run periodic analysis
                report = await self.full_case_analysis(
                    case_id=case_id,
                    case_title=f"{case_id} - Continuous Analysis",
                    evidence_items=evidence_stream,
                    documents=document_stream
                )

                if report:
                    logger.info(
                        f"Periodic analysis complete: "
                        f"Confidence {report.confidence_score:.0%}, "
                        f"Issues: {len(report.critical_issues)}"
                    )

                # Wait for next interval
                await asyncio.sleep(analysis_interval)

            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

    def get_analysis_summary(self, report: AIAnalysisReport) -> Dict[str, Any]:
        """
        Get concise summary of analysis results.

        Args:
            report: Analysis report

        Returns:
            Summary dictionary
        """
        if not report:
            return {}

        return {
            "case_id": report.case_id,
            "case_title": report.case_title,
            "status": report.pipeline_status.value,
            "overall_confidence": f"{report.confidence_score:.0%}",
            "processing_time": f"{report.processing_time_seconds:.2f}s",
            "evidence_analyzed": len(report.evidence_analyses),
            "documents_analyzed": len(report.document_analyses),
            "key_findings": len(report.key_findings),
            "critical_issues": len(report.critical_issues),
            "recommendations": len(report.recommendations),
            "argument_strength": (
                report.argument_analysis.overall_strength.value
                if report.argument_analysis else "N/A"
            )
        }

    def generate_briefing_memo(self, report: AIAnalysisReport) -> str:
        """
        Generate executive briefing memo from analysis.

        Args:
            report: Analysis report

        Returns:
            Formatted briefing memo
        """
        memo = f"""
        CASE ANALYSIS BRIEFING MEMO
        ==========================

        Case: {report.case_title}
        Case ID: {report.case_id}
        Analysis Date: {report.analysis_timestamp}

        OVERALL ASSESSMENT
        ------------------
        Confidence Level: {report.confidence_score:.0%}
        Pipeline Status: {report.pipeline_status.value}

        KEY FINDINGS
        -----------
        """

        for i, finding in enumerate(report.key_findings[:5], 1):
            memo += f"\n{i}. {finding}"

        memo += "\n\nCRITICAL ISSUES\n"
        memo += "---------------\n"

        for issue in report.critical_issues[:5]:
            severity = issue.get('severity', 'UNKNOWN').upper()
            description = issue.get('description', '')
            memo += f"\n[{severity}] {description}"

        memo += "\n\nRECOMMENDED ACTIONS\n"
        memo += "------------------\n"

        for i, rec in enumerate(report.recommendations[:5], 1):
            memo += f"\n{i}. {rec}"

        if report.argument_analysis:
            memo += f"\n\nARGUMENT STRENGTH ASSESSMENT\n"
            memo += f"---------------------------\n"
            memo += f"Overall Strength: {report.argument_analysis.overall_strength.value}\n"
            memo += f"Supporting Arguments: {len(report.argument_analysis.supporting_arguments)}\n"
            memo += f"Counter-Arguments Identified: {len(report.argument_analysis.counter_arguments)}\n"
            memo += f"Vulnerabilities: {len(report.argument_analysis.vulnerabilities)}\n"

        memo += f"\n\nPROCESSING DETAILS\n"
        memo += f"------------------\n"
        memo += f"Processing Time: {report.processing_time_seconds:.2f} seconds\n"
        memo += f"Evidence Items: {len(report.evidence_analyses)}\n"
        memo += f"Documents: {len(report.document_analyses)}\n"

        return memo


# Integration function for master_integration_bridge.py
async def integrate_ai_analysis(
    context: Any,  # CaseContext from master_integration_bridge
    config: Dict[str, Any],
    ai_config: Optional[AIIntegrationConfig] = None
) -> Dict[str, Any]:
    """
    Integrate AI analysis into master workflow stage.

    Args:
        context: Case context from master integration
        config: Stage configuration
        ai_config: AI-specific configuration

    Returns:
        AI analysis results
    """
    ai_config = ai_config or AIIntegrationConfig(
        case_type=config.get('case_type', 'general')
    )
    bridge = AIIntegrationBridge(ai_config)

    # Extract evidence and documents from context
    evidence_items = [
        (f['name'], f.get('content', ''))
        for f in getattr(context, 'evidence_files', [])
    ]

    documents = [
        (d['name'], d.get('content', ''))
        for d in getattr(context, 'documents', [])
    ]

    # Run full analysis
    report = await bridge.full_case_analysis(
        case_id=context.case_id,
        case_title=context.case_title,
        evidence_items=evidence_items,
        documents=documents,
        case_context={"case_type": config.get('case_type', 'general')}
    )

    if report:
        return {
            "status": "completed",
            "ai_analysis": report.to_dict(),
            "summary": bridge.get_analysis_summary(report),
            "briefing_memo": bridge.generate_briefing_memo(report)
        }
    else:
        return {
            "status": "failed",
            "error": "AI analysis failed"
        }


if __name__ == "__main__":
    # Example usage
    config = AIIntegrationConfig(
        case_type="custody",
        enable_ai_analysis=True
    )
    bridge = AIIntegrationBridge(config)

    # Sample data
    evidence = [
        ("ev001", "Witness testimony"),
        ("ev002", "Documentary evidence"),
    ]

    documents = [
        ("doc001", "MOTION FOR MODIFICATION..."),
        ("doc002", "AFFIDAVIT..."),
    ]

    # Run async analysis
    async def main():
        report = await bridge.full_case_analysis(
            case_id="case_001",
            case_title="Test Case",
            evidence_items=evidence,
            documents=documents
        )

        if report:
            print(bridge.get_analysis_summary(report))
            print("\n" + bridge.generate_briefing_memo(report))

    asyncio.run(main())
