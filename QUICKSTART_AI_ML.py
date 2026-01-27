#!/usr/bin/env python3
"""
FRED Supreme Litigation OS - AI/ML Integration Quick Start Guide

This script demonstrates how to use the newly integrated AI/ML components
for comprehensive litigation case analysis.
"""

import asyncio
import sys
from pathlib import Path
from ai.ai_pipeline_orchestrator import AIPipelineOrchestrator
from ai.evidence_llm_analyzer import EvidenceLLMAnalyzer
from ai.nlp_document_processor import NLPDocumentProcessor
from ai.argument_reasoning import (
    ArgumentReasoningGraph,
    ArgumentType,
    RelationType
)
from src.ai_integration_bridge import AIIntegrationBridge, AIIntegrationConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def example_1_basic_evidence_analysis() -> None:
    """Example 1: Analyze evidence using LLM"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Evidence Analysis")
    print("="*70)

    analyzer = EvidenceLLMAnalyzer()

    evidence_content = """
    Witness statement from January 15, 2024:

    I, Michael Johnson, was present on January 15, 2024 at 6:00 PM when
    the defendant was supposed to pick up the child for scheduled visitation.

    The defendant arrived 35 minutes late at 6:35 PM. When I asked about
    the delay, the defendant stated they were running behind and did not
    provide additional explanation.

    This tardiness caused the child significant distress and disrupted the
    scheduled dinner plans with the defendant's family.

    This pattern of late pickups has occurred on multiple occasions.
    """

    print("\nAnalyzing evidence...")
    result = analyzer.analyze_evidence(
        evidence_id="ev_witness_001",
        content=evidence_content,
        case_type="custody"
    )

    print(f"Evidence Type: {result.evidence_type.value}")
    print(f"Confidence: {result.confidence:.0%}")
    print("Scores:")
    if result.scores:
        print(f"  Relevance: {result.scores.relevance_score:.0%}")
        print(f"  Reliability: {result.scores.reliability_score:.0%}")
        print(f"  Impact: {result.scores.impact_score:.0%}")
        print(f"  Overall Strength: {result.scores.overall_strength:.0%}")
        print(f"  Credibility: {result.scores.credibility_level.value}")
    print(f"Semantic Summary: {result.semantic_summary}")
    print(f"Key Phrases: {', '.join(result.key_phrases)}")
    print(f"Legal Implications: {', '.join(result.legal_implications)}")


def example_2_document_processing():
    """Example 2: Process legal document using NLP"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Document Processing with NLP")
    print("="*70)

    processor = NLPDocumentProcessor()

    document_content = """
    MOTION FOR ENFORCEMENT OF CUSTODY ORDER

    TO THE HONORABLE COURT:

    Plaintiff, Jane Smith, by and through her undersigned counsel, respectfully
    submits this Motion for Enforcement of the Custody Order dated January 1, 2024,
    as follows:

    FACTUAL BACKGROUND

    1. This Honorable Court entered a Custody Order on January 1, 2024,
       establishing a custody and visitation schedule whereby Defendant was
       entitled to visitation on Friday evenings beginning at 6:00 PM.

    2. The Defendant has repeatedly failed to comply with this Court Order,
       arriving late on multiple occasions and disrupting the schedule.

    3. These violations have caused emotional harm to the minor child and
       undermined the Court's authority.

    WHEREFORE

    Plaintiff respectfully requests that this Court:

    1. Enforce the Custody Order dated January 1, 2024
    2. Impose sanctions on the Defendant for non-compliance
    3. Modify the visitation schedule if necessary to ensure compliance
    4. Award Plaintiff her reasonable attorney fees and costs

    Respectfully submitted,
    Jane Smith's Attorney
    Dated: March 1, 2024
    """

    print("\nProcessing document...")
    metadata = processor.process_document(
        content=document_content,
        document_title="Motion_for_Enforcement_001"
    )

    print(f"\nDocument Type: {metadata.document_type.value}")
    print(f"Sentiment: {metadata.sentiment.value}")
    print(f"Confidence: {metadata.confidence_score:.0%}")
    print(f"\nParties Involved: {', '.join(metadata.parties_involved)}")
    print(f"Key Concepts: {', '.join(metadata.key_concepts)}")
    print(f"Action Items: {metadata.action_items}")
    print(f"Deadlines: {metadata.deadlines}")
    print(f"\nSummary: {metadata.summary}")


def example_3_argument_reasoning():
    """Example 3: Build argument reasoning graph"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Argument Reasoning Graph Analysis")
    print("="*70)

    arg_system = ArgumentReasoningGraph()

    # Create main claim
    main_claim = arg_system.create_node(
        text="Defendant violated the custody order",
        arg_type=ArgumentType.CLAIM,
        source="case_facts",
        confidence=0.90
    )
    print(f"\nMain Claim: {main_claim.text}")

    # Create supporting evidence
    evidence1 = arg_system.create_node(
        text="Witness testimony showing late arrival",
        arg_type=ArgumentType.EVIDENCE,
        source="witness_statement",
        confidence=0.85
    )

    evidence2 = arg_system.create_node(
        text="Text message timestamps confirming late pickups",
        arg_type=ArgumentType.EVIDENCE,
        source="communications",
        confidence=0.95
    )

    reasoning = arg_system.create_node(
        text="Pattern of non-compliance demonstrates intentional disregard for court order",
        arg_type=ArgumentType.REASONING,
        source="analysis",
        confidence=0.80
    )

    # Create relationships
    arg_system.create_edge(
        source_id=evidence1.node_id,
        target_id=main_claim.node_id,
        relation_type=RelationType.SUPPORTS,
        strength=0.85,
        reasoning="Witness testimony directly establishes violation"
    )

    arg_system.create_edge(
        source_id=evidence2.node_id,
        target_id=reasoning.node_id,
        relation_type=RelationType.SUPPORTS,
        strength=0.95,
        reasoning="Documentary evidence confirms pattern"
    )

    arg_system.create_edge(
        source_id=reasoning.node_id,
        target_id=main_claim.node_id,
        relation_type=RelationType.SUPPORTS,
        strength=0.80,
        reasoning="Pattern analysis strengthens main claim"
    )

    # Analyze
    analysis = arg_system.analyze_case(
        case_id="custody_enforcement_001",
        case_title="Smith v. Doe - Custody Order Enforcement",
        evidence_nodes=[evidence1, evidence2],
        argument_nodes=[main_claim, reasoning],
        edges=list(arg_system.edges.values())
    )

    print(f"Overall Argument Strength: {analysis.overall_strength.value}")
    print(f"Overall Score: {analysis.overall_score:.0%}")
    print(f"Supporting Arguments: {len(analysis.supporting_arguments)}")
    print(f"Vulnerabilities: {len(analysis.vulnerabilities)}")
    print(f"Recommendations: {len(analysis.recommendations)}")
    print("Key Recommendations:")
    for i, rec in enumerate(analysis.recommendations[:3], 1):
        print(f"  {i}. {rec}")


async def example_4_full_case_analysis():
    """Example 4: Complete end-to-end case analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Full AI Pipeline Case Analysis")
    print("="*70)

    orchestrator = AIPipelineOrchestrator()

    # Real custody case data
    evidence_items = [
        ("witness_001", """
            I observed the defendant arriving 35 minutes late for scheduled
            visitation on January 15, 2024, causing distress to the child.
        """),
        ("text_logs_001", """
            Multiple text messages showing scheduled pickup times at 6:00 PM
            with actual arrivals at 6:35 PM, 6:28 PM, and 6:45 PM on
            different dates.
        """),
        ("affidavit_001", """
            Under oath, the custodial parent states the defendant has
            consistently violated the visitation schedule, arriving late
            on 12 separate occasions over the past 3 months.
        """),
    ]

    documents = [
        ("motion_001", """
            MOTION FOR ENFORCEMENT OF CUSTODY ORDER

            Plaintiff seeks enforcement of the custody order dated January 1, 2024,
            due to repeated violations by the defendant regarding scheduled
            visitation times.
        """),
        ("affidavit_001", """
            AFFIDAVIT IN SUPPORT OF MOTION

            State of Michigan
            County of Wayne

            I, Jane Smith, being duly sworn, state that:
            1. The defendant has violated the custody order on 12 occasions
            2. Each violation involved late arrival for scheduled visitation
            3. These violations have caused emotional harm to the child
        """),
    ]

    print("\nRunning comprehensive case analysis...")
    report = orchestrator.process_case(
        case_id="smith_v_doe_2024",
        case_title="Smith v. Doe - Custody Order Enforcement",
        evidence_items=evidence_items,
        documents=documents,
        case_type="custody"
    )

    print("✓ Analysis Complete")
    print(f"  Overall Confidence: {report.confidence_score:.0%}")
    print(f"  Processing Time: {report.processing_time_seconds:.2f} seconds")
    print(f"  Pipeline Status: {report.pipeline_status.value}")

    print(f"Evidence Analyzed: {len(report.evidence_analyses)}")
    print(f"Documents Processed: {len(report.document_analyses)}")

    print(f"Key Findings ({len(report.key_findings)}):")
    for i, finding in enumerate(report.key_findings[:5], 1):
        print(f"  {i}. {finding}")

    print(f"\nCritical Issues ({len(report.critical_issues)}):")
    for issue in report.critical_issues[:3]:
        print(f"  • [{issue.get('severity', 'UNKNOWN')}] {issue.get('description', '')}")

    print(f"\nRecommendations ({len(report.recommendations)}):")
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"  {i}. {rec}")

    return report


async def example_5_ai_integration_bridge():
    """Example 5: Using AI integration bridge"""
    print("\n" + "="*70)
    print("EXAMPLE 5: AI Integration Bridge (Master Workflow Integration)")
    print("="*70)

    config = AIIntegrationConfig(
        case_type="custody",
        enable_ai_analysis=True
    )
    bridge = AIIntegrationBridge(config)

    evidence = [
        ("ev001", "Witness testimony about custody violation"),
        ("ev002", "Documentation showing missed visitation"),
    ]

    documents = [
        ("doc001", "MOTION FOR MODIFICATION\n\nPlaintiff seeks modification..."),
    ]

    print("\nRunning analysis through AI integration bridge...")
    report = await bridge.full_case_analysis(
        case_id="case_001",
        case_title="Custody Modification",
        evidence_items=evidence,
        documents=documents
    )

    if report:
        summary = bridge.get_analysis_summary(report)
        print("Analysis Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("Briefing Memo Preview:")
        memo = bridge.generate_briefing_memo(report)
        print(memo[:500] + "...")


def main():
    """Run all examples"""
    print("\n" + "█"*70)
    print("█ FRED SUPREME LITIGATION OS - AI/ML INTEGRATION EXAMPLES")
    print("█"*70)

    # Synchronous examples
    print("\n[Running synchronous examples...]")
    example_1_basic_evidence_analysis()
    example_2_document_processing()
    example_3_argument_reasoning()

    # Asynchronous examples
    print("\n[Running asynchronous examples...]")
    asyncio.run(example_4_full_case_analysis())
    asyncio.run(example_5_ai_integration_bridge())

    print("\n" + "█"*70)
    print("█ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("█"*70)
    print("\nFor more information, see docs/AI_ML_INTEGRATION.md")


if __name__ == "__main__":
    main()
