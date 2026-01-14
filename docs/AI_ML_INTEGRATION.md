# AI/ML Integration Guide

## Overview

The FRED Supreme Litigation OS now includes advanced AI/ML capabilities for comprehensive legal case analysis. This document provides a complete guide to the AI/ML integration, including the LLM (Large Language Model), NLP (Natural Language Processing), and ARG (Argument Reasoning Graph) systems.

## Table of Contents

1. [Components](#components)
2. [Architecture](#architecture)
3. [Usage Guide](#usage-guide)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Performance Considerations](#performance-considerations)
7. [Troubleshooting](#troubleshooting)

## Components

### 1. Evidence LLM Analyzer (`ai/evidence_llm_analyzer.py`)

Uses Hugging Face transformers to provide AI-powered evidence analysis.

**Key Features:**

- Evidence type classification (documentary, testimonial, physical, etc.)
- Semantic summary extraction
- Key phrase identification
- Entity extraction (persons, organizations, locations, dates)
- Evidence scoring based on:
  - Relevance to case type
  - Reliability assessment
  - Impact potential
  - Chain of custody integrity
- Credibility assessment
- Evidence comparison
- Batch processing

**Main Classes:**

- `EvidenceLLMAnalyzer`: Main analyzer class
- `AnalyzedEvidence`: Result data structure
- `EvidenceScore`: Comprehensive scoring system

**Example:**

```python
from ai.evidence_llm_analyzer import EvidenceLLMAnalyzer

analyzer = EvidenceLLMAnalyzer()
evidence = analyzer.analyze_evidence(
    evidence_id="ev001",
    content="Witness testimony...",
    case_type="custody"
)

print(f"Overall Strength: {evidence.scores.overall_strength:.0%}")
print(f"Credibility: {evidence.scores.credibility_level.value}")
```

### 2. NLP Document Processor (`ai/nlp_document_processor.py`)

Processes legal documents using NLP techniques.

**Key Features:**

- Document type classification (motion, affidavit, complaint, etc.)
- Entity extraction
- Party identification
- Sentiment analysis
- Key concept extraction
- Action item identification
- Deadline extraction
- Relationship extraction
- Batch processing
- Summary report generation

**Main Classes:**

- `NLPDocumentProcessor`: Main processor
- `DocumentMetadata`: Result structure with extracted information

**Example:**

```python
from ai.nlp_document_processor import NLPDocumentProcessor

processor = NLPDocumentProcessor()
metadata = processor.process_document(
    content="MOTION FOR MODIFICATION...",
    document_title="Motion1"
)

print(f"Document Type: {metadata.document_type.value}")
print(f"Parties: {metadata.parties_involved}")
print(f"Key Concepts: {metadata.key_concepts}")
```

### 3. Argument Reasoning Graph (`ai/argument_reasoning.py`)

Creates structured representations of legal arguments.

**Key Features:**

- Argument node creation (claims, evidence, reasoning)
- Relationship definition (supports, contradicts, implies, etc.)
- Path finding through argument graphs
- Argument strength assessment
- Vulnerability identification
- Counter-argument analysis
- Strategic recommendations
- BFS-based path analysis

**Main Classes:**

- `ArgumentReasoningGraph`: Main ARG system
- `ArgumentNode`: Individual argument elements
- `ArgumentEdge`: Relationships between arguments
- `ArgumentAnalysis`: Complete analysis result

**Example:**

```python
from ai.argument_reasoning import ArgumentReasoningGraph, ArgumentType, RelationType

arg_system = ArgumentReasoningGraph()

claim = arg_system.create_node(
    text="Defendant violated custody order",
    arg_type=ArgumentType.CLAIM,
    source="facts",
    confidence=0.85
)

evidence = arg_system.create_node(
    text="Witness testimony",
    arg_type=ArgumentType.EVIDENCE,
    source="witness",
    confidence=0.9
)

arg_system.create_edge(
    source_id=evidence.node_id,
    target_id=claim.node_id,
    relation_type=RelationType.SUPPORTS,
    strength=0.85
)
```

### 4. AI Pipeline Orchestrator (`ai/ai_pipeline_orchestrator.py`)

Orchestrates all AI components for complete case analysis.

**Key Features:**

- End-to-end case processing pipeline
- Evidence analysis stage
- Document processing stage
- Argument construction stage
- Logical reasoning stage
- Analysis validation
- Comprehensive reporting
- Multiple export formats (JSON, text)
- Concurrent processing support

**Main Classes:**

- `AIPipelineOrchestrator`: Main orchestrator
- `AIAnalysisReport`: Complete analysis report

**Example:**

```python
from ai.ai_pipeline_orchestrator import AIPipelineOrchestrator

orchestrator = AIPipelineOrchestrator()

report = orchestrator.process_case(
    case_id="case_001",
    case_title="Custody Modification",
    evidence_items=[
        ("ev001", "Witness testimony..."),
        ("ev002", "Documentation..."),
    ],
    documents=[
        ("doc001", "MOTION FOR MODIFICATION..."),
        ("doc002", "AFFIDAVIT..."),
    ],
    case_type="custody"
)

# Export results
json_report = orchestrator.export_report(report, "json")
text_report = orchestrator.export_report(report, "text")
```

### 5. GitHub Integration (`integrations/github_integration.py`)

Provides GitHub API connectivity for repository and CI/CD management.

**Key Features:**

- Repository information retrieval
- Issue management
- Pull request handling
- Workflow automation
- Branch management
- Label management
- Data export (JSON, CSV)

**Main Classes:**

- `GitHubAPIClient`: GitHub API client

**Example:**

```python
from integrations.github_integration import GitHubAPIClient

client = GitHubAPIClient(
    token="your_github_token",
    owner="yourname",
    repo="litigation-system"
)

# Get repository info
repo = client.get_repository()
print(f"Stars: {repo.stars}")

# List issues
issues = client.list_issues(state="open", limit=10)
for issue in issues:
    print(f"#{issue.number}: {issue.title}")

# Create issue
new_issue = client.create_issue(
    title="Case Analysis Needed",
    body="Please analyze the new evidence",
    labels=["analysis", "urgent"]
)
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│         AI Pipeline Orchestrator                 │
│  (Main coordination and result compilation)      │
└─────────────────────────────────────────────────┘
    ↓          ↓          ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Evidence│ │  NLP   │ │  ARG   │ │ Stage  │ │ Export │
│ LLM    │ │Proc.   │ │ System │ │Manager │ │Manager │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
    ↓          ↓          ↓
┌────────┬────────┬────────┐
│ Hugging│ Regex  │ Graph  │
│  Face  │Patterns│ Theory │
└────────┴────────┴────────┘
```

## Usage Guide

### Basic Case Analysis

```python
from ai.ai_pipeline_orchestrator import AIPipelineOrchestrator

# Initialize orchestrator
orchestrator = AIPipelineOrchestrator()

# Prepare case data
evidence_items = [
    ("ev001", "witness_testimony_content"),
    ("ev002", "documentary_evidence_content"),
]

documents = [
    ("doc001", "motion_content"),
    ("doc002", "affidavit_content"),
]

# Process case
report = orchestrator.process_case(
    case_id="case_001",
    case_title="Case Title",
    evidence_items=evidence_items,
    documents=documents,
    case_type="custody"
)

# Access results
print(f"Status: {report.pipeline_status}")
print(f"Confidence: {report.confidence_score:.0%}")
print(f"Findings: {report.key_findings}")
print(f"Issues: {report.critical_issues}")
print(f"Recommendations: {report.recommendations}")
```

### Evidence-Only Analysis

```python
from ai.evidence_llm_analyzer import EvidenceLLMAnalyzer

analyzer = EvidenceLLMAnalyzer()

# Analyze single evidence
result = analyzer.analyze_evidence(
    evidence_id="ev001",
    content="Evidence content",
    case_type="ppo"
)

# Batch analyze
evidence_list = [
    ("ev001", "content1"),
    ("ev002", "content2"),
]
results = analyzer.batch_analyze_evidence(evidence_list)

# Compare evidence
comparison = analyzer.compare_evidence(result1, result2)

# Export
json_report = analyzer.export_analysis_report(results, "json")
csv_report = analyzer.export_analysis_report(results, "csv")
```

### Document Analysis

```python
from ai.nlp_document_processor import NLPDocumentProcessor

processor = NLPDocumentProcessor()

# Process single document
metadata = processor.process_document(
    content="Document content",
    document_title="Doc1"
)

# Batch process
documents = [
    ("doc1", "content1"),
    ("doc2", "content2"),
]
results = processor.batch_process_documents(documents)

# Generate summary
report = processor.generate_summary_report(results)
```

### Argument Analysis

```python
from ai.argument_reasoning import ArgumentReasoningGraph, ArgumentType, RelationType

arg_system = ArgumentReasoningGraph()

# Create nodes
claim = arg_system.create_node(
    text="Main claim",
    arg_type=ArgumentType.CLAIM,
    source="source1",
    confidence=0.85
)

evidence1 = arg_system.create_node(
    text="Supporting evidence",
    arg_type=ArgumentType.EVIDENCE,
    source="source2",
    confidence=0.9
)

# Create relationships
arg_system.create_edge(
    source_id=evidence1.node_id,
    target_id=claim.node_id,
    relation_type=RelationType.SUPPORTS,
    strength=0.85
)

# Analyze
analysis = arg_system.analyze_case(
    case_id="case_001",
    case_title="Test Case",
    evidence_nodes=[evidence1],
    argument_nodes=[claim],
    edges=list(arg_system.edges.values())
)

# Export
text_report = arg_system.export_analysis(analysis, "text")
json_report = arg_system.export_analysis(analysis, "json")
```

## API Reference

### Evidence LLM Analyzer

#### `analyze_evidence(evidence_id, content, case_type, context)`

Performs comprehensive analysis of a piece of evidence.

**Parameters:**

- `evidence_id` (str): Unique identifier
- `content` (str): Evidence content
- `case_type` (str): Type of case (custody, ppo, etc.)
- `context` (dict, optional): Additional context

**Returns:** `AnalyzedEvidence`

#### `batch_analyze_evidence(evidence_list, case_type)`

Analyzes multiple evidence items efficiently.

**Parameters:**

- `evidence_list` (list): List of (id, content) tuples
- `case_type` (str): Type of case

**Returns:** `List[AnalyzedEvidence]`

#### `compare_evidence(evidence1, evidence2)`

Compares two pieces of evidence.

**Parameters:**

- `evidence1` (AnalyzedEvidence)
- `evidence2` (AnalyzedEvidence)

**Returns:** `Dict` with comparison results

### NLP Document Processor

#### `process_document(content, document_title, case_context)`

Processes a legal document.

**Parameters:**

- `content` (str): Document content
- `document_title` (str): Document title
- `case_context` (dict, optional): Additional context

**Returns:** `DocumentMetadata`

#### `batch_process_documents(documents, case_context)`

Processes multiple documents.

**Parameters:**

- `documents` (list): List of (id, content) tuples
- `case_context` (dict, optional): Additional context

**Returns:** `List[DocumentMetadata]`

### Argument Reasoning Graph

#### `create_node(text, arg_type, source, confidence, metadata)`

Creates an argument node.

**Returns:** `ArgumentNode`

#### `create_edge(source_id, target_id, relation_type, strength, reasoning)`

Creates a relationship between arguments.

**Returns:** `ArgumentEdge`

#### `analyze_case(case_id, case_title, evidence_nodes, argument_nodes, edges)`

Performs comprehensive case analysis.

**Returns:** `ArgumentAnalysis`

### AI Pipeline Orchestrator

#### `process_case(case_id, case_title, evidence_items, documents, case_type, case_context)`

Processes complete case through entire pipeline.

**Returns:** `AIAnalysisReport`

#### `export_report(report, format, output_path)`

Exports analysis report.

**Parameters:**

- `report` (AIAnalysisReport)
- `format` (str): "json" or "text"
- `output_path` (str, optional): File path to save

**Returns:** `str` (formatted report)

## Examples

### Example 1: Complete Custody Case Analysis

```python
from ai.ai_pipeline_orchestrator import AIPipelineOrchestrator

orchestrator = AIPipelineOrchestrator()

# Real custody case data
evidence = [
    ("ev_witness_001", """
        Witness statement from Jan 15, 2024:
        I observed the defendant consistently returning the child
        30-60 minutes late from scheduled visitation. This occurred
        on multiple occasions throughout 2024.
    """),
    ("ev_docs_001", """
        Text message timestamps showing:
        - Jan 5: Pickup at 6:30 PM instead of 6:00 PM
        - Jan 12: Return at 8:15 PM instead of 8:00 PM
        - Jan 19: Failure to return by agreed time
    """),
]

documents = [
    ("motion", """
        MOTION FOR ENFORCEMENT OF CUSTODY ORDER

        Plaintiff brings this Motion to Enforce the Custody Order
        dated January 1, 2024, as defendant has repeatedly violated
        the visitation schedule.

        The violations have caused emotional distress to the child
        and disrupted the child's educational schedule.
    """),
    ("affidavit", """
        AFFIDAVIT IN SUPPORT OF MOTION

        State of Michigan
        County of Wayne

        I, Jane Smith, being duly sworn, state:

        1. I am the plaintiff in this action
        2. The defendant has missed pickup times on 12+ occasions
        3. The defendant refuses to acknowledge the violations
        4. The child is experiencing anxiety due to schedule uncertainty
    """),
]

# Process case
report = orchestrator.process_case(
    case_id="smith_v_doe_2024",
    case_title="Smith v. Doe - Custody Enforcement",
    evidence_items=evidence,
    documents=documents,
    case_type="custody",
    case_context={"jurisdiction": "Wayne County", "year": 2024}
)

# Review findings
print("KEY FINDINGS:")
for finding in report.key_findings:
    print(f"  • {finding}")

print("\nCRITICAL ISSUES:")
for issue in report.critical_issues:
    print(f"  • [{issue['severity']}] {issue['description']}")

print("\nRECOMMENDATIONS:")
for rec in report.recommendations:
    print(f"  • {rec}")

# Export final report
json_report = orchestrator.export_report(
    report,
    "json",
    output_path="/output/case_analysis.json"
)
```

### Example 2: Evidence Strength Comparison

```python
from ai.evidence_llm_analyzer import EvidenceLLMAnalyzer

analyzer = EvidenceLLMAnalyzer()

# Analyze two pieces of evidence
strong_evidence = analyzer.analyze_evidence(
    "ev_001",
    "Official court order dated January 1, 2024, signed by Judge Smith"
)

weak_evidence = analyzer.analyze_evidence(
    "ev_002",
    "Hearsay claim that someone said something happened"
)

# Compare
comparison = analyzer.compare_evidence(strong_evidence, weak_evidence)

print(f"Strength Difference: {comparison['strength_difference']:.0%}")
print(f"Recommendation: {comparison['recommendation']}")
```

### Example 3: Document Processing with NLP

```python
from ai.nlp_document_processor import NLPDocumentProcessor

processor = NLPDocumentProcessor()

# Process motion
motion_content = """
    MOTION FOR MODIFICATION OF CUSTODY

    TO THE HONORABLE COURT:

    Plaintiff, by and through counsel, respectfully submits this Motion
    for Modification of Custody based on material change in circumstances.

    FACTS:
    1. Custody order dated January 15, 2024
    2. Defendant has violated order multiple times
    3. Child's best interest served by modification

    WHEREFORE: Plaintiff requests the Court modify custody.

    Respectfully submitted,
    Attorney for Plaintiff
    Dated: March 1, 2024
"""

metadata = processor.process_document(motion_content, "Motion_1")

print(f"Document Type: {metadata.document_type.value}")
print(f"Parties: {', '.join(metadata.parties_involved)}")
print(f"Summary: {metadata.summary}")
print(f"Key Concepts: {', '.join(metadata.key_concepts)}")
print(f"Action Items: {len(metadata.action_items)} identified")
print(f"Deadlines: {metadata.deadlines}")
```

## Performance Considerations

1. **Parallel Processing**: Use `max_workers` parameter in orchestrator for optimal performance
2. **Batch Processing**: Process multiple items at once when possible
3. **Model Loading**: First run loads transformers (takes time)
4. **Memory**: Large case files may require more memory
5. **Caching**: Consider caching results for repeated analysis

## Troubleshooting

### Transformers Not Available

If transformers library is not installed:

```bash
pip install transformers torch
```

### GPU Memory Issues

Use CPU mode:

```python
analyzer = EvidenceLLMAnalyzer()
# Already defaults to CPU (-1 device)
```

### Slow Performance

- Use batch processing
- Reduce max_workers if memory-constrained
- Limit document size to first 512 characters for analysis

### API Errors

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Integration with court records APIs
- Real-time case monitoring
- Predictive outcome analysis
- Multi-language support
- Custom model training
- Cloud deployment options

## Support

For issues or questions, refer to:

- Code comments in source files
- Test suite in `tests/test_ai_modules.py`
- GitHub integration for issue tracking
