"""AI and Machine Learning modules for legal document processing.

This package provides:
- NLP-based document processing
- Evidence analysis with LLM support
- Argument reasoning and analysis
- Pipeline orchestration for AI workflows
"""

from .nlp_document_processor import (
    NLPDocumentProcessor,
    DocumentMetadata,
    DocumentType,
    SentimentType,
    EntityInfo,
    Relationship,
)

try:
    from .evidence_llm_analyzer import (
        EvidenceLLMAnalyzer,
        AnalyzedEvidence,
        EvidenceType,
        CredibilityLevel,
        EvidenceScore,
    )
except ImportError:
    pass

try:
    from .argument_reasoning import ArgumentReasoningGraph, ArgumentAnalysis, ArgumentType, RelationType
except ImportError:
    pass

try:
    from .ai_pipeline_orchestrator import AIPipelineOrchestrator
except ImportError:
    pass

__all__ = [
    "NLPDocumentProcessor",
    "DocumentMetadata",
    "DocumentType",
    "SentimentType",
    "EntityInfo",
    "Relationship",
    "EvidenceLLMAnalyzer",
    "AnalyzedEvidence",
    "EvidenceType",
    "CredibilityLevel",
    "EvidenceScore",
    "ArgumentReasoningGraph",
    "ArgumentAnalysis",
    "ArgumentType",
    "RelationType",
    "AIPipelineOrchestrator",
]
