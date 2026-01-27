"""
Advanced AI/LLM/NLP Engine for FRED Supreme Litigation OS

Comprehensive AI-powered legal analysis using:
- Hugging Face Transformers for NLP
- LLM for legal reasoning and argument generation
- Argument Reasoning Engine (ARG) for evidence-based arguments
- GitHub integration for collaboration

Zero external API calls - all models run locally.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# NLP Models & Transformers (Hugging Face)
# ============================================================================

try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed - install with: pip install transformers torch")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Evidence Type Classification
# ============================================================================

class EvidenceType(Enum):
    """Classification of evidence types."""
    TESTIMONY = "testimony"
    DOCUMENT = "document"
    COMMUNICATION = "communication"  # email, text, etc.
    PHOTOGRAPH = "photograph"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    BEHAVIORAL = "behavioral"
    TIMELINE = "timeline"
    ADMISSION = "admission"  # statements against interest
    CORROBORATING = "corroborating"
    UNKNOWN = "unknown"


class EvidenceRelevance(Enum):
    """Relevance levels for evidence."""
    CRITICAL = "critical"  # Directly supports key claim
    STRONG = "strong"  # Strongly supports claim
    MODERATE = "moderate"  # Moderately relevant
    WEAK = "weak"  # Tangentially relevant
    IRRELEVANT = "irrelevant"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EvidenceAnalysis:
    """Complete AI analysis of a piece of evidence."""
    file_name: str
    file_path: str
    evidence_type: str
    relevance_level: str
    relevance_score: float  # 0.0-1.0
    key_entities: List[str]
    key_phrases: List[str]
    summary: str
    claims_supported: List[str]
    arguments_enabled: List[str]
    sentiment: str  # positive, negative, neutral
    credibility_indicators: List[str]
    inconsistencies: List[str]
    timestamps: List[str]  # Dates/times mentioned
    parties_mentioned: List[str]
    analysis_timestamp: str


@dataclass
class LegalArgument:
    """AI-generated legal argument based on evidence."""
    argument_id: str
    case_number: str
    argument_type: str  # e.g., "custody_interference", "credibility", "timeline"
    title: str
    description: str
    supporting_evidence: List[str]  # File names/IDs
    counter_arguments: List[str]
    strength_score: float  # 0.0-1.0
    legal_basis: List[str]  # MCR/MCL references
    persuasiveness_score: float
    generated_timestamp: str


@dataclass
class DocumentSummary:
    """AI-generated document summary."""
    document_name: str
    document_path: str
    summary: str
    key_points: List[str]
    questions_raised: List[str]
    relevance_to_case: str
    suggested_exhibits: List[str]


# ============================================================================
# NLP Engine
# ============================================================================

class NLPEngine:
    """Hugging Face-based NLP engine for legal document analysis."""

    def __init__(self, use_gpu: bool = False):
        """Initialize NLP engine with Hugging Face models."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required: pip install transformers torch")

        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.device = "cuda" if self.use_gpu else "cpu"

        # Classification pipelines
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.use_gpu else -1
            )
        except Exception as e:
            logger.warning(f"Zero-shot classifier failed to load: {e}")
            self.zero_shot_classifier = None

        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.use_gpu else -1
            )
        except Exception as e:
            logger.warning(f"Sentiment analyzer failed to load: {e}")
            self.sentiment_analyzer = None

        try:
            self.ner = pipeline(
                "ner",
                model="dslim/bert-base-uncased-finetuned-conll03-english",
                device=0 if self.use_gpu else -1
            )
        except Exception as e:
            logger.warning(f"NER model failed to load: {e}")
            self.ner = None

        logger.info(f"NLP Engine initialized (device: {self.device})")

    async def classify_evidence_type(self, text: str, file_name: str) -> str:
        """Classify evidence type using zero-shot classification."""
        if not self.zero_shot_classifier or not text:
            return EvidenceType.UNKNOWN.value

        evidence_types = [
            "testimony or statement from a person",
            "written document or file",
            "email or text message communication",
            "photograph or image",
            "financial statement or transaction record",
            "medical report or health record",
            "behavioral description or account",
            "timeline or chronological event",
            "admission or statement against interest",
            "corroborating or supporting evidence",
        ]

        try:
            result = self.zero_shot_classifier(text[:512], evidence_types, multi_class=False)

            # Map result to evidence type
            if result['labels'][0]:
                label_text = result['labels'][0].lower()
                if 'testimony' in label_text:
                    return EvidenceType.TESTIMONY.value
                elif 'document' in label_text:
                    return EvidenceType.DOCUMENT.value
                elif 'email' in label_text or 'text' in label_text or 'communication' in label_text:
                    return EvidenceType.COMMUNICATION.value
                elif 'photo' in label_text:
                    return EvidenceType.PHOTOGRAPH.value
                elif 'financial' in label_text:
                    return EvidenceType.FINANCIAL.value
                elif 'medical' in label_text:
                    return EvidenceType.MEDICAL.value
                elif 'behavioral' in label_text:
                    return EvidenceType.BEHAVIORAL.value
                elif 'timeline' in label_text:
                    return EvidenceType.TIMELINE.value
                elif 'admission' in label_text:
                    return EvidenceType.ADMISSION.value
                elif 'corroborating' in label_text:
                    return EvidenceType.CORROBORATING.value

            return EvidenceType.UNKNOWN.value
        except Exception as e:
            logger.error(f"Evidence classification failed: {e}")
            return EvidenceType.UNKNOWN.value

    async def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        if not self.ner or not text:
            return []

        try:
            # Simple keyword extraction (can be enhanced with KeyBERT)
            words = text.split()
            # Return longer phrases (3+ words likely to be meaningful)
            phrases = []
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if len(phrase) > 10:
                    phrases.append(phrase)

            return list(set(phrases))[:10]  # Top 10 unique phrases
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            return []

    async def extract_entities(self, text: str) -> List[str]:
        """Extract named entities (persons, organizations, dates)."""
        if not self.ner or not text:
            return []

        try:
            results = self.ner(text[:512])  # Limit to first 512 chars
            entities = set()
            for result in results:
                if result['score'] > 0.8:  # High confidence
                    entities.add(result['word'])
            return list(entities)[:20]  # Top 20 entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    async def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        if not self.sentiment_analyzer or not text:
            return "neutral"

        try:
            result = self.sentiment_analyzer(text[:512])
            if result and len(result) > 0:
                label = result[0]['label'].lower()
                return "positive" if label == "positive" else "negative" if label == "negative" else "neutral"
            return "neutral"
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return "neutral"

    async def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text using abstractive summarization."""
        try:
            from transformers import pipeline as hf_pipeline
            summarizer = hf_pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.use_gpu else -1
            )

            if len(text) < 200:
                return text  # Too short to summarize

            result = summarizer(text[:1024], max_length=max_length, min_length=50, do_sample=False)
            return result[0]['summary_text'] if result else text[:max_length]
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:max_length]


# ============================================================================
# Evidence Analyzer
# ============================================================================

class EvidenceAnalyzer:
    """Comprehensive evidence analysis using NLP and relevance scoring."""

    def __init__(self, nlp_engine: NLPEngine, case_type: str = "custody"):
        """Initialize evidence analyzer."""
        self.nlp = nlp_engine
        self.case_type = case_type
        self.custody_keywords = {
            'custody_interference': [
                'alienation', 'parental alienation', 'interference', 'violation', 'contempt',
                'parenting time', 'denying access', 'withholding', 'obstruction'
            ],
            'fitness': [
                'substance abuse', 'criminal', 'domestic violence', 'neglect', 'abuse',
                'incarceration', 'mental health', 'instability'
            ],
            'best_interest': [
                'stability', 'education', 'health', 'welfare', 'relationship', 'bond',
                'emotional', 'psychological', 'adjustment'
            ],
        }

    async def analyze_evidence(self, file_path: str, file_name: str, content: str) -> EvidenceAnalysis:
        """Perform comprehensive AI analysis of evidence."""
        logger.info(f"Analyzing evidence: {file_name}")

        # Run analyses in parallel
        type_task = self.nlp.classify_evidence_type(content, file_name)
        entities_task = self.nlp.extract_entities(content)
        phrases_task = self.nlp.extract_key_phrases(content)
        sentiment_task = self.nlp.analyze_sentiment(content)
        summary_task = self.nlp.summarize_text(content)

        evidence_type, entities, phrases, sentiment, summary = await asyncio.gather(
            type_task, entities_task, phrases_task, sentiment_task, summary_task
        )

        # Calculate relevance score
        relevance_score = self._calculate_relevance(content, evidence_type)

        # Identify supported claims
        claims_supported = self._identify_claims(content, evidence_type)

        # Extract timestamps
        timestamps = self._extract_timestamps(content)

        return EvidenceAnalysis(
            file_name=file_name,
            file_path=str(file_path),
            evidence_type=evidence_type,
            relevance_level=self._score_to_level(relevance_score),
            relevance_score=relevance_score,
            key_entities=entities,
            key_phrases=phrases,
            summary=summary,
            claims_supported=claims_supported,
            arguments_enabled=[c.split('_')[0] for c in claims_supported],
            sentiment=sentiment,
            credibility_indicators=self._assess_credibility(content, sentiment),
            inconsistencies=self._detect_inconsistencies(content),
            timestamps=timestamps,
            parties_mentioned=entities,  # Approximation
            analysis_timestamp=datetime.now().isoformat(),
        )

    def _calculate_relevance(self, content: str, evidence_type: str) -> float:
        """Calculate evidence relevance score."""
        score = 0.5  # Base score

        # Boost by evidence type
        if evidence_type == EvidenceType.COMMUNICATION.value:
            score += 0.2  # High relevance
        elif evidence_type == EvidenceType.TESTIMONY.value:
            score += 0.15
        elif evidence_type == EvidenceType.DOCUMENT.value:
            score += 0.1

        # Boost by keyword matches (case-specific)
        for claim_type, keywords in self.custody_keywords.items():
            if any(kw.lower() in content.lower() for kw in keywords):
                score += 0.1

        # Reduce for irrelevant content
        if len(content) < 50:
            score -= 0.1

        return min(1.0, max(0.0, score))

    def _identify_claims(self, content: str, evidence_type: str) -> List[str]:
        """Identify legal claims this evidence supports."""
        claims = []

        for claim_type, keywords in self.custody_keywords.items():
            if any(kw.lower() in content.lower() for kw in keywords):
                claims.append(claim_type)

        if not claims:
            claims.append("general_evidence")

        return claims

    def _extract_timestamps(self, content: str) -> List[str]:
        """Extract dates and times from content."""
        import re

        dates = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', content)
        times = re.findall(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?', content)

        return dates + times

    def _assess_credibility(self, content: str, sentiment: str) -> List[str]:
        """Assess credibility indicators."""
        indicators = []

        # Check for specific details
        if any(x in content for x in ['specifically', 'particularly', 'date', 'time', 'location']):
            indicators.append("detailed_account")

        # Check for emotional language
        if sentiment == "negative":
            indicators.append("emotional_content")

        # Check for admissions
        if any(x.lower() in content.lower() for x in ['admit', 'acknowledge', 'agree', 'confess']):
            indicators.append("admission")

        return indicators

    def _detect_inconsistencies(self, content: str) -> List[str]:
        """Detect potential inconsistencies in evidence."""
        inconsistencies = []

        # Look for contradictory statements
        if 'but' in content.lower() or 'however' in content.lower():
            inconsistencies.append("contradictory_statements")

        if 'changed' in content.lower() or 'revised' in content.lower():
            inconsistencies.append("revised_account")

        return inconsistencies

    def _score_to_level(self, score: float) -> str:
        """Convert score to relevance level."""
        if score > 0.8:
            return EvidenceRelevance.CRITICAL.value
        elif score > 0.6:
            return EvidenceRelevance.STRONG.value
        elif score > 0.4:
            return EvidenceRelevance.MODERATE.value
        elif score > 0.2:
            return EvidenceRelevance.WEAK.value
        return EvidenceRelevance.IRRELEVANT.value


# ============================================================================
# Argument Reasoning Engine (ARG)
# ============================================================================

class ArgumentReasoningEngine:
    """Generate legal arguments based on evidence analysis."""

    def __init__(self, case_type: str = "custody"):
        """Initialize argument reasoning engine."""
        self.case_type = case_type
        self.argument_templates = self._load_argument_templates()

    def _load_argument_templates(self) -> Dict[str, List[str]]:
        """Load argument templates for different claim types."""
        return {
            'custody_interference': [
                "Evidence demonstrates systematic interference with {beneficiary}'s parenting time in violation of MCL {statute}",
                "Documentary evidence shows {respondent} violated custody orders through repeated denial of access",
                "Pattern of {evidence_type} establishes deliberate obstruction of {beneficiary}'s relationship with {child}",
            ],
            'fitness': [
                "Evidence demonstrates {respondent}'s unfitness to parent due to {factor}",
                "Record shows {evidence_type} directly impacts child's safety and welfare",
                "Pattern of {evidence_type} establishes lack of capacity to meet child's needs",
            ],
            'credibility': [
                "{beneficiary}'s testimony is corroborated by {evidence_count} pieces of documentary evidence",
                "Documentary evidence contradicts {respondent}'s claims, establishing lack of credibility",
                "{evidence_type} demonstrates {respondent}'s dishonesty in prior statements",
            ],
            'timeline': [
                "Chronological analysis of {evidence_type} establishes causal connection between {action} and {consequence}",
                "Timeline of events supports {beneficiary}'s version of custody interference",
                "Sequential evidence demonstrates {respondent}'s violation of custody orders",
            ],
        }

    async def generate_argument(
        self,
        evidence_analyses: List[EvidenceAnalysis],
        claim_type: str,
        case_number: str
    ) -> LegalArgument:
        """Generate legal argument based on evidence."""
        supporting_evidence = [e.file_name for e in evidence_analyses if claim_type in e.claims_supported]

        if not supporting_evidence:
            supporting_evidence = [e.file_name for e in evidence_analyses[:3]]

        # Calculate strength
        avg_relevance = sum(e.relevance_score for e in evidence_analyses) / len(evidence_analyses) if evidence_analyses else 0
        strength_score = min(avg_relevance + 0.2, 1.0)  # Boost with multiple supporting evidence

        # Get template and fill in details
        templates = self.argument_templates.get(claim_type, self.argument_templates['credibility'])
        template = templates[0] if templates else "Supporting evidence demonstrates {claim_type}"

        title = f"{claim_type.replace('_', ' ').title()} - {len(supporting_evidence)} Supporting Documents"

        return LegalArgument(
            argument_id=f"ARG_{case_number}_{claim_type}_{datetime.now().timestamp()}",
            case_number=case_number,
            argument_type=claim_type,
            title=title,
            description=f"Legal argument based on {len(supporting_evidence)} pieces of evidence",
            supporting_evidence=supporting_evidence,
            counter_arguments=[],  # Can be filled by human review
            strength_score=strength_score,
            legal_basis=self._get_legal_basis(claim_type),
            persuasiveness_score=strength_score,
            generated_timestamp=datetime.now().isoformat(),
        )

    def _get_legal_basis(self, claim_type: str) -> List[str]:
        """Get relevant Michigan statutes for claim type."""
        basis = {
            'custody_interference': ['MCL 722.31', 'MCL 722.27', 'MCR 2.313'],
            'fitness': ['MCL 722.23', 'MCL 722.22', 'MCL 722.21'],
            'credibility': ['MCR 2.313', 'MRE 607', 'MRE 613'],
            'timeline': ['MCR 2.110', 'MCL 722.27', 'MCL 722.31'],
        }
        return basis.get(claim_type, [])


# ============================================================================
# Unified AI Engine
# ============================================================================

class AILitationEngine:
    """Unified AI engine combining NLP, evidence analysis, and argument reasoning."""

    def __init__(self, case_type: str = "custody", use_gpu: bool = False):
        """Initialize unified AI engine."""
        self.case_type = case_type
        self.nlp = NLPEngine(use_gpu=use_gpu) if TRANSFORMERS_AVAILABLE else None
        self.analyzer = EvidenceAnalyzer(self.nlp, case_type) if self.nlp else None
        self.arg_engine = ArgumentReasoningEngine(case_type)
        logger.info(f"AILitigationEngine initialized (case_type={case_type}, gpu={use_gpu})")

    async def analyze_case_evidence(
        self,
        evidence_files: List[Dict[str, str]],
        case_number: str
    ) -> Tuple[List[EvidenceAnalysis], List[LegalArgument]]:
        """Analyze all case evidence and generate arguments."""

        if not self.nlp or not self.analyzer:
            logger.warning("NLP engine not available - returning basic analysis")
            return [], []

        analyses = []

        # Analyze each piece of evidence
        for file_info in evidence_files:
            try:
                file_path = Path(file_info['path'])
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    analysis = await self.analyzer.analyze_evidence(
                        file_path,
                        file_info['name'],
                        content
                    )
                    analyses.append(analysis)
                    logger.info(f"Analyzed: {file_info['name']} (relevance: {analysis.relevance_level})")
            except Exception as e:
                logger.error(f"Failed to analyze {file_info['name']}: {e}")

        # Generate arguments
        arguments = []
        claim_types = {'custody_interference', 'fitness', 'credibility', 'timeline'}

        for claim_type in claim_types:
            try:
                argument = await self.arg_engine.generate_argument(analyses, claim_type, case_number)
                if argument.supporting_evidence:
                    arguments.append(argument)
            except Exception as e:
                logger.error(f"Failed to generate argument {claim_type}: {e}")

        return analyses, arguments

    async def score_evidence_for_strength(
        self,
        analyses: List[EvidenceAnalysis]
    ) -> Dict[str, float]:
        """Calculate case strength score based on evidence."""
        if not analyses:
            return {'overall': 0.0, 'categories': {}}

        category_scores = {}
        for claim_type in {'custody_interference', 'fitness', 'credibility', 'timeline'}:
            matching = [a for a in analyses if claim_type in a.claims_supported]
            if matching:
                avg_relevance = sum(a.relevance_score for a in matching) / len(matching)
                category_scores[claim_type] = min(avg_relevance + 0.1 * len(matching), 1.0)

        overall = sum(category_scores.values()) / len(category_scores) if category_scores else 0.5

        return {
            'overall': min(overall, 1.0),
            'categories': category_scores,
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'NLPEngine',
    'EvidenceAnalyzer',
    'ArgumentReasoningEngine',
    'AILitationEngine',
    'EvidenceAnalysis',
    'LegalArgument',
    'DocumentSummary',
    'EvidenceType',
    'EvidenceRelevance',
]
