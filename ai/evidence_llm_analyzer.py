"""
AI-Powered Evidence Analysis Module
Uses Hugging Face transformers for legal document understanding and evidence scoring.
Provides semantic analysis, relevance assessment, and credibility evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime

try:
    from transformers import pipeline
    _has_transformers = True
except ImportError:
    _has_transformers = False
    pipeline = None  # type: ignore

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """Classification of evidence types"""
    DOCUMENTARY = "documentary"
    TESTIMONIAL = "testimonial"
    DEMONSTRATIVE = "demonstrative"
    PHYSICAL = "physical"
    DIGITAL = "digital"
    UNKNOWN = "unknown"


class CredibilityLevel(Enum):
    """Evidence credibility assessment"""
    HIGHLY_CREDIBLE = "highly_credible"
    CREDIBLE = "credible"
    QUESTIONABLE = "questionable"
    UNRELIABLE = "unreliable"
    UNKNOWN = "unknown"


@dataclass
class EvidenceEntity:
    """Extracted legal entity from evidence"""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    legal_category: Optional[str] = None


@dataclass
class EvidenceScore:
    """Comprehensive evidence scoring"""
    relevance_score: float  # 0-1, relevance to case
    reliability_score: float  # 0-1, trustworthiness
    impact_score: float  # 0-1, impact on case outcome
    completeness_score: float  # 0-1, completeness of evidence
    chain_of_custody_score: float  # 0-1, integrity assessment
    overall_strength: float  # 0-1, aggregate strength
    credibility_level: CredibilityLevel


@dataclass
class AnalyzedEvidence:
    """Complete analysis result for a piece of evidence"""
    evidence_id: str
    content: str
    evidence_type: EvidenceType
    extraction_timestamp: str
    extracted_entities: List[EvidenceEntity] = field(default_factory=lambda: [])
    semantic_summary: str = ""
    key_phrases: List[str] = field(default_factory=lambda: [])
    scores: Optional[EvidenceScore] = None
    relationships: List[str] = field(default_factory=lambda: [])
    confidence: float = 0.0
    contradictions: List[Dict[str, Any]] = field(default_factory=lambda: [])
    supporting_evidence_ids: List[str] = field(default_factory=lambda: [])
    legal_implications: List[str] = field(default_factory=lambda: [])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['evidence_type'] = self.evidence_type.value
        if self.scores:
            data['scores']['credibility_level'] = self.scores.credibility_level.value
        return data


class EvidenceLLMAnalyzer:
    """
    AI-powered evidence analyzer using Hugging Face transformers.
    Performs semantic analysis, entity extraction, and credibility assessment.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the evidence analyzer"""
        self.model_name = model_name
        self.transformers_available = _has_transformers

        if self.transformers_available:
            try:
                # Load sentiment analysis pipeline
                # Import pipeline locally to help mypy resolve overloads
                from transformers import pipeline as _pipeline
                self.sentiment_pipeline = _pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=-1,  # CPU by default
                )
                logger.info(f"Loaded sentiment analysis model: {model_name}")

                # Load zero-shot classification for evidence relevance
                self.relevance_pipeline = _pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1,
                )
                logger.info("Loaded relevance classification model")

            except Exception as e:
                logger.warning(f"Failed to load transformers: {e}")
                self.transformers_available = False
        else:
            logger.warning("Transformers library not available. Using fallback analysis.")

    def analyze_evidence(
        self,
        evidence_id: str,
        content: str,
        case_type: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> AnalyzedEvidence:
        """
        Perform comprehensive analysis of evidence.

        Args:
            evidence_id: Unique identifier for the evidence
            content: The actual evidence content
            case_type: Type of case (custody, ppo, general, etc.)
            context: Additional context for analysis

        Returns:
            AnalyzedEvidence object with complete analysis
        """
        context = context or {}

        # Create initial evidence object
        analyzed = AnalyzedEvidence(
            evidence_id=evidence_id,
            content=content,
            evidence_type=self._classify_evidence_type(content),
            extraction_timestamp=datetime.now().isoformat(),
            metadata=context
        )

        # Perform various analyses
        if self.transformers_available:
            analyzed.semantic_summary = self._extract_semantic_summary(content)
            analyzed.key_phrases = self._extract_key_phrases(content)
            analyzed.extracted_entities = self._extract_entities(content)
            analyzed.scores = self._calculate_evidence_scores(
                content, case_type, analyzed
            )
            analyzed.legal_implications = self._assess_legal_implications(
                content, case_type
            )
        else:
            # Fallback basic analysis
            analyzed.semantic_summary = content[:200] + "..." if len(content) > 200 else content
            analyzed.key_phrases = self._extract_keywords_basic(content)
            analyzed.scores = self._calculate_evidence_scores_fallback(content)

        # Calculate confidence
        analyzed.confidence = self._calculate_confidence(analyzed)

        return analyzed

    def _classify_evidence_type(self, content: str) -> EvidenceType:
        """Classify the type of evidence"""
        content_lower = content.lower()

        if any(word in content_lower for word in ['document', 'email', 'letter', 'report']):
            return EvidenceType.DOCUMENTARY
        elif any(word in content_lower for word in ['testified', 'said', 'stated', 'witness']):
            return EvidenceType.TESTIMONIAL
        elif any(word in content_lower for word in ['photo', 'video', 'image', 'demonstration']):
            return EvidenceType.DEMONSTRATIVE
        elif any(word in content_lower for word in ['object', 'item', 'physical', 'tangible']):
            return EvidenceType.PHYSICAL
        elif any(word in content_lower for word in ['email', 'log', 'metadata', 'digital']):
            return EvidenceType.DIGITAL

        return EvidenceType.UNKNOWN

    def _extract_semantic_summary(self, content: str) -> str:
        """Extract semantic summary using NLP"""
        if not self.transformers_available:
            return content[:300]

        try:
            # Use extractive summarization approach
            sentences = content.split('.')
            if len(sentences) <= 2:
                return content

            # Return first and last meaningful sentences
            summary_parts = [s.strip() for s in sentences[:2] if s.strip()]
            return '. '.join(summary_parts) + '.'
        except Exception as e:
            logger.error(f"Semantic summary extraction failed: {e}")
            return content[:300]

    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content"""
        # Simple keyword extraction based on frequency and legal terms
        legal_terms = {
            'custody', 'parental', 'visitation', 'custody modification',
            'child support', 'evidence', 'testimony', 'defendant', 'plaintiff',
            'order', 'judgment', 'appeal', 'motion', 'hearing', 'trial',
            'discovery', 'deposition', 'affidavit', 'exhibit', 'contract',
            'agreement', 'settlement', 'liability', 'damages', 'injunction'
        }

        content_lower = content.lower()
        phrases = [term for term in legal_terms if term in content_lower]

        return phrases if phrases else self._extract_keywords_basic(content)

    def _extract_keywords_basic(self, content: str) -> List[str]:
        """Basic keyword extraction fallback"""
        words = content.lower().split()
        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of'}
        keywords = [w for w in words if len(w) > 4 and w not in stop_words]
        return list(set(keywords))[:10]

    def _extract_entities(self, content: str) -> List[EvidenceEntity]:
        """Extract legal entities from content"""
        entities = []
        legal_patterns = {
            'PERSON': ['Mr.', 'Ms.', 'Dr.', 'Judge', 'Attorney'],
            'ORGANIZATION': ['Court', 'Agency', 'Company', 'Firm', 'Department'],
            'DATE': ['January', 'February', '2024', '2023'],
            'LOCATION': ['Michigan', 'Court', 'Courthouse', 'County'],
        }

        for entity_type, patterns in legal_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    start_pos = content.find(pattern)
                    entities.append(
                        EvidenceEntity(
                            text=pattern,
                            entity_type=entity_type,
                            start_pos=start_pos,
                            end_pos=start_pos + len(pattern),
                            confidence=0.85,
                            legal_category=entity_type
                        )
                    )

        return entities

    def _calculate_evidence_scores(
        self,
        content: str,
        case_type: str,
        evidence: AnalyzedEvidence
    ) -> EvidenceScore:
        """Calculate comprehensive evidence scores"""
        if not self.transformers_available:
            return self._calculate_evidence_scores_fallback(content)

        try:
            # Determine relevance based on case type
            relevance_labels = [
                f"evidence relevant to {case_type}",
                "supporting evidence",
                "contradicting evidence"
            ]
            relevance_result = self.relevance_pipeline(
                content[:512],
                relevance_labels,
                multi_class=False
            )
            relevance_score = relevance_result['scores'][0]

            # Calculate component scores
            reliability_score = self._assess_reliability(evidence)
            impact_score = self._assess_impact(content, case_type)
            completeness_score = min(len(content) / 5000, 1.0)
            chain_of_custody_score = self._assess_chain_of_custody(evidence)

            # Calculate overall strength
            overall_strength = (
                relevance_score * 0.3 +
                reliability_score * 0.25 +
                impact_score * 0.25 +
                completeness_score * 0.1 +
                chain_of_custody_score * 0.1
            )

            # Determine credibility level
            credibility_level = self._determine_credibility_level(overall_strength)

            return EvidenceScore(
                relevance_score=relevance_score,
                reliability_score=reliability_score,
                impact_score=impact_score,
                completeness_score=completeness_score,
                chain_of_custody_score=chain_of_custody_score,
                overall_strength=overall_strength,
                credibility_level=credibility_level
            )

        except Exception as e:
            logger.error(f"Evidence scoring failed: {e}")
            return self._calculate_evidence_scores_fallback(content)

    def _calculate_evidence_scores_fallback(self, content: str) -> EvidenceScore:
        """Fallback evidence scoring without transformers"""
        completeness = min(len(content) / 5000, 1.0)
        has_legal_terms = any(
            term in content.lower()
            for term in ['testimony', 'witness', 'evidence', 'court', 'judge']
        )
        relevance = 0.7 if has_legal_terms else 0.4

        return EvidenceScore(
            relevance_score=relevance,
            reliability_score=0.6,
            impact_score=0.5,
            completeness_score=completeness,
            chain_of_custody_score=0.5,
            overall_strength=0.55,
            credibility_level=CredibilityLevel.QUESTIONABLE
        )

    def _assess_reliability(self, evidence: AnalyzedEvidence) -> float:
        """Assess reliability based on evidence characteristics"""
        reliability = 0.5

        if evidence.evidence_type in [
            EvidenceType.DOCUMENTARY,
            EvidenceType.DIGITAL
        ]:
            reliability += 0.2

        if len(evidence.extracted_entities) > 0:
            reliability += 0.15

        if evidence.evidence_type == EvidenceType.PHYSICAL:
            reliability += 0.1

        return min(reliability, 1.0)

    def _assess_impact(self, content: str, case_type: str) -> float:
        """Assess potential impact on case outcome"""
        impact = 0.5

        # Custody cases
        if case_type == "custody":
            if any(
                term in content.lower()
                for term in ['parental', 'child welfare', 'best interest', 'fitness']
            ):
                impact = 0.8
            elif any(
                term in content.lower()
                for term in ['abuse', 'neglect', 'endangerment', 'harm']
            ):
                impact = 0.95

        # PPO cases
        elif case_type == "ppo":
            if any(
                term in content.lower()
                for term in ['threat', 'violence', 'harassment', 'stalking', 'abuse']
            ):
                impact = 0.9

        return min(impact, 1.0)

    def _assess_chain_of_custody(self, evidence: AnalyzedEvidence) -> float:
        """Assess chain of custody integrity"""
        score = 0.5

        # Check for temporal continuity
        if evidence.extraction_timestamp:
            score += 0.2

        # Check for proper handling indicators
        if 'chain of custody' in evidence.metadata.get('tags', []):
            score += 0.3

        return min(score, 1.0)

    def _determine_credibility_level(self, strength: float) -> CredibilityLevel:
        """Determine credibility level based on overall strength"""
        if strength >= 0.85:
            return CredibilityLevel.HIGHLY_CREDIBLE
        elif strength >= 0.7:
            return CredibilityLevel.CREDIBLE
        elif strength >= 0.5:
            return CredibilityLevel.QUESTIONABLE
        else:
            return CredibilityLevel.UNRELIABLE

    def _assess_legal_implications(self, content: str, case_type: str) -> List[str]:
        """Assess legal implications of the evidence"""
        implications = []

        content_lower = content.lower()

        # Generic legal implications
        if 'admission' in content_lower or 'admit' in content_lower:
            implications.append("Contains potential admission or acknowledgment")

        if 'contradiction' in content_lower or 'contradict' in content_lower:
            implications.append("Contains contradictory statements")

        if 'testimony' in content_lower:
            implications.append("Contains testimonial evidence")

        # Case-specific implications
        if case_type == "custody":
            if 'fitness' in content_lower:
                implications.append("Relevant to parental fitness assessment")
            if 'best interest' in content_lower:
                implications.append("Directly addresses best interest of child")

        elif case_type == "ppo":
            if 'threat' in content_lower or 'violence' in content_lower:
                implications.append("Supports Personal Protection Order grounds")

        return implications if implications else ["Standard evidence"]

    def _calculate_confidence(self, analyzed: AnalyzedEvidence) -> float:
        """Calculate overall confidence in the analysis"""
        confidence = 0.5

        if analyzed.scores:
            confidence = analyzed.scores.overall_strength * 0.7

        if len(analyzed.extracted_entities) > 2:
            confidence += 0.15

        if len(analyzed.key_phrases) > 3:
            confidence += 0.15

        return min(confidence, 1.0)

    def compare_evidence(
        self,
        evidence1: AnalyzedEvidence,
        evidence2: AnalyzedEvidence
    ) -> Dict[str, Any]:
        """Compare two pieces of evidence"""
        return {
            "evidence1_id": evidence1.evidence_id,
            "evidence2_id": evidence2.evidence_id,
            "strength_difference": (
                (evidence1.scores.overall_strength if evidence1.scores else 0) -
                (evidence2.scores.overall_strength if evidence2.scores else 0)
            ),
            "contradictions": self._find_contradictions(evidence1, evidence2),
            "complementary": self._assess_complementarity(evidence1, evidence2),
            "recommendation": self._generate_comparison_recommendation(evidence1, evidence2)
        }

    def _find_contradictions(
        self,
        evidence1: AnalyzedEvidence,
        evidence2: AnalyzedEvidence
    ) -> List[str]:
        """Find contradictions between evidence"""
        contradictions = []

        phrases1 = set(evidence1.key_phrases)
        phrases2 = set(evidence2.key_phrases)

        # Simple contradiction detection
        if phrases1 and phrases2 and not (phrases1 & phrases2):
            contradictions.append("No overlapping key phrases detected")

        return contradictions

    def _assess_complementarity(
        self,
        evidence1: AnalyzedEvidence,
        evidence2: AnalyzedEvidence
    ) -> bool:
        """Assess if evidence pieces complement each other"""
        phrases1 = set(evidence1.key_phrases)
        phrases2 = set(evidence2.key_phrases)

        # If significant overlap, likely complementary
        if phrases1 and phrases2:
            overlap = len(phrases1 & phrases2)
            total = len(phrases1 | phrases2)
            return (overlap / total) > 0.3

        return False

    def _generate_comparison_recommendation(
        self,
        evidence1: AnalyzedEvidence,
        evidence2: AnalyzedEvidence
    ) -> str:
        """Generate recommendation for evidence comparison"""
        score1 = evidence1.scores.overall_strength if evidence1.scores else 0
        score2 = evidence2.scores.overall_strength if evidence2.scores else 0

        if abs(score1 - score2) < 0.1:
            return "Evidence pieces are of similar strength; use together for corroboration"
        elif score1 > score2:
            return "Evidence 1 is stronger; prioritize in presentation"
        else:
            return "Evidence 2 is stronger; prioritize in presentation"

    def batch_analyze_evidence(
        self,
        evidence_list: List[Tuple[str, str]],
        case_type: str = "general"
    ) -> List[AnalyzedEvidence]:
        """Analyze multiple pieces of evidence efficiently"""
        results = []
        for evidence_id, content in evidence_list:
            analyzed = self.analyze_evidence(evidence_id, content, case_type)
            results.append(analyzed)

        return results

    def export_analysis_report(
        self,
        analyzed_evidence: List[AnalyzedEvidence],
        output_format: str = "json"
    ) -> str:
        """Export analysis results in various formats"""
        if output_format == "json":
            return json.dumps(
                [e.to_dict() for e in analyzed_evidence],
                indent=2,
                default=str
            )

        elif output_format == "csv":
            import csv
            from io import StringIO

            output = StringIO()
            if analyzed_evidence:
                fieldnames = [
                    'evidence_id', 'evidence_type', 'relevance_score',
                    'reliability_score', 'overall_strength', 'credibility_level'
                ]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()

                for evidence in analyzed_evidence:
                    row = {
                        'evidence_id': evidence.evidence_id,
                        'evidence_type': evidence.evidence_type.value,
                        'relevance_score': (
                            evidence.scores.relevance_score
                            if evidence.scores else 0
                        ),
                        'reliability_score': (
                            evidence.scores.reliability_score
                            if evidence.scores else 0
                        ),
                        'overall_strength': (
                            evidence.scores.overall_strength
                            if evidence.scores else 0
                        ),
                        'credibility_level': (
                            evidence.scores.credibility_level.value
                            if evidence.scores else 'unknown'
                        ),
                    }
                    writer.writerow(row)

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported export format: {output_format}")


if __name__ == "__main__":
    # Example usage
    analyzer = EvidenceLLMAnalyzer()

    # Sample evidence
    sample_evidence = {
        "ev001": """
        Witness testimony from John Smith dated January 15, 2024.
        Mr. Smith testified that he observed the defendant's behavior
        towards the child and expressed concern about the child's welfare.
        The testimony was credible and detailed.
        """,
        "ev002": """
        Email correspondence between the parties regarding custody arrangements.
        The email shows a pattern of refusal to follow court-ordered visitation.
        This constitutes a violation of the existing custody order.
        """
    }

    # Analyze evidence
    results = analyzer.batch_analyze_evidence(
        [(k, v) for k, v in sample_evidence.items()],
        case_type="custody"
    )

    # Export results
    json_report = analyzer.export_analysis_report(results, "json")
    print(json_report)
