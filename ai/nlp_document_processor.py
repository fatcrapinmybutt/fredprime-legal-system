"""NLP-Based Document Processor Module
Performs entity extraction, sentiment analysis, document classification,
and relationship extraction for legal documents.
"""

from __future__ import annotations

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import re
import threading
import os

# Check transformers availability at runtime
try:
    import transformers  # noqa: F401
    _transformers_available = True
except ImportError:
    _transformers_available = False

logger = logging.getLogger(__name__)


class SentimentType(Enum):
    """Document sentiment classification"""
    HIGHLY_POSITIVE = "highly_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    HIGHLY_NEGATIVE = "highly_negative"


class DocumentType(Enum):
    """Classification of document types"""
    MOTION = "motion"
    AFFIDAVIT = "affidavit"
    COMPLAINT = "complaint"
    ANSWER = "answer"
    DISCOVERY = "discovery"
    NOTICE = "notice"
    ORDER = "order"
    CORRESPONDENCE = "correspondence"
    REPORT = "report"
    EVIDENCE = "evidence"
    UNKNOWN = "unknown"


@dataclass
class EntityInfo:
    """Information about an extracted entity"""
    text: str
    entity_type: str  # PERSON, ORG, DATE, LOCATION, etc.
    start_pos: int
    end_pos: int
    confidence: float
    context: Optional[str] = None


@dataclass
class Relationship:
    """Relationship between two entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: Optional[str] = None


@dataclass
class DocumentMetadata:
    """Metadata about processed document"""
    title: str
    document_type: DocumentType
    extracted_date: Optional[str]
    parties_involved: List[str]
    jurisdiction: Optional[str]
    summary: str
    entities: List[EntityInfo] = field(default_factory=lambda: [])
    relationships: List[Relationship] = field(default_factory=lambda: [])
    sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = 0.0
    key_concepts: List[str] = field(default_factory=lambda: [])
    action_items: List[str] = field(default_factory=lambda: [])
    deadlines: List[str] = field(default_factory=lambda: [])
    confidence_score: float = 0.0

    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['document_type'] = self.document_type.value
        data['sentiment'] = self.sentiment.value
        return data


class NLPDocumentProcessor:
    """
    NLP-based document processor for legal documents.
    Handles entity extraction, sentiment analysis, classification, and relationship extraction.
    """

    # Legal entity type patterns
    LEGAL_PATTERNS = {
        'PERSON': [
            r'\b(?:Mr\.|Ms\.|Dr\.|Judge|Attorney|Esq\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        ],
        'ORGANIZATION': [
            r'\b(?:Court|Agency|Company|Firm|Department|Court of Appeals|Supreme Court)',
            r'\b[A-Z][a-z]+\s+(?:Court|Agency|Corporation|Inc\.|LLC|LLP)'
        ],
        'LOCATION': [
            r'\b(?:Michigan|Ohio|Indiana|Illinois|Wisconsin|County|District)',
            r'\b(?:Third Circuit|Fourth Circuit|Federal Court)'
        ],
        'DATE': [
            r'\b(?:January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}'
        ],
        'LEGAL_CONCEPT': [
            r'\b(?:custody|visitation|support|modification|enforcement|violation)',
            r'\b(?:motion|affidavit|complaint|order|judgment|appeal)'
        ]
    }

    DOCUMENT_PATTERNS = {
        DocumentType.MOTION: [
            r'MOTION\s+(?:FOR|TO)', r'Motion for', r'The Plaintiff/Defendant moves'
        ],
        DocumentType.AFFIDAVIT: [
            r'AFFIDAVIT', r'State of Michigan', r'County of', r'I,\s+[A-Z]'
        ],
        DocumentType.COMPLAINT: [
            r'COMPLAINT', r'VERIFIED COMPLAINT', r'Plaintiff\s+v\.\s+Defendant'
        ],
        DocumentType.ANSWER: [
            r'ANSWER', r'AFFIRMATIVE DEFENSE', r'The Defendant admits'
        ],
        DocumentType.ORDER: [
            r'ORDER', r'IT IS ORDERED', r'THIS COURT ORDERS'
        ],
        DocumentType.NOTICE: [
            r'NOTICE', r'NOTICE OF', r'Notice is hereby given'
        ],
        DocumentType.CORRESPONDENCE: [
            r'(?:RE:|Subject:|Regarding:)'
        ]
    }

    def __init__(self):
        """Initialize the NLP document processor"""
        self.transformers_available = _transformers_available
        self.sentiment_pipeline: Optional[Any] = None
        self.ner_pipeline: Optional[Any] = None

        if self.transformers_available:
            try:
                # Import pipeline locally to keep static analysis simple
                from transformers import pipeline as _pipeline  # type: ignore

                self.sentiment_pipeline = _pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,
                )
                logger.info("Loaded sentiment analysis pipeline")

                try:
                    self.ner_pipeline = _pipeline(
                        "token-classification",
                        model="dslim/bert-base-NER",
                        aggregation_strategy="simple",
                        device=-1,
                    )
                    logger.info("Loaded NER pipeline")
                except Exception as e:
                    logger.warning("NER pipeline failed: %s", e)

            except Exception as e:
                logger.warning("Failed to load pipelines: %s", e)
                self.transformers_available = False

    def process_document(
        self,
        content: str,
        document_title: str = "Untitled",
        case_context: Optional[Dict[str, Any]] = None,
        delegate_to_background: bool = False
    ) -> DocumentMetadata:
        """
        Process a legal document with comprehensive NLP analysis.

        If delegate_to_background is True, delegate heavy processing to a background
        thread and return a lightweight placeholder metadata object immediately.
        """
        case_context = case_context or {}

        if delegate_to_background:
            # Quick classification and summary, then delegate full processing
            doc_type = self._classify_document_type(content)
            summary = self._generate_summary(content)
            extracted_date = self._extract_date(content)

            placeholder = DocumentMetadata(
                title=document_title,
                document_type=doc_type,
                extracted_date=extracted_date,
                parties_involved=[],
                jurisdiction=None,
                summary=f"Processing delegated to background agent: {summary}",
                entities=[],
                relationships=[],
                sentiment=SentimentType.NEUTRAL,
                sentiment_score=0.0,
                key_concepts=[],
                action_items=[],
                deadlines=[],
                confidence_score=0.0
            )

            def _bg_worker():
                try:
                    result = self._process_document_internal(content, document_title, case_context)
                    # Ensure output directory exists and write result for retrieval
                    out_dir = os.path.join(os.getcwd(), 'output')
                    os.makedirs(out_dir, exist_ok=True)
                    safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", document_title)[:200]
                    out_path = os.path.join(out_dir, f"bg_result_{safe_title}.json")
                    with open(out_path, 'w', encoding='utf-8') as fh:
                        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)
                    logger.info(f"Background processing completed, wrote: {out_path}")
                except Exception as e:
                    logger.exception(f"Background processing failed for {document_title}: {e}")

            thread = threading.Thread(target=_bg_worker, daemon=True)
            thread.start()

            return placeholder

        # Default synchronous processing
        return self._process_document_internal(content, document_title, case_context)

    def _process_document_internal(
        self,
        content: str,
        document_title: str = "Untitled",
        case_context: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """
        Internal synchronous processing implementation extracted from process_document.
        """
        case_context = case_context or {}

        # Classify document type
        doc_type = self._classify_document_type(content)

        # Extract date
        extracted_date = self._extract_date(content)

        # Extract entities
        entities = self._extract_entities(content)

        # Extract parties
        parties = self._extract_parties(content, entities)

        # Generate summary
        summary = self._generate_summary(content)

        # Analyze sentiment
        sentiment, sentiment_score = self._analyze_sentiment(content)

        # Extract relationships
        relationships = self._extract_relationships(entities, content)

        # Extract key concepts
        key_concepts = self._extract_key_concepts(content)

        # Extract action items and deadlines
        action_items = self._extract_action_items(content)
        deadlines = self._extract_deadlines(content)

        # Determine jurisdiction
        jurisdiction = self._extract_jurisdiction(entities)

        # Create metadata
        metadata = DocumentMetadata(
            title=document_title,
            document_type=doc_type,
            extracted_date=extracted_date,
            parties_involved=parties,
            jurisdiction=jurisdiction,
            summary=summary,
            entities=entities,
            relationships=relationships,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            key_concepts=key_concepts,
            action_items=action_items,
            deadlines=deadlines,
            confidence_score=self._calculate_confidence(entities, relationships)
        )

        return metadata

    def _classify_document_type(self, content: str) -> DocumentType:
        """Classify the type of document"""
        for doc_type, patterns in self.DOCUMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return doc_type

        return DocumentType.UNKNOWN

    def _extract_date(self, content: str) -> Optional[str]:
        """Extract primary date from document"""
        date_pattern = (
            r'(?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},?\s+\d{4}|'
            r'\d{1,2}/\d{1,2}/\d{4}'
        )
        match = re.search(date_pattern, content)
        return match.group(0) if match else None

    def _extract_entities(self, content: str) -> List[EntityInfo]:
        """Extract named entities from content"""
        entities: List[EntityInfo] = []

        if self.transformers_available and self.ner_pipeline:
            try:
                # Use transformer NER
                ner_results = self.ner_pipeline(content[:512])
                for result in ner_results:
                    start_pos = int(result.get('start', 0))
                    end_pos = int(result.get('end', len(result['word'])))
                    confidence = float(result.get('score', 0.8))
                    entities.append(
                        EntityInfo(
                            text=result['word'],
                            entity_type=result['entity_group'],
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=confidence,
                            context=content[
                                max(0, start_pos - 50):
                                min(len(content), end_pos + 50)
                            ]
                        )
                    )
            except Exception as e:
                logger.warning(f"Transformer NER failed: {e}, using pattern matching")
                entities = self._extract_entities_patterns(content)
        else:
            entities = self._extract_entities_patterns(content)

        return entities

    def _extract_entities_patterns(self, content: str) -> List[EntityInfo]:
        """Extract entities using regex patterns"""
        entities: List[EntityInfo] = []

        for entity_type, patterns in self.LEGAL_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    entities.append(
                        EntityInfo(
                            text=match.group(0),
                            entity_type=entity_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=0.75,
                            context=content[
                                max(0, match.start() - 50):
                                min(len(content), match.end() + 50)
                            ]
                        )
                    )

        return entities

    def _extract_parties(self, content: str, entities: List[EntityInfo]) -> List[str]:
        """Extract involved parties (plaintiff, defendant, etc.)"""
        parties: List[str] = []

        # Look for explicit party mentions
        party_patterns = [
            r'Plaintiff[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Defendant[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(?:v\.|versus)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]

        for pattern in party_patterns:
            for match in re.finditer(pattern, content):
                parties.append(match.group(1))

        # Extract from entities
        person_entities = [e for e in entities if e.entity_type == 'PERSON']
        for entity in person_entities[:5]:  # Limit to top 5
            if entity.text not in parties:
                parties.append(entity.text)

        return list(set(parties))

    def _generate_summary(self, content: str) -> str:
        """Generate summary of document"""
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # Return first meaningful sentence or truncate
        for sentence in sentences:
            if len(sentence) > 50:
                return sentence[:200] + "..." if len(sentence) > 200 else sentence

        return content[:200] + "..." if len(content) > 200 else content

    def _analyze_sentiment(self, content: str) -> Tuple[SentimentType, float]:
        """Analyze document sentiment"""
        if not self.transformers_available or not self.sentiment_pipeline:
            return self._analyze_sentiment_basic(content)

        try:
            # Analyze sentiment of first 512 characters
            result = self.sentiment_pipeline(content[:512])[0]
            label = result['label']
            score = result['score']

            # Map to sentiment types
            sentiment_map = {
                'POSITIVE': SentimentType.POSITIVE,
                'NEGATIVE': SentimentType.NEGATIVE,
            }
            sentiment = sentiment_map.get(label, SentimentType.NEUTRAL)

            # Refine score
            if sentiment == SentimentType.POSITIVE and score > 0.9:
                sentiment = SentimentType.HIGHLY_POSITIVE
            elif sentiment == SentimentType.NEGATIVE and score > 0.9:
                sentiment = SentimentType.HIGHLY_NEGATIVE

            return sentiment, score

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return self._analyze_sentiment_basic(content)

    def _analyze_sentiment_basic(self, content: str) -> Tuple[SentimentType, float]:
        """Basic sentiment analysis using keyword matching"""
        content_lower = content.lower()

        positive_words = {'support', 'positive', 'good', 'excellent', 'compliance'}
        negative_words = {'violation', 'breach', 'non-compliance', 'failure', 'abuse'}

        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)

        if neg_count > pos_count:
            score = min(neg_count / 5.0, 1.0)
            if score > 0.7:
                return SentimentType.HIGHLY_NEGATIVE, score
            return SentimentType.NEGATIVE, score
        elif pos_count > neg_count:
            score = min(pos_count / 5.0, 1.0)
            if score > 0.7:
                return SentimentType.HIGHLY_POSITIVE, score
            return SentimentType.POSITIVE, score
        else:
            return SentimentType.NEUTRAL, 0.5

    def _extract_relationships(
        self,
        entities: List[EntityInfo],
        content: str
    ) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships: List[Relationship] = []

        # Simple relationship detection
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1:]:
                # Check if entities are close in text
                if abs(entity1.end_pos - entity2.start_pos) < 100:
                    relationship_type = self._determine_relationship_type(
                        entity1, entity2, content
                    )
                    if relationship_type:
                        relationships.append(
                            Relationship(
                                source_entity=entity1.text,
                                target_entity=entity2.text,
                                relationship_type=relationship_type,
                                confidence=0.7,
                                context=content[
                                    max(0, entity1.start_pos - 50):
                                    min(len(content), entity2.end_pos + 50)
                                ]
                            )
                        )

        return relationships

    def _determine_relationship_type(
        self,
        entity1: EntityInfo,
        entity2: EntityInfo,
        content: str
    ) -> Optional[str]:
        """Determine the type of relationship between entities"""
        context = content[entity1.start_pos:entity2.end_pos].lower()

        relationship_patterns = {
            'versus': [r'\s+v\.?\s+', r'\s+versus\s+'],
            'represents': [r'represents', r'attorney for', r'counsel'],
            'involves': [r'involves', r'regarding', r'concerning'],
            'dated': [r'dated', r'on', r'as of'],
        }

        for rel_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context):
                    return rel_type

        return None

    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key legal concepts"""
        concepts: List[str] = []

        legal_terms = {
            'custody', 'parental', 'visitation', 'support', 'modification',
            'enforcement', 'violation', 'order', 'judgment', 'appeal',
            'discovery', 'motion', 'affidavit', 'evidence', 'testimony',
            'witness', 'plaintiff', 'defendant', 'liability', 'damages'
        }

        content_lower = content.lower()
        for term in legal_terms:
            if term in content_lower:
                concepts.append(term)

        return concepts

    def _extract_action_items(self, content: str) -> List[str]:
        """Extract action items from document"""
        items: List[str] = []

        action_patterns = [
            r'(?:must|shall|will|should)\s+([^.!?]+)',
            r'(?:required|required to)\s+([^.!?]+)',
            r'(?:ordered to)\s+([^.!?]+)',
        ]

        for pattern in action_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                action = match.group(1).strip()
                if len(action) > 10:
                    items.append(action)

        return items[:5]  # Limit to top 5

    def _extract_deadlines(self, content: str) -> List[str]:
        """Extract deadlines and important dates"""
        deadlines: List[str] = []

        deadline_patterns = [
            r'(?:by|before|on or before)\s+(' +
            r'(?:January|February|March|April|May|June|July|August|September|' +
            r'October|November|December)\s+\d{1,2},?\s+\d{4})',
            r'within\s+(\d+\s+(?:days|hours|weeks|months))',
        ]

        for pattern in deadline_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                deadlines.append(match.group(1))

        return deadlines

    def _extract_jurisdiction(self, entities: List[EntityInfo]) -> Optional[str]:
        """Extract jurisdiction from entities"""
        location_entities = [e for e in entities if e.entity_type == 'LOCATION']
        if location_entities:
            return location_entities[0].text

        return None

    def _calculate_confidence(
        self,
        entities: List[EntityInfo],
        relationships: List[Relationship]
    ) -> float:
        """Calculate confidence in document processing"""
        confidence = 0.5

        if len(entities) > 5:
            confidence += 0.15

        if len(relationships) > 0:
            confidence += 0.15

        if all(e.confidence > 0.7 for e in entities):
            confidence += 0.2

        return min(confidence, 1.0)

    def batch_process_documents(
        self,
        documents: List[Tuple[str, str]],
        case_context: Optional[Dict[str, Any]] = None
    ) -> List[DocumentMetadata]:
        """Process multiple documents"""
        results = []
        for doc_id, content in documents:
            metadata = self.process_document(content, doc_id, case_context)
            results.append(metadata)

        return results

    def generate_summary_report(
        self,
        documents: List[DocumentMetadata]
    ) -> Dict[str, Any]:
        """Generate summary report of processed documents"""
        return {
            "total_documents": len(documents),
            "document_types": self._count_document_types(documents),
            "total_entities": sum(len(d.entities) for d in documents),
            "total_relationships": sum(len(d.relationships) for d in documents),
            "sentiment_distribution": self._get_sentiment_distribution(documents),
            "parties_involved": self._aggregate_parties(documents),
            "jurisdictions": self._aggregate_jurisdictions(documents),
            "key_deadlines": self._aggregate_deadlines(documents),
            "overall_confidence": sum(d.confidence_score for d in documents) / len(documents)
            if documents else 0
        }

    def _count_document_types(self, documents: List[DocumentMetadata]) -> Dict[str, int]:
        """Count document types"""
        counts = defaultdict(int)
        for doc in documents:
            counts[doc.document_type.value] += 1
        return dict(counts)

    def _get_sentiment_distribution(self, documents: List[DocumentMetadata]) -> Dict[str, int]:
        """Get sentiment distribution"""
        distribution = defaultdict(int)
        for doc in documents:
            distribution[doc.sentiment.value] += 1
        return dict(distribution)

    def _aggregate_parties(self, documents: List[DocumentMetadata]) -> List[str]:
        """Aggregate all parties across documents"""
        parties = set()
        for doc in documents:
            parties.update(doc.parties_involved)
        return list(parties)

    def _aggregate_jurisdictions(self, documents: List[DocumentMetadata]) -> List[str]:
        """Aggregate jurisdictions"""
        jurisdictions = set()
        for doc in documents:
            if doc.jurisdiction:
                jurisdictions.add(doc.jurisdiction)
        return list(jurisdictions)

    def _aggregate_deadlines(self, documents: List[DocumentMetadata]) -> List[str]:
        """Aggregate all deadlines"""
        deadlines = []
        for doc in documents:
            deadlines.extend(doc.deadlines)
        return deadlines


if __name__ == "__main__":
    # Example usage
    processor = NLPDocumentProcessor()

    sample_doc = """
    MOTION FOR MODIFICATION OF CUSTODY

    TO THE HONORABLE COURT:

    Plaintiff, John Smith, by and through his attorney, respectfully submits this Motion
    for Modification of Custody as follows:

    1. This Honorable Court has jurisdiction over this matter.
    2. The current custody order dated January 15, 2024, should be modified.
    3. There has been a material change in circumstances since the last order.

    WHEREFORE, Plaintiff respectfully requests this Court modify the custody arrangement.

    Dated: March 1, 2024
    """

    metadata = processor.process_document(sample_doc, "Motion_for_Modification")
    print(json.dumps(metadata.to_dict(), indent=2, default=str))
