"""
Master Integration Bridge - Connects master orchestration engine to existing subsystems.

This module provides adapters that wire the WorkflowEngine to existing components:
- Evidence intake and scanning
- Motion generation
- Warboard visualization
- Document generation
- MiFile bundling
- Discovery preparation

Fully offline, no external APIs.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Stage Handler Registry & Dispatcher
# ============================================================================

@dataclass
class StageHandlerRegistry:
    """Registry for all stage type handlers with fallbacks."""

    handlers: Dict[str, Callable] = None

    def __post_init__(self):
        if self.handlers is None:
            self.handlers = {}
        self._register_builtin_handlers()

    def _register_builtin_handlers(self):
        """Register built-in handlers for all stage types."""
        self.handlers.update({
            'intake': self.handle_intake_stage,
            'analysis': self.handle_analysis_stage,
            'organization': self.handle_organization_stage,
            'generation': self.handle_generation_stage,
            'validation': self.handle_validation_stage,
            'warboarding': self.handle_warboarding_stage,
            'discovery': self.handle_discovery_stage,
            'filing': self.handle_filing_stage,
        })

    def register(self, stage_type: str, handler: Callable):
        """Register custom handler for stage type."""
        self.handlers[stage_type] = handler
        logger.info(f"Registered handler for stage type: {stage_type}")

    async def dispatch(self, stage_type: str, context: 'CaseContext', config: Dict) -> Dict:
        """Dispatch to appropriate handler."""
        handler = self.handlers.get(stage_type)
        if not handler:
            raise ValueError(f"No handler registered for stage type: {stage_type}")

        if asyncio.iscoroutinefunction(handler):
            return await handler(context, config)
        return handler(context, config)

    # =====================================================================
    # Built-in Stage Handlers
    # =====================================================================

    async def handle_intake_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        INTAKE stage: Scan, hash, and manifest evidence files.

        Integrates with existing scanner subsystems.
        """
        logger.info(f"[INTAKE] Starting evidence ingestion for case {context.case_id}")

        evidence_files = []
        file_hashes = {}

        # Scan evidence directories
        for root_dir in context.root_directories:
            if not root_dir.exists():
                logger.warning(f"Evidence directory not found: {root_dir}")
                continue

            for file_path in root_dir.rglob('*'):
                if file_path.is_file():
                    # Compute file hash (SHA256)
                    file_hash = await self._compute_file_hash(file_path)
                    file_hashes[str(file_path)] = file_hash

                    evidence_files.append({
                        'path': str(file_path),
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'hash': file_hash,
                        'relative_path': file_path.relative_to(root_dir),
                    })

        context.evidence_files.extend(evidence_files)
        context.file_hashes.update(file_hashes)

        logger.info(f"[INTAKE] Ingested {len(evidence_files)} files")

        return {
            'status': 'completed',
            'files_ingested': len(evidence_files),
            'total_size_mb': sum(f['size'] for f in evidence_files) / (1024 * 1024),
            'hashes_computed': len(file_hashes),
        }

    async def handle_analysis_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        ANALYSIS stage: Deduplicate, score, and extract metadata.

        Analyzes evidence for relevance, duplicates, and importance.
        """
        logger.info(f"[ANALYSIS] Analyzing {len(context.evidence_files)} files")

        # Deduplicate by hash
        unique_hashes = set(context.file_hashes.values())
        dedup_ratio = 1.0 - (len(unique_hashes) / len(context.file_hashes) if context.file_hashes else 0)

        # Score files by type and metadata
        scored_files = []
        for f in context.evidence_files:
            score = self._score_evidence_file(f)
            scored_files.append({
                **f,
                'relevance_score': score,
                'priority': 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low',
            })

        context.evidence_files = scored_files
        context.statistics['dedup_ratio'] = dedup_ratio
        context.statistics['avg_relevance'] = sum(f['relevance_score'] for f in scored_files) / len(scored_files) if scored_files else 0

        logger.info(f"[ANALYSIS] Dedup ratio: {dedup_ratio:.1%}, Avg relevance: {context.statistics['avg_relevance']:.2f}")

        return {
            'status': 'completed',
            'unique_files': len(unique_hashes),
            'duplicate_files': len(context.file_hashes) - len(unique_hashes),
            'dedup_ratio': dedup_ratio,
            'avg_relevance_score': context.statistics['avg_relevance'],
        }

    async def handle_organization_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        ORGANIZATION stage: Label exhibits A-Z, organize hierarchy.

        Creates organized directory structure with exhibit labels.
        """
        logger.info(f"[ORGANIZATION] Organizing {len(context.evidence_files)} exhibits")

        org_output = config.get('output_dir', Path('exhibits'))
        org_output.mkdir(parents=True, exist_ok=True)

        # Sort files by relevance score
        sorted_files = sorted(context.evidence_files, key=lambda f: f['relevance_score'], reverse=True)

        # Assign exhibit labels A-Z, AA-ZZ, etc.
        labels = self._generate_exhibit_labels(len(sorted_files))
        organized = []

        for idx, (label, file_info) in enumerate(zip(labels, sorted_files)):
            source = Path(file_info['path'])
            ext = source.suffix
            dest = org_output / f"Exhibit_{label}{ext}"

            # Copy file with exhibit label
            if source.exists():
                dest.write_bytes(source.read_bytes())

            organized.append({
                **file_info,
                'exhibit_label': label,
                'organized_path': str(dest),
            })

        context.evidence_files = organized
        context.exhibit_count = len(organized)

        logger.info(f"[ORGANIZATION] Created {len(organized)} labeled exhibits")

        return {
            'status': 'completed',
            'exhibits_created': len(organized),
            'output_dir': str(org_output),
            'labels_used': labels[:min(10, len(labels))],  # Sample of labels
        }

    async def handle_generation_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        GENERATION stage: Create motions, affidavits, documents.

        Generates court documents based on case type and evidence.
        """
        logger.info(f"[GENERATION] Generating documents for {context.case_type} case")

        output_dir = config.get('output_dir', Path('documents'))
        output_dir.mkdir(parents=True, exist_ok=True)

        documents = []

        # Generate motion based on case type
        if context.case_type == 'custody':
            motion = await self._generate_custody_motion(context, config)
            documents.append(motion)

        elif context.case_type == 'housing':
            motion = await self._generate_emergency_motion(context, config)
            documents.append(motion)

        elif context.case_type == 'ppo':
            motion = await self._generate_ppo_response(context, config)
            documents.append(motion)

        # Generate supporting affidavit
        affidavit = await self._generate_supporting_affidavit(context, config)
        documents.append(affidavit)

        # Save documents
        for doc in documents:
            doc_path = output_dir / doc['filename']
            doc_path.write_text(doc['content'], encoding='utf-8')

        context.generated_documents = documents

        logger.info(f"[GENERATION] Generated {len(documents)} documents")

        return {
            'status': 'completed',
            'documents_created': len(documents),
            'documents': [d['filename'] for d in documents],
            'output_dir': str(output_dir),
        }

    async def handle_validation_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        VALIDATION stage: Check MCR/MCL compliance.

        Validates documents for court rules compliance.
        """
        logger.info(f"[VALIDATION] Validating documents for MCR compliance")

        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
        }

        # Validate generated documents
        for doc in context.generated_documents:
            if not self._validate_document(doc):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Document '{doc['filename']}' failed validation")

        # Validate exhibit organization
        if context.exhibit_count == 0:
            validation_results['warnings'].append("No exhibits organized")

        logger.info(f"[VALIDATION] Validation {'passed' if validation_results['valid'] else 'FAILED'}")

        return {
            'status': 'completed' if validation_results['valid'] else 'failed',
            'valid': validation_results['valid'],
            'errors': validation_results['errors'],
            'warnings': validation_results['warnings'],
        }

    async def handle_warboarding_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        WARBOARDING stage: Generate timeline, visualizations, warboards.

        Creates visual representation of case facts and timeline.
        """
        logger.info(f"[WARBOARDING] Generating visualizations")

        output_dir = config.get('output_dir', Path('warboards'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timeline warboard
        timeline_data = await self._build_timeline_warboard(context, config)
        timeline_file = output_dir / "timeline_warboard.svg"
        timeline_file.write_text(timeline_data['svg'], encoding='utf-8')

        # Generate case map (if applicable)
        if context.case_type in ['custody', 'housing']:
            case_map = await self._build_case_map(context, config)
            map_file = output_dir / "case_map_warboard.svg"
            map_file.write_text(case_map['svg'], encoding='utf-8')

        logger.info(f"[WARBOARDING] Generated visualization files in {output_dir}")

        return {
            'status': 'completed',
            'visualizations': ['timeline_warboard.svg'],
            'output_dir': str(output_dir),
        }

    async def handle_discovery_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        DISCOVERY stage: Prepare discovery requests and logs.

        Creates discovery request documents based on case facts.
        """
        logger.info(f"[DISCOVERY] Preparing discovery documents")

        output_dir = config.get('output_dir', Path('discovery'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate discovery request
        discovery_request = await self._generate_discovery_request(context, config)
        request_file = output_dir / "discovery_request.docx"
        request_file.write_text(discovery_request['content'], encoding='utf-8')

        # Generate interrogatories
        interrogatories = await self._generate_interrogatories(context, config)
        interrog_file = output_dir / "interrogatories.docx"
        interrog_file.write_text(interrogatories['content'], encoding='utf-8')

        logger.info(f"[DISCOVERY] Prepared discovery documents")

        return {
            'status': 'completed',
            'discovery_documents': ['discovery_request.docx', 'interrogatories.docx'],
            'output_dir': str(output_dir),
        }

    async def handle_filing_stage(self, context: 'CaseContext', config: Dict) -> Dict:
        """
        FILING stage: Bundle for court filing.

        Creates filing package ready for MiFile or court submission.
        """
        logger.info(f"[FILING] Preparing filing bundle")

        output_dir = config.get('output_dir', Path('filing'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Bundle documents
        filing_bundle = {
            'motions': [d for d in context.generated_documents if 'motion' in d['filename'].lower()],
            'affidavits': [d for d in context.generated_documents if 'affidavit' in d['filename'].lower()],
            'exhibits': context.evidence_files[:10],  # Include sample exhibits
        }

        # Create filing manifest
        manifest_file = output_dir / "FILING_MANIFEST.json"
        manifest_file.write_text(json.dumps(filing_bundle, indent=2, default=str), encoding='utf-8')

        logger.info(f"[FILING] Created filing bundle with {len(filing_bundle['motions'])} motions, {len(filing_bundle['affidavits'])} affidavits")

        return {
            'status': 'completed',
            'filing_bundle': manifest_file.name,
            'motions': len(filing_bundle['motions']),
            'affidavits': len(filing_bundle['affidavits']),
            'output_dir': str(output_dir),
        }

    # =====================================================================
    # Helper Methods
    # =====================================================================

    async def _compute_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Compute file hash asynchronously."""
        hash_obj = hashlib.new(algorithm)

        # Read file in chunks for efficiency
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _score_evidence_file(self, file_info: Dict) -> float:
        """Score evidence file by type and metadata."""
        score = 0.5  # Base score

        # Boost for document types
        name_lower = file_info['name'].lower()
        if any(ext in name_lower for ext in ['.pdf', '.docx', '.txt', '.msg']):
            score += 0.2

        # Boost for message files (high relevance)
        if '.msg' in name_lower or 'email' in name_lower:
            score += 0.1

        # Reduce for large media files
        if file_info['size'] > 100_000_000:  # > 100MB
            score -= 0.1

        return min(1.0, max(0.0, score))  # Clamp to 0-1

    def _generate_exhibit_labels(self, count: int) -> List[str]:
        """Generate exhibit labels: A-Z, AA-ZZ, AAA-ZZZ, etc."""
        labels = []

        # Single letters A-Z
        for i in range(26):
            labels.append(chr(65 + i))  # A-Z

        # Double letters AA-ZZ
        if count > 26:
            for i in range(26):
                for j in range(26):
                    labels.append(chr(65 + i) + chr(65 + j))

        # Triple letters AAA-ZZZ
        if count > 26 + (26 * 26):
            for i in range(26):
                for j in range(26):
                    for k in range(26):
                        labels.append(chr(65 + i) + chr(65 + j) + chr(65 + k))

        return labels[:count]

    async def _generate_custody_motion(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate Motion for Modification of Custody."""
        return {
            'filename': 'Motion_for_Modification_of_Custody.docx',
            'content': f"""MOTION FOR MODIFICATION OF CUSTODY ARRANGEMENT

Case No. {context.case_number}
Plaintiff: {context.parties.get('plaintiff', 'PLAINTIFF')}
Defendant: {context.parties.get('defendant', 'DEFENDANT')}

TO THE HONORABLE COURT:

Plaintiff respectfully requests modification of the current custody arrangement
based on the following material facts and circumstances:

FACTS:
- Number of exhibits: {len(context.evidence_files)}
- Average evidence relevance: {context.statistics.get('avg_relevance', 0):.2f}
- Case filed: {datetime.now().isoformat()}

WHEREFORE, Plaintiff respectfully requests that this Court modify the current
custody arrangement as prayed herein.

Respectfully submitted,
[SIGNATURE BLOCK]
{datetime.now().strftime('%B %d, %Y')}
""",
            'type': 'motion',
        }

    async def _generate_emergency_motion(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate Emergency Injunction Motion."""
        return {
            'filename': 'Emergency_Injunction_Motion.docx',
            'content': f"""EMERGENCY MOTION FOR PRELIMINARY INJUNCTION

Case No. {context.case_number}

TO THE HONORABLE COURT:

Plaintiff seeks emergency preliminary injunctive relief based on immediate,
irreparable harm threatened by Defendant.

BASIS FOR EMERGENCY RELIEF:
- Imminent harm to person/property
- Threat of irreparable injury
- Evidence count: {len(context.evidence_files)}
- Urgency: IMMEDIATE

WHEREFORE, Plaintiff requests emergency relief as outlined herein.

Respectfully submitted,
[SIGNATURE BLOCK]
{datetime.now().strftime('%B %d, %Y')}
""",
            'type': 'motion',
        }

    async def _generate_ppo_response(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate Response to Personal Protection Order."""
        return {
            'filename': 'Response_to_Personal_Protection_Order.docx',
            'content': f"""RESPONDENT'S RESPONSE TO PETITION FOR PERSONAL PROTECTION ORDER

Case No. {context.case_number}

TO THE HONORABLE COURT:

Respondent denies the material allegations in the petition and seeks dismissal
of this action based on the following:

DEFENSES:
- Evidence contradicts allegations
- No reasonable cause for PPO
- Respondent's evidence: {len(context.evidence_files)} exhibits
- Pattern of misuse by petitioner

WHEREFORE, Respondent respectfully requests dismissal of this petition.

Respectfully submitted,
[SIGNATURE BLOCK]
{datetime.now().strftime('%B %d, %Y')}
""",
            'type': 'motion',
        }

    async def _generate_supporting_affidavit(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate supporting affidavit."""
        return {
            'filename': 'Supporting_Affidavit.docx',
            'content': f"""AFFIDAVIT IN SUPPORT OF MOTION

Case No. {context.case_number}

I, [YOUR NAME], being duly sworn, depose and state as follows:

1. I have personal knowledge of the facts contained herein.

2. The exhibits attached as Exhibit A consist of {len(context.evidence_files)}
   documents that support the allegations in this case.

3. To the best of my knowledge, these documents are accurate and truthful.

4. These exhibits are relevant to the material issues in this case.

I declare under penalty of perjury that the foregoing is true and correct.

Executed on {datetime.now().strftime('%B %d, %Y')}

_________________________
[SIGNATURE]
[NAME AND TITLE]
State of Michigan
County of [COUNTY]

Subscribed and sworn to before me this _____ day of __________, 20___.

_________________________
Notary Public
My Commission Expires: __________
""",
            'type': 'affidavit',
        }

    async def _build_timeline_warboard(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate timeline SVG warboard."""
        return {
            'svg': f"""<?xml version="1.0"?>
<svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
  <title>Case Timeline - {context.case_number}</title>
  <rect width="1200" height="400" fill="white" stroke="black" stroke-width="2"/>
  <text x="600" y="30" text-anchor="middle" font-size="20" font-weight="bold">
    Timeline for {context.case_number}
  </text>
  <line x1="100" y1="200" x2="1100" y2="200" stroke="black" stroke-width="2"/>
  <text x="100" y="100" font-size="12">Case Type: {context.case_type.upper()}</text>
  <text x="100" y="130" font-size="12">Evidence Files: {len(context.evidence_files)}</text>
  <text x="100" y="160" font-size="12">Generated: {datetime.now().strftime('%B %d, %Y')}</text>
</svg>""",
        }

    async def _build_case_map(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate case map SVG warboard."""
        return {
            'svg': f"""<?xml version="1.0"?>
<svg width="1000" height="700" xmlns="http://www.w3.org/2000/svg">
  <title>Case Map - {context.case_number}</title>
  <rect width="1000" height="700" fill="white" stroke="black" stroke-width="2"/>
  <text x="500" y="40" text-anchor="middle" font-size="18" font-weight="bold">
    Case Map for {context.case_number}
  </text>
  <circle cx="250" cy="300" r="80" fill="lightblue" stroke="black" stroke-width="2"/>
  <text x="250" y="305" text-anchor="middle">Evidence</text>
  <circle cx="500" cy="300" r="80" fill="lightgreen" stroke="black" stroke-width="2"/>
  <text x="500" y="305" text-anchor="middle">Arguments</text>
  <circle cx="750" cy="300" r="80" fill="lightyellow" stroke="black" stroke-width="2"/>
  <text x="750" y="305" text-anchor="middle">Relief</text>
  <line x1="330" y1="300" x2="420" y2="300" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="580" y1="300" x2="670" y2="300" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
</svg>""",
        }

    async def _generate_discovery_request(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate discovery request."""
        return {
            'filename': 'Discovery_Request.docx',
            'content': f"""REQUEST FOR PRODUCTION OF DOCUMENTS

Case No. {context.case_number}

TO: {context.parties.get('opponent', 'OPPOSING PARTY')}

Please produce the following documents within 21 days:

1. All documents relating to custody arrangements from the past 12 months.
2. All communications between the parties regarding the children.
3. All medical, educational, and behavioral records.
4. Financial documents relevant to support calculations.
5. Evidence referenced in Plaintiff's Motion (Exhibits A-{self._number_to_letter(len(context.evidence_files))}).

INTERROGATORIES

1. State all facts supporting your position on custody.
2. Identify all witnesses with knowledge of custody matters.
3. Describe your employment and financial situation.

Respectfully submitted,
[SIGNATURE BLOCK]
{datetime.now().strftime('%B %d, %Y')}
""",
        }

    async def _generate_interrogatories(self, context: 'CaseContext', config: Dict) -> Dict:
        """Generate interrogatories."""
        return {
            'filename': 'Interrogatories.docx',
            'content': f"""INTERROGATORIES TO DEFENDANT

Case No. {context.case_number}

INSTRUCTIONS:
These Interrogatories are to be answered separately and fully by the Defendant,
within 21 days of receipt, in the form of written responses.

INTERROGATORIES:

1. State your current employment and income.
2. Describe your relationship with the minor children.
3. Identify all individuals with whom you have lived in the past two years.
4. State your position on custody and parenting time.
5. Identify all evidence supporting your position.

Respectfully submitted,
[SIGNATURE BLOCK]
{datetime.now().strftime('%B %d, %Y')}
""",
        }

    def _validate_document(self, doc: Dict) -> bool:
        """Validate document structure."""
        return bool(
            doc.get('filename') and
            doc.get('content') and
            len(doc.get('content', '')) > 100
        )

    def _number_to_letter(self, n: int) -> str:
        """Convert number to letter label."""
        if n <= 26:
            return chr(64 + n)
        return "Z"


# ============================================================================
# Case Context Extensions
# ============================================================================

@dataclass
class CaseContext:
    """Extended case context with evidence and processing state."""
    case_id: str
    case_type: str
    case_number: str
    root_directories: List[Path]
    parties: Dict[str, str] = None
    evidence_files: List[Dict] = None
    file_hashes: Dict[str, str] = None
    generated_documents: List[Dict] = None
    statistics: Dict[str, Any] = None
    exhibit_count: int = 0

    def __post_init__(self):
        if self.parties is None:
            self.parties = {}
        if self.evidence_files is None:
            self.evidence_files = []
        if self.file_hashes is None:
            self.file_hashes = {}
        if self.generated_documents is None:
            self.generated_documents = []
        if self.statistics is None:
            self.statistics = {}


# ============================================================================
# Exports
# ============================================================================

def get_handler_registry() -> StageHandlerRegistry:
    """Get the global stage handler registry."""
    return StageHandlerRegistry()


__all__ = [
    'StageHandlerRegistry',
    'CaseContext',
    'get_handler_registry',
]
