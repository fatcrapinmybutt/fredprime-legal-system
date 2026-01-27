"""Evidence ⇄ Judiciary Correlation Engine (EJCE).

MI-only, evidence-locked, fail-closed analysis core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence


class Severity(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    critical = "Critical"


class DeviationCode(str, Enum):
    missing_findings = "D1"
    ignored_evidence = "D2"
    relied_without_admission = "D3"
    wrong_standard = "D4"
    benchbook_deviation = "D5"
    ignored_objection = "D6"
    procedural_shortcut = "D7"
    retaliatory_sequence = "D8"
    jurisdictional_excess = "D9"
    pattern_repeat = "D10"


@dataclass(frozen=True)
class EvidenceAtom:
    atom_id: str
    atom_type: str
    source_path: str
    record_status: str
    authority_pinpoint: Optional[str] = None
    fact_pinpoint: Optional[str] = None
    cited_authorities: Sequence[str] = field(default_factory=tuple)
    judicial_acknowledged: bool = False
    judicial_ruled_on: bool = False
    judicial_excluded: bool = False
    judicial_relied_upon: bool = False

    def validate(self) -> None:
        if not self.atom_id:
            raise ValueError("EvidenceAtom.atom_id is required")
        if not self.source_path:
            raise ValueError("EvidenceAtom.source_path is required")


@dataclass(frozen=True)
class JudicialActionNode:
    action_id: str
    judge_id: str
    case_id: str
    action_type: str
    date: datetime
    authority_cited: Sequence[str] = field(default_factory=tuple)
    evidence_referenced: Sequence[str] = field(default_factory=tuple)
    evidence_required: Sequence[str] = field(default_factory=tuple)
    findings_made: Sequence[str] = field(default_factory=tuple)
    findings_required: Sequence[str] = field(default_factory=tuple)
    silence_vectors: Sequence[str] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.action_id:
            raise ValueError("JudicialActionNode.action_id is required")
        if not self.judge_id:
            raise ValueError("JudicialActionNode.judge_id is required")
        if not self.case_id:
            raise ValueError("JudicialActionNode.case_id is required")


@dataclass(frozen=True)
class MDPNode:
    mdp_node_id: str
    vehicle: str
    triggering_condition: str
    required_steps: Sequence[str] = field(default_factory=tuple)
    required_findings: Sequence[str] = field(default_factory=tuple)
    evidence_classes_required: Sequence[str] = field(default_factory=tuple)
    prohibited_shortcuts: Sequence[str] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.mdp_node_id:
            raise ValueError("MDPNode.mdp_node_id is required")
        if not self.vehicle:
            raise ValueError("MDPNode.vehicle is required")


@dataclass(frozen=True)
class TrainingRule:
    training_rule_id: str
    source: str
    instruction: str
    presumption_level: str

    def weight(self) -> float:
        weights = {"low": 0.5, "medium": 1.0, "high": 2.0, "absolute": 3.0}
        return weights.get(self.presumption_level, 0.0)


@dataclass(frozen=True)
class DeviationEdge:
    deviation_id: str
    deviation_code: DeviationCode
    severity: Severity
    judicial_action_id: str
    mdp_node_id: str
    evidence_atom_id: Optional[str]
    authority_violated: str
    explanation: Optional[str] = None


@dataclass(frozen=True)
class EJCEConfig:
    criticality_min: float = 1.0
    pattern_min: float = 1.0
    harm_min: float = 1.0
    jurisdiction_min: float = 1.0
    integrity_min: float = 2.0


@dataclass(frozen=True)
class EJCEIndices:
    criticality: float
    pattern_confidence: float
    procedural_harm: float
    jurisdiction_risk: float
    integrity_risk: float


@dataclass(frozen=True)
class EJCEGates:
    remedies: Sequence[str]
    jtc_eligible: bool


@dataclass(frozen=True)
class EJCEAnalysis:
    deviations: Sequence[DeviationEdge]
    indices: EJCEIndices
    gates: EJCEGates
    omission_scores: Dict[str, float]


def _severity_weight(severity: Severity) -> float:
    return {
        Severity.low: 0.5,
        Severity.medium: 1.0,
        Severity.high: 2.0,
        Severity.critical: 3.0,
    }[severity]


def _authority_weight(authority: str) -> float:
    if authority.startswith("MCR") or authority.startswith("MCL"):
        return 1.5
    if authority.startswith("MRE"):
        return 1.3
    if authority.startswith("MJI") or authority.startswith("SCAO"):
        return 1.2
    return 1.0


def _score_omissions(action: JudicialActionNode) -> float:
    missing_findings = len(set(action.findings_required) - set(action.findings_made))
    ignored_evidence = max(0, len(action.evidence_required) - len(action.evidence_referenced))
    ignored_objections = action.silence_vectors.count("ignored_objection")
    benchbook_deviation = action.silence_vectors.count("benchbook_deviation") * 1.5
    reliance_without_admission = action.silence_vectors.count("reliance_without_admission")
    return (
        missing_findings
        + ignored_evidence
        + ignored_objections
        + benchbook_deviation
        + reliance_without_admission
    )


def _detect_missing_findings(action: JudicialActionNode, mdp: MDPNode) -> Iterable[DeviationEdge]:
    missing = set(mdp.required_findings) - set(action.findings_made)
    for finding in sorted(missing):
        yield DeviationEdge(
            deviation_id=f"DEV-{action.action_id}-{finding}",
            deviation_code=DeviationCode.missing_findings,
            severity=Severity.critical,
            judicial_action_id=action.action_id,
            mdp_node_id=mdp.mdp_node_id,
            evidence_atom_id=None,
            authority_violated=f"Missing finding required by {mdp.vehicle}: {finding}",
        )


def _detect_ignored_evidence(
    action: JudicialActionNode, mdp: MDPNode, evidence: Dict[str, EvidenceAtom]
) -> Iterable[DeviationEdge]:
    referenced = set(action.evidence_referenced)
    for atom_id, atom in evidence.items():
        if atom_id not in referenced:
            yield DeviationEdge(
                deviation_id=f"DEV-{action.action_id}-{atom_id}",
                deviation_code=DeviationCode.ignored_evidence,
                severity=Severity.high,
                judicial_action_id=action.action_id,
                mdp_node_id=mdp.mdp_node_id,
                evidence_atom_id=atom_id,
                authority_violated="Evidence in record omitted from action",
            )


def _compute_indices(
    deviations: Sequence[DeviationEdge],
    omission_scores: Dict[str, float],
    training_rules: Sequence[TrainingRule],
) -> EJCEIndices:
    criticality = sum(
        _severity_weight(dev.severity) * _authority_weight(dev.authority_violated)
        for dev in deviations
        if dev.severity in {Severity.high, Severity.critical}
    )
    pattern_confidence = len({dev.judicial_action_id for dev in deviations})
    procedural_harm = sum(omission_scores.values())
    jurisdiction_risk = sum(
        1.0
        for dev in deviations
        if dev.deviation_code == DeviationCode.jurisdictional_excess
    )
    integrity_risk = sum(rule.weight() for rule in training_rules)
    return EJCEIndices(
        criticality=criticality,
        pattern_confidence=pattern_confidence,
        procedural_harm=procedural_harm,
        jurisdiction_risk=jurisdiction_risk,
        integrity_risk=integrity_risk,
    )


def _apply_gates(indices: EJCEIndices, config: EJCEConfig, training_rules: Sequence[TrainingRule]) -> EJCEGates:
    remedies: List[str] = []
    if (
        indices.criticality >= config.criticality_min
        and indices.pattern_confidence >= config.pattern_min
        and indices.procedural_harm >= config.harm_min
        and (indices.jurisdiction_risk >= config.jurisdiction_min or indices.integrity_risk >= config.integrity_min)
    ):
        remedies.extend(["Reassignment", "Superintending_Control"])
    jtc_eligible = (
        any(rule.presumption_level == "absolute" for rule in training_rules)
        and indices.procedural_harm >= config.harm_min
    )
    return EJCEGates(remedies=remedies, jtc_eligible=jtc_eligible)


def run_ejce_analysis(
    evidence_atoms: Sequence[EvidenceAtom],
    judicial_actions: Sequence[JudicialActionNode],
    mdp_nodes: Sequence[MDPNode],
    training_rules: Sequence[TrainingRule],
    config: Optional[EJCEConfig] = None,
) -> EJCEAnalysis:
    config = config or EJCEConfig()
    evidence = {atom.atom_id: atom for atom in evidence_atoms}
    for atom in evidence_atoms:
        atom.validate()
    for action in judicial_actions:
        action.validate()
    for mdp in mdp_nodes:
        mdp.validate()

    deviations: List[DeviationEdge] = []
    omission_scores: Dict[str, float] = {}

    for action in judicial_actions:
        for mdp in mdp_nodes:
            deviations.extend(_detect_missing_findings(action, mdp))
            deviations.extend(_detect_ignored_evidence(action, mdp, evidence))
        omission_scores[action.action_id] = _score_omissions(action)

    indices = _compute_indices(deviations, omission_scores, training_rules)
    gates = _apply_gates(indices, config, training_rules)

    return EJCEAnalysis(
        deviations=deviations,
        indices=indices,
        gates=gates,
        omission_scores=omission_scores,
    )
