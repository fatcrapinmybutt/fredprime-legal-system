"""
Argument Reasoning Graph (ARG) Module
Creates structured representations of legal arguments and evidence relationships.
Enables reasoning about evidence strength, contradictions, and argument coherence.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ArgumentType(Enum):
    """Types of legal arguments"""
    CLAIM = "claim"  # Core claim or assertion
    EVIDENCE = "evidence"  # Supporting evidence
    REASONING = "reasoning"  # Logical reasoning
    ASSUMPTION = "assumption"  # Underlying assumptions
    PREMISE = "premise"  # Logical premise
    CONCLUSION = "conclusion"  # Logical conclusion
    COUNTER_ARGUMENT = "counter_argument"  # Counter-argument


class RelationType(Enum):
    """Types of relationships between argument elements"""
    SUPPORTS = "supports"  # Element supports another
    CONTRADICTS = "contradicts"  # Element contradicts another
    STRENGTHENS = "strengthens"  # Element strengthens argument
    WEAKENS = "weakens"  # Element weakens argument
    DEPENDS_ON = "depends_on"  # Element depends on another
    IMPLIES = "implies"  # Element logically implies another
    REBUTS = "rebuts"  # Element rebuts another argument


class ArgumentStrength(Enum):
    """Strength assessment of arguments"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


@dataclass
class ArgumentNode:
    """Node in the argument graph"""
    node_id: str
    text: str
    arg_type: ArgumentType
    source: str  # Which document/evidence this comes from
    confidence: float  # 0-1 confidence in this element
    supporting_evidence: List[str] = field(default_factory=list)  # IDs of supporting evidence
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        data = asdict(self)
        data['arg_type'] = self.arg_type.value
        return data


@dataclass
class ArgumentEdge:
    """Edge in the argument graph"""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float  # 0-1 strength of relationship
    reasoning: str = ""  # Explanation of relationship
    supporting_basis: List[str] = field(default_factory=list)  # Why this relationship exists
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        data = asdict(self)
        data['relation_type'] = self.relation_type.value
        return data


@dataclass
class ArgumentPath:
    """Path through the argument graph"""
    path_nodes: List[str]  # Sequence of node IDs
    path_edges: List[Tuple[str, str, RelationType]]  # Sequence of edges
    strength: float  # Overall path strength
    coherence: float  # Logical coherence
    total_score: float  # Combined score


@dataclass
class ArgumentAnalysis:
    """Complete argument analysis"""
    case_id: str
    case_title: str
    main_claim: Optional[ArgumentNode]
    supporting_arguments: List[ArgumentNode]
    counter_arguments: List[ArgumentNode]
    key_evidence: List[ArgumentNode]
    argument_paths: List[ArgumentPath]
    vulnerabilities: List[Dict[str, Any]]
    strengths: List[Dict[str, Any]]
    recommendations: List[str]
    overall_strength: ArgumentStrength
    overall_score: float  # 0-1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        data = asdict(self)
        data['main_claim'] = self.main_claim.to_dict() if self.main_claim else None
        data['supporting_arguments'] = [a.to_dict() for a in self.supporting_arguments]
        data['counter_arguments'] = [a.to_dict() for a in self.counter_arguments]
        data['key_evidence'] = [e.to_dict() for e in self.key_evidence]
        data['overall_strength'] = self.overall_strength.value
        return data


class ArgumentReasoningGraph:
    """
    Argument Reasoning Graph system for legal analysis.
    Creates structured representations of arguments and evidence relationships.
    """

    def __init__(self):
        """Initialize ARG system"""
        self.nodes: Dict[str, ArgumentNode] = {}
        self.edges: Dict[Tuple[str, str], ArgumentEdge] = {}
        self.node_counter = 0

    def create_node(
        self,
        text: str,
        arg_type: ArgumentType,
        source: str,
        confidence: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArgumentNode:
        """Create a new argument node"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        node = ArgumentNode(
            node_id=node_id,
            text=text,
            arg_type=arg_type,
            source=source,
            confidence=confidence,
            metadata=metadata or {}
        )

        self.nodes[node_id] = node
        return node

    def create_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float = 0.7,
        reasoning: str = "",
        supporting_basis: Optional[List[str]] = None
    ) -> ArgumentEdge:
        """Create a new argument edge"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("One or both nodes don't exist")

        edge = ArgumentEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            reasoning=reasoning,
            supporting_basis=supporting_basis or []
        )

        self.edges[(source_id, target_id)] = edge
        return edge

    def analyze_case(
        self,
        case_id: str,
        case_title: str,
        evidence_nodes: List[ArgumentNode],
        argument_nodes: List[ArgumentNode],
        edges: List[ArgumentEdge]
    ) -> ArgumentAnalysis:
        """Perform comprehensive argument analysis"""

        # Add all nodes to graph
        all_nodes = evidence_nodes + argument_nodes
        for node in all_nodes:
            self.nodes[node.node_id] = node

        # Add all edges
        for edge in edges:
            self.edges[(edge.source_id, edge.target_id)] = edge

        # Identify main claim
        main_claim = self._identify_main_claim(argument_nodes)

        # Identify supporting and counter arguments
        supporting = self._identify_supporting_arguments(main_claim, argument_nodes, edges)
        counter = self._identify_counter_arguments(supporting, argument_nodes, edges)

        # Identify key evidence
        key_evidence = self._identify_key_evidence(
            evidence_nodes, main_claim, supporting
        )

        # Find strong argument paths
        argument_paths = self._find_strong_paths(main_claim, evidence_nodes, edges)

        # Identify vulnerabilities
        vulnerabilities = self._identify_vulnerabilities(
            main_claim, supporting, counter, edges
        )

        # Identify strengths
        strengths = self._identify_strengths(
            main_claim, supporting, key_evidence, edges
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            vulnerabilities, strengths, counter
        )

        # Calculate overall strength
        overall_strength, overall_score = self._calculate_overall_strength(
            main_claim, supporting, key_evidence, vulnerabilities
        )

        return ArgumentAnalysis(
            case_id=case_id,
            case_title=case_title,
            main_claim=main_claim,
            supporting_arguments=supporting,
            counter_arguments=counter,
            key_evidence=key_evidence,
            argument_paths=argument_paths,
            vulnerabilities=vulnerabilities,
            strengths=strengths,
            recommendations=recommendations,
            overall_strength=overall_strength,
            overall_score=overall_score
        )

    def _identify_main_claim(self, argument_nodes: List[ArgumentNode]) -> Optional[ArgumentNode]:
        """Identify the main claim from arguments"""
        # Look for claims or conclusions
        for node in argument_nodes:
            if node.arg_type in [ArgumentType.CLAIM, ArgumentType.CONCLUSION]:
                return node if node.confidence > 0.7 else argument_nodes[0] if argument_nodes else None

        return argument_nodes[0] if argument_nodes else None

    def _identify_supporting_arguments(
        self,
        main_claim: Optional[ArgumentNode],
        argument_nodes: List[ArgumentNode],
        edges: List[ArgumentEdge]
    ) -> List[ArgumentNode]:
        """Identify arguments that support the main claim"""
        supporting = []

        if not main_claim:
            return supporting

        for edge in edges:
            if edge.target_id == main_claim.node_id:
                if edge.relation_type == RelationType.SUPPORTS:
                    node = self.nodes.get(edge.source_id)
                    if node and node.arg_type in [
                        ArgumentType.EVIDENCE,
                        ArgumentType.REASONING,
                        ArgumentType.PREMISE
                    ]:
                        supporting.append(node)

        return supporting

    def _identify_counter_arguments(
        self,
        supporting_args: List[ArgumentNode],
        argument_nodes: List[ArgumentNode],
        edges: List[ArgumentEdge]
    ) -> List[ArgumentNode]:
        """Identify counter-arguments"""
        counter = []

        for node in argument_nodes:
            if node.arg_type == ArgumentType.COUNTER_ARGUMENT:
                counter.append(node)

        return counter

    def _identify_key_evidence(
        self,
        evidence_nodes: List[ArgumentNode],
        main_claim: Optional[ArgumentNode],
        supporting_args: List[ArgumentNode]
    ) -> List[ArgumentNode]:
        """Identify key evidence"""
        key_evidence = []

        # High confidence evidence
        high_confidence = [e for e in evidence_nodes if e.confidence > 0.8]
        key_evidence.extend(high_confidence[:5])

        return key_evidence

    def _find_strong_paths(
        self,
        main_claim: Optional[ArgumentNode],
        evidence_nodes: List[ArgumentNode],
        edges: List[ArgumentEdge]
    ) -> List[ArgumentPath]:
        """Find strong argument paths from evidence to claim"""
        paths = []

        if not main_claim:
            return paths

        # BFS to find paths
        for start_node in evidence_nodes:
            path_nodes = self._find_paths_bfs(
                start_node.node_id,
                main_claim.node_id,
                edges
            )

            for nodes in path_nodes:
                strength = self._calculate_path_strength(nodes, edges)
                coherence = self._calculate_path_coherence(nodes, edges)

                if strength > 0.6:
                    path_edges = self._extract_path_edges(nodes)
                    paths.append(
                        ArgumentPath(
                            path_nodes=nodes,
                            path_edges=path_edges,
                            strength=strength,
                            coherence=coherence,
                            total_score=strength * 0.6 + coherence * 0.4
                        )
                    )

        return sorted(paths, key=lambda p: p.total_score, reverse=True)[:5]

    def _find_paths_bfs(
        self,
        start_id: str,
        end_id: str,
        edges: List[ArgumentEdge],
        max_depth: int = 5
    ) -> List[List[str]]:
        """Find paths using BFS"""
        paths = []
        queue = deque([(start_id, [start_id], 0)])

        # Build adjacency list
        graph = defaultdict(list)
        for edge in edges:
            graph[edge.source_id].append(edge.target_id)

        while queue:
            node, path, depth = queue.popleft()

            if depth > max_depth:
                continue

            if node == end_id:
                paths.append(path)
                continue

            for neighbor in graph[node]:
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor], depth + 1))

        return paths

    def _extract_path_edges(self, nodes: List[str]) -> List[Tuple[str, str, RelationType]]:
        """Extract edges from a path"""
        edges = []
        for i in range(len(nodes) - 1):
            edge_key = (nodes[i], nodes[i + 1])
            if edge_key in self.edges:
                edge = self.edges[edge_key]
                edges.append((nodes[i], nodes[i + 1], edge.relation_type))

        return edges

    def _calculate_path_strength(self, nodes: List[str], edges: List[ArgumentEdge]) -> float:
        """Calculate strength of a path"""
        if not nodes:
            return 0.0

        strength_sum = 0.0
        count = 0

        for i in range(len(nodes) - 1):
            edge_key = (nodes[i], nodes[i + 1])
            if edge_key in self.edges:
                edge = self.edges[edge_key]
                # Support relationships strengthen path
                if edge.relation_type == RelationType.SUPPORTS:
                    strength_sum += edge.strength
                # Contradict relationships weaken path
                elif edge.relation_type == RelationType.CONTRADICTS:
                    strength_sum -= edge.strength * 0.5
                else:
                    strength_sum += edge.strength * 0.7

                count += 1

        return (strength_sum / count) if count > 0 else 0.5

    def _calculate_path_coherence(self, nodes: List[str], edges: List[ArgumentEdge]) -> float:
        """Calculate logical coherence of a path"""
        # More edges = higher coherence (well-connected argument)
        coherence = len(nodes) / 10.0  # Normalize by reasonable path length
        return min(coherence, 1.0)

    def _identify_vulnerabilities(
        self,
        main_claim: Optional[ArgumentNode],
        supporting: List[ArgumentNode],
        counter: List[ArgumentNode],
        edges: List[ArgumentEdge]
    ) -> List[Dict[str, Any]]:
        """Identify vulnerabilities in the argument"""
        vulnerabilities = []

        # Missing evidence
        if len(supporting) < 3:
            vulnerabilities.append({
                "type": "weak_support",
                "severity": "high",
                "description": "Main claim has limited supporting evidence",
                "recommendation": "Gather additional supporting evidence"
            })

        # Strong counter-arguments
        for counter_arg in counter:
            if counter_arg.confidence > 0.8:
                vulnerabilities.append({
                    "type": "strong_counter_argument",
                    "severity": "high",
                    "description": f"Strong counter-argument: {counter_arg.text[:100]}",
                    "recommendation": "Prepare rebuttal or concessions"
                })

        # Weak evidence
        weak_evidence = [s for s in supporting if s.confidence < 0.6]
        if weak_evidence:
            vulnerabilities.append({
                "type": "weak_evidence",
                "severity": "medium",
                "description": f"{len(weak_evidence)} supporting arguments have low confidence",
                "recommendation": "Strengthen or replace weak evidence"
            })

        return vulnerabilities

    def _identify_strengths(
        self,
        main_claim: Optional[ArgumentNode],
        supporting: List[ArgumentNode],
        key_evidence: List[ArgumentNode],
        edges: List[ArgumentEdge]
    ) -> List[Dict[str, Any]]:
        """Identify strengths in the argument"""
        strengths = []

        # Strong main claim
        if main_claim and main_claim.confidence > 0.8:
            strengths.append({
                "type": "strong_main_claim",
                "value": main_claim.confidence,
                "description": "Main claim is well-supported and clearly stated"
            })

        # Multiple supporting arguments
        if len(supporting) >= 3:
            strengths.append({
                "type": "multiple_supporting_arguments",
                "value": len(supporting),
                "description": f"Strong support from {len(supporting)} arguments"
            })

        # High-confidence evidence
        high_conf_evidence = [e for e in key_evidence if e.confidence > 0.85]
        if high_conf_evidence:
            strengths.append({
                "type": "high_confidence_evidence",
                "value": len(high_conf_evidence),
                "description": f"{len(high_conf_evidence)} highly credible evidence items"
            })

        return strengths

    def _generate_recommendations(
        self,
        vulnerabilities: List[Dict[str, Any]],
        strengths: List[Dict[str, Any]],
        counter_arguments: List[ArgumentNode]
    ) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []

        # Address high-severity vulnerabilities
        for vuln in vulnerabilities:
            if vuln.get("severity") == "high":
                recommendations.append(f"Priority: {vuln.get('recommendation')}")

        # Leverage strengths
        if len(strengths) >= 2:
            recommendations.append("Lead with strongest evidence and arguments")

        # Preempt counter-arguments
        if counter_arguments:
            recommendations.append("Anticipate and prepare rebuttals for counter-arguments")

        # Build narrative
        recommendations.append("Create coherent narrative linking evidence to main claim")

        return recommendations

    def _calculate_overall_strength(
        self,
        main_claim: Optional[ArgumentNode],
        supporting: List[ArgumentNode],
        key_evidence: List[ArgumentNode],
        vulnerabilities: List[Dict[str, Any]]
    ) -> Tuple[ArgumentStrength, float]:
        """Calculate overall argument strength"""
        score = 0.5

        # Weight main claim
        if main_claim:
            score += main_claim.confidence * 0.2

        # Weight supporting arguments
        if supporting:
            avg_support = sum(s.confidence for s in supporting) / len(supporting)
            score += avg_support * 0.3
            # Bonus for multiple supports
            score += min(len(supporting) / 5, 0.1)

        # Weight evidence
        if key_evidence:
            avg_evidence = sum(e.confidence for e in key_evidence) / len(key_evidence)
            score += avg_evidence * 0.3

        # Penalty for vulnerabilities
        high_severity_count = len([v for v in vulnerabilities if v.get("severity") == "high"])
        score -= high_severity_count * 0.1

        score = max(0.0, min(score, 1.0))

        # Determine strength level
        if score >= 0.85:
            strength = ArgumentStrength.VERY_STRONG
        elif score >= 0.7:
            strength = ArgumentStrength.STRONG
        elif score >= 0.55:
            strength = ArgumentStrength.MODERATE
        elif score >= 0.4:
            strength = ArgumentStrength.WEAK
        else:
            strength = ArgumentStrength.VERY_WEAK

        return strength, score

    def export_analysis(
        self,
        analysis: ArgumentAnalysis,
        format: str = "json"
    ) -> str:
        """Export analysis in various formats"""
        if format == "json":
            return json.dumps(analysis.to_dict(), indent=2, default=str)

        elif format == "text":
            text = f"""
            ARGUMENT ANALYSIS REPORT
            ========================
            Case: {analysis.case_title}
            Date: {analysis.created_at}
            Overall Strength: {analysis.overall_strength.value} ({analysis.overall_score:.2%})

            MAIN CLAIM
            ----------
            {analysis.main_claim.text if analysis.main_claim else 'No main claim identified'}

            SUPPORTING ARGUMENTS ({len(analysis.supporting_arguments)})
            --------------------
            """
            for arg in analysis.supporting_arguments:
                text += f"\n- {arg.text} (Confidence: {arg.confidence:.0%})"

            text += f"\n\nVULNERABILITIES ({len(analysis.vulnerabilities)})\n"
            text += "-------------------\n"
            for vuln in analysis.vulnerabilities:
                text += f"\n- [{vuln.get('severity', 'unknown').upper()}] {vuln.get('description', '')}"

            text += f"\n\nRECOMMENDATIONS\n"
            text += "---------------\n"
            for rec in analysis.recommendations:
                text += f"\n- {rec}"

            return text

        else:
            raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # Example usage
    arg_system = ArgumentReasoningGraph()

    # Create nodes
    main_claim = arg_system.create_node(
        "Defendant violated custody order",
        ArgumentType.CLAIM,
        "case_facts",
        confidence=0.85
    )

    evidence1 = arg_system.create_node(
        "Witness testimony showing unauthorized visitation",
        ArgumentType.EVIDENCE,
        "witness_statement",
        confidence=0.9
    )

    evidence2 = arg_system.create_node(
        "Documentation of scheduled visitation times",
        ArgumentType.EVIDENCE,
        "court_order",
        confidence=0.95
    )

    reasoning = arg_system.create_node(
        "Pattern of non-compliance with court order",
        ArgumentType.REASONING,
        "analysis",
        confidence=0.8
    )

    # Create edges
    arg_system.create_edge(
        evidence1.node_id,
        main_claim.node_id,
        RelationType.SUPPORTS,
        strength=0.85,
        reasoning="Testimony establishes violation"
    )

    arg_system.create_edge(
        evidence2.node_id,
        reasoning.node_id,
        RelationType.SUPPORTS,
        strength=0.9,
        reasoning="Documentation proves scheduled times"
    )

    # Analyze
    analysis = arg_system.analyze_case(
        case_id="case_001",
        case_title="Custody Violation",
        evidence_nodes=[evidence1, evidence2],
        argument_nodes=[main_claim, reasoning],
        edges=list(arg_system.edges.values())
    )

    print(arg_system.export_analysis(analysis, "text"))
