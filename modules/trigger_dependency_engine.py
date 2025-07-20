from typing import Dict, Iterable, List


def trigger_dependency_engine(modules: Iterable[str]) -> Dict[str, List[str]]:
    """Build execution dependencies based on module naming."""
    graph: Dict[str, List[str]] = {}
    for m in modules:
        if "filing" in m or "motion" in m:
            graph[m] = [
                "timeline_fusion_engine",
                "evidence_contradiction_matrix",
            ]
        elif "sanction" in m:
            graph[m] = ["canon_enforcer", "perjury_alert_scan"]
        elif "federal" in m:
            graph[m] = ["entity_shell_tracer", "notice_of_claim_generator"]
    return graph
