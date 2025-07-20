from modules.trigger_dependency_engine import trigger_dependency_engine


def test_trigger_dependency_engine() -> None:
    mods = ["filing_generator", "sanction_tracker", "federal_link"]
    result = trigger_dependency_engine(mods)
    assert result["filing_generator"] == [
        "timeline_fusion_engine",
        "evidence_contradiction_matrix",
    ]
    assert result["sanction_tracker"] == [
        "canon_enforcer",
        "perjury_alert_scan",
    ]
    assert result["federal_link"] == [
        "entity_shell_tracer",
        "notice_of_claim_generator",
    ]
