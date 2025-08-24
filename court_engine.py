"""Minimal checklist gatekeeper for motion generation."""

from __future__ import annotations

from typing import Any, Dict, List

CHECKLISTS: Dict[str, List[str]] = {
    "Motion to Set Aside / Stay Enforcement": [
        "final_order_or_judgment_identified",
        "grounds_under_MCR_2_612_specified",
        "supporting_facts_cited",
        "timeliness_explained",
    ],
    "Motion for Sanctions (MCR 1.109(E)/2.114)": [
        "challenged_paper_identified",
        "specific_rule_violation_stated",
        "safe_harbor_if_applicable_addressed",
        "requested_relief_stated",
    ],
    "Federal Complaint Draft (42 USC §1983/§1985 + IIED + Abuse of Process + Malicious Prosecution)": [
        "jurisdictional_basis_pleaded",
        "parties_properly_named",
        "facts_support_each_element",
        "prayer_for_relief_specific",
    ],
}


def requirements_met(
    motion_type: str, materials: Dict[str, Any], case_meta: Dict[str, Any]
) -> bool:
    """Heuristic check to avoid placeholder filings."""
    if not case_meta or not case_meta.get("court_name"):
        return False
    items = CHECKLISTS.get(motion_type, [])
    signals = 0
    mats = materials.get("materials", [])
    pool_text = " ".join(m.get("excerpt", "") for m in mats)
    for need in items:
        if any(
            key in pool_text.lower()
            for key in [
                "order",
                "judgment",
                "violation",
                "relief",
                "jurisdiction",
                "element",
                "facts",
                "timely",
                "time",
                "stay",
            ]
        ):
            signals += 1
    return signals >= max(2, len(items) // 2)
