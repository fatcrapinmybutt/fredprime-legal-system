"""Minimal Michigan court rules and references used by agent scaffolds.

This module provides a small, searchable set of commonly-referenced
Michigan Court Rules (MCR), Michigan Rules of Evidence (MRE), and links.
It's intentionally small and intended for scaffolding and offline testing.
Add more entries as needed.
"""

RULES = {
    "MCR 1.109": {
        "title": "Service of Process; Notice",
        "summary": "Rules governing service of process and notice requirements for civil actions in Michigan state courts.",
        "source": "https://courts.michigan.gov",
    },
    "MCR 2.101": {
        "title": "Commencing an Action",
        "summary": "Requirements and procedures to commence a civil action in Michigan circuit court.",
        "source": "https://courts.michigan.gov",
    },
    "MCR 2.104": {
        "title": "Time and Place of Filing",
        "summary": "Procedural rules about filing and docketing civil complaints and related documents.",
        "source": "https://courts.michigan.gov",
    },
    "MRE 401": {
        "title": "Definition of Relevant Evidence",
        "summary": "Evidence having any tendency to make a fact more or less probable than it would be without the evidence.",
        "source": "https://courts.michigan.gov",
    },
    "MRE 403": {
        "title": "Excluding Relevant Evidence for Prejudice, Confusion, Waste of Time, or Other Reasons",
        "summary": "Allows exclusion of relevant evidence when its probative value is substantially outweighed by a danger of unfair prejudice, confusion, or waste of time.",
        "source": "https://courts.michigan.gov",
    },
    "MCR 7.203": {
        "title": "Briefs and Appendices in Appellate Practice",
        "summary": "Standards and requirements for briefs and appendices in Michigan appellate procedure.",
        "source": "https://courts.michigan.gov",
    },
    "MCL 600.5805": {
        "title": "Actions to Recover Real Property; Statute of Limitations (example)",
        "summary": "Commonly-cited statute in property actions; include for reference and linking to statutory text when needed.",
        "source": "https://legislature.mi.gov",
    },
    "MCR 2.116": {
        "title": "Summary Disposition",
        "summary": "Procedure for bringing or responding to motions for summary disposition (analogous to summary judgment in some contexts).",
        "source": "https://courts.michigan.gov",
    },
    "MCR 2.602": {
        "title": "Service of Process by Publication",
        "summary": "Rules for substituted service and publication where personal service cannot be made.",
        "source": "https://courts.michigan.gov",
    },
    "MCR 3.203": {
        "title": "Child Custody and Parenting Time",
        "summary": "Procedural provisions for domestic relations pleadings and custody-related motions.",
        "source": "https://courts.michigan.gov",
    },
    "MRE 702": {
        "title": "Testimony by Expert Witnesses",
        "summary": "Standards for admissibility of expert testimony based on specialized knowledge, skill, experience, training, or education.",
        "source": "https://courts.michigan.gov",
    },
}


def search_rules(query: str):
    """Return matching rule keys for a simple substring search (case-insensitive)."""
    q = query.lower()
    results = []
    for key, info in RULES.items():
        if q in key.lower() or q in info.get("title", "").lower() or q in info.get("summary", "").lower():
            results.append((key, info))
    return results


def get_rule(key: str):
    """Return the rule dict for `key` or None if not found."""
    return RULES.get(key)
