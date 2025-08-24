from agents import run_analysis_agents


def test_run_analysis_agents_empty() -> None:
    assert run_analysis_agents("") == {
        "parties": [],
        "claims": [],
        "statutes": [],
        "court_rules": [],
        "timeline": [],
        "exhibits": [],
    }
