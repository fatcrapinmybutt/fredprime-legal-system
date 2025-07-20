from modules.upload_impact_analyzer import upload_impact_analyzer


def test_upload_impact_analyzer() -> None:
    uploads = [{"filename": "motion.pdf"}, {"filename": "timeline.txt"}]
    history = ["motion_generator", "timeline_builder"]
    links = upload_impact_analyzer(uploads, history)
    assert ("motion.pdf", "motion_generator") in links
    assert ("timeline.txt", "timeline_builder") in links
