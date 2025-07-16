import os
def test_openai_key():
    assert os.getenv("OPENAI_KEY"), "Missing OPENAI_KEY"
def test_license():
    assert os.getenv("MBP_LICENSE", "").startswith("MBP-"), "Invalid MBP_LICENSE"
