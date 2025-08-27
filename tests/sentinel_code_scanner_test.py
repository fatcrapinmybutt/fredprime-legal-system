from scripts.sentinel_code_scanner import scan_paths


def test_scan_paths(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("print('x')", encoding="utf-8")
    records = scan_paths([str(tmp_path)])
    assert any(r["path"] == str(sample.resolve()) for r in records)
