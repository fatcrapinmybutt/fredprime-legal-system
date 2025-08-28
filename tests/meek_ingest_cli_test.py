from meek_ingest_cli import ingest_directory, main


def test_ingest_directory_counts_files(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    assert ingest_directory(tmp_path) == 1


def test_main_prints_count(tmp_path, capsys):
    file_path = tmp_path / "another.txt"
    file_path.write_text("hi", encoding="utf-8")
    main([str(tmp_path)])
    captured = capsys.readouterr()
    assert "Ingested 1 files" in captured.out
