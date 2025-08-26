import LAWFORGE_MASTER_UPGRADE_v1 as lm


def test_chunk_text_basic():
    data = "word " * 50
    chunks = lm.chunk_text(data, tokens=10, overlap=2)
    assert isinstance(chunks, list)
    assert chunks
