import pytest
import vector_memory


def test_vector_memory_dependency_guard():
    if vector_memory.chromadb is None or vector_memory.SentenceTransformer is None:
        with pytest.raises(ImportError):
            vector_memory.VectorMemory()
    else:
        vm = vector_memory.get_vm()
        assert vm is vector_memory.get_vm()
