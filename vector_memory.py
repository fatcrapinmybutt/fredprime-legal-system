from __future__ import annotations

"""Vector store integration backed by FAISS/Chroma."""
from typing import Any, Dict, Optional

try:  # pragma: no cover - heavy optional deps
    import chromadb  # type: ignore[import]
    from chromadb.config import Settings  # type: ignore[import]
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    SentenceTransformer = None  # type: ignore


class VectorMemory:
    """Light wrapper around ChromaDB for semantic search."""

    def __init__(
        self, path: str = "LegalResults/vector_store", collection: str = "evidence"
    ) -> None:
        if chromadb is None or Settings is None or SentenceTransformer is None:
            raise ImportError(
                "chromadb and sentence_transformers are required for VectorMemory"
            )
        self.client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert_doc(self, doc_id: str, text: str, meta: Dict[str, Any]) -> None:
        if not text.strip():
            return
        embedding = self.model.encode([text], normalize_embeddings=True).tolist()
        self.col.upsert(
            ids=[doc_id],
            embeddings=embedding,
            metadatas=[meta],
            documents=[text[:2000]],
        )

    def query(self, query_text: str, k: int = 10) -> Dict[str, Any]:
        embedding = self.model.encode([query_text], normalize_embeddings=True).tolist()
        return self.col.query(query_embeddings=embedding, n_results=k)


_VM: Optional[VectorMemory] = None


def get_vm() -> VectorMemory:
    """Return a singleton instance of :class:`VectorMemory`."""
    global _VM
    if _VM is None:
        _VM = VectorMemory()
    return _VM
