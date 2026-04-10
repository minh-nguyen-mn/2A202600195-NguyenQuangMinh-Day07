from __future__ import annotations

from typing import Any, Callable

# from .chunking import _dot
from .chunking import compute_similarity, FixedSizeChunker
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0
        # self._chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        # self._chunker = FixedSizeChunker(chunk_size=300, overlap=30)
        self._chunker = FixedSizeChunker(chunk_size=800, overlap=100)
        
        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
            self._collection = None
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        # raise NotImplementedError("Implement EmbeddingStore._make_record")
        chunks = self._chunker.chunk(doc.content)

        records = []

        for i, chunk in enumerate(chunks):
            print(f"   🔎 Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            embedding = self._embedding_fn(chunk)

            records.append({
                "id": f"{doc.id}_{self._next_index}_{i}",
                "doc_id": doc.id,
                "content": chunk,
                "embedding": embedding,
                "metadata": doc.metadata or {},
            })

        self._next_index += 1
        return records

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        # raise NotImplementedError("Implement EmbeddingStore._search_records")
        q_emb = self._embedding_fn(query)

        scored_results = []

        for r in records:
            # score = _dot(q_emb, r["embedding"])
            score = compute_similarity(q_emb, r["embedding"])
            scored_results.append({
                "id": r["id"],
                "doc_id": r["doc_id"],
                "content": r["content"],
                "embedding": r["embedding"],
                "metadata": r.get("metadata", {}),
                "score": score
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]
    
    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        # raise NotImplementedError("Implement EmbeddingStore.add_documents")
        total_chunks = 0

        for doc in docs:
            print(f"\n📄 Processing document: {doc.id}")

            records = self._make_record(doc)

            print(f"   → {len(records)} chunks")

            self._store.extend(records)
            total_chunks += len(records)

            print(f"   → Stored. Total chunks so far: {len(self._store)}")

        print(f"\n✅ DONE: {len(docs)} docs → {total_chunks} chunks")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        # raise NotImplementedError("Implement EmbeddingStore.search")
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        # raise NotImplementedError("Implement EmbeddingStore.get_collection_size")
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        # raise NotImplementedError("Implement EmbeddingStore.search_with_filter")
        if not metadata_filter:
            return self.search(query, top_k)

        filtered = []
        for r in self._store:
            ok = True
            for k, v in metadata_filter.items():
                if r.get("metadata", {}).get(k) != v:
                    ok = False
                    break
            if ok:
                filtered.append(r)

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        # raise NotImplementedError("Implement EmbeddingStore.delete_document")
        before = len(self._store)
        self._store = [r for r in self._store if r["doc_id"] != doc_id]
        return len(self._store) != before