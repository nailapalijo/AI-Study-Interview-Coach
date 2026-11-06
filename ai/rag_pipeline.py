"""Retrieval-Augmented Generation: chunk -> embed -> FAISS -> retrieve -> grounded answer."""
from __future__ import annotations

import logging

import numpy as np

import config
from ai import prompts
from database.db import Database
from utils.text_chunker import chunk_text

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = ("I couldn't find the answer to that in your uploaded documents, so I won't guess. "
                     "Try rephrasing, selecting different documents, or uploading a document that covers this topic.")


class VectorIndex:
    """Inner-product index over L2-normalised vectors (= cosine similarity).

    Uses FAISS (IndexFlatIP) when installed; otherwise an exact NumPy fallback with identical results.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[int] = []
        self._faiss = None
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(dim)
            self.backend = "faiss"
        except ImportError:
            self.backend = "numpy"

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(vectors) != len(ids):
            raise ValueError("Vector shape does not match index dimension / ids")
        if self._faiss is not None:
            self._faiss.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        self.ids.extend(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.ids:
            return []
        k = min(k, len(self.ids))
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            pairs = zip(idx[0], scores[0])
        else:
            sims = (self._matrix @ q[0])
            top = np.argsort(-sims)[:k]
            pairs = ((i, sims[i]) for i in top)
        return [(self.ids[int(i)], float(s)) for i, s in pairs if int(i) >= 0]


class RAGPipeline:
    def __init__(self, db: Database, embedder):
        self.db = db
        self.embedder = embedder
        self._cache: dict[tuple, tuple[VectorIndex, dict[int, dict]]] = {}

    # ---- indexing ----------------------------------------------------------
    def ingest(self, doc_id: int, text: str | None = None) -> int:
        """Chunk + embed + persist a document. Returns the number of chunks."""
        if text is None:
            doc = self.db.get_document(doc_id)
            if not doc:
                raise ValueError(f"Document {doc_id} not found")
            text = doc["text"]
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Document has no text to index.")
        vectors = self.embedder.embed(chunks)
        self.db.replace_chunks(doc_id, chunks, vectors)
        self.invalidate()
        return len(chunks)

    def invalidate(self) -> None:
        self._cache.clear()

    def _index_for(self, doc_ids: list[int] | None):
        key = tuple(sorted(doc_ids)) if doc_ids is not None else ("*",)
        if key in self._cache:
            return self._cache[key]
        rows = [r for r in self.db.get_chunks(doc_ids) if r["vector"] is not None]
        if not rows:
            return None
        dim = rows[0]["dim"]
        rows = [r for r in rows if r["dim"] == dim]
        index = VectorIndex(dim)
        index.add(np.stack([r["vector"] for r in rows]), [r["id"] for r in rows])
        self._cache[key] = (index, {r["id"]: r for r in rows})
        return self._cache[key]

    # ---- retrieval -------------------------------------------------------------
    def retrieve(self, query: str, doc_ids: list[int] | None = None, k: int | None = None) -> list[dict]:
        entry = self._index_for(doc_ids)
        if entry is None:
            return []
        index, rows = entry
        hits = index.search(self.embedder.embed_query(query), k or config.TOP_K)
        return [{"chunk_id": cid, "doc_id": rows[cid]["doc_id"], "doc_name": rows[cid]["doc_name"],
                 "chunk_index": rows[cid]["chunk_index"], "text": rows[cid]["text"], "score": round(score, 4)}
                for cid, score in hits]

    def sample_context(self, doc_ids: list[int], max_chars: int = 3000, topic: str | None = None) -> str:
        """Context for generation tasks (quizzes, personalisation): topic-based or evenly spread chunks."""
        if topic:
            parts = [h["text"] for h in self.retrieve(topic, doc_ids, k=6) if h["score"] >= config.RELEVANCE_THRESHOLD]
        else:
            rows = self.db.get_chunks(doc_ids, with_embeddings=False)
            step = max(1, len(rows) // 6)
            parts = [r["text"] for r in rows[::step][:6]]
        out, size = [], 0
        for p in parts:
            if size + len(p) > max_chars:
                break
            out.append(p)
            size += len(p)
        return "\n\n".join(out)

    # ---- answering ---------------------------------------------------------------
    def answer(self, question: str, llm, doc_ids: list[int] | None, history: list[dict] | None = None) -> dict:
        """Return {answer, sources, grounded}. Never calls the LLM when nothing relevant is retrieved."""
        hits = self.retrieve(question, doc_ids)
        relevant = [h for h in hits if h["score"] >= config.RELEVANCE_THRESHOLD]
        if not relevant:
            return {"answer": NOT_FOUND_MESSAGE, "sources": [], "grounded": False}
        messages = prompts.rag_messages(question, relevant, history or [])
        text = llm.generate(messages, max_new_tokens=400, temperature=0.2)
        if prompts.NOT_FOUND_TOKEN in text.upper().replace(" ", "_"):
            return {"answer": NOT_FOUND_MESSAGE, "sources": [], "grounded": False}
        return {"answer": text, "sources": relevant, "grounded": True}
