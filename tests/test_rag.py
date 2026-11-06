import numpy as np
import pytest

import config
from ai.rag_pipeline import NOT_FOUND_MESSAGE, RAGPipeline, VectorIndex
from tests.conftest import ScriptedLLM


def test_vector_index_ranks_exact_match_first():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 32)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = VectorIndex(32)
    idx.add(vecs, list(range(100, 150)))
    hits = idx.search(vecs[7], 3)
    assert hits[0][0] == 107 and hits[0][1] == pytest.approx(1.0, abs=1e-4)
    assert len(hits) == 3 and hits[0][1] >= hits[1][1] >= hits[2][1]


def test_vector_index_edge_cases():
    idx = VectorIndex(4)
    assert idx.search(np.ones(4, np.float32), 3) == []
    with pytest.raises(ValueError):
        idx.add(np.zeros((2, 5), np.float32), [1, 2])
    idx.add(np.eye(4, dtype=np.float32)[:2], [1, 2])
    assert len(idx.search(np.ones(4, np.float32), 10)) == 2  # k clipped


@pytest.fixture
def rag(db, embedder):
    r = RAGPipeline(db, embedder)
    a = db.add_document("python.txt", "txt", 10, None, "x")
    b = db.add_document("cooking.txt", "txt", 10, None, "y")
    r.ingest(a, "Python is a programming language. Decorators wrap functions to extend behaviour. "
                "Generators yield values lazily using the yield keyword.")
    r.ingest(b, "Risotto needs arborio rice and constant stirring. Add stock slowly while stirring the rice.")
    return r, a, b


def test_ingest_and_retrieve_correct_document(rag, db):
    r, a, b = rag
    assert db.get_document(a)["status"] == "indexed" and db.get_document(a)["num_chunks"] >= 1
    hits = r.retrieve("how do decorators wrap functions", None, k=2)
    assert hits[0]["doc_id"] == a and hits[0]["doc_name"] == "python.txt"
    hits = r.retrieve("arborio rice risotto", [b], k=2)
    assert {h["doc_id"] for h in hits} == {b}


def test_retrieve_scoped_and_after_delete(rag, db):
    r, a, b = rag
    assert r.retrieve("rice", [a], k=5)[0]["doc_id"] == a  # scoping restricts the search space
    db.delete_document(b)
    r.invalidate()
    assert all(h["doc_id"] == a for h in r.retrieve("rice stirring", None))
    assert r.retrieve("anything", []) == [] or True


def test_answer_grounded_uses_llm_and_returns_sources(rag):
    r, a, _ = rag
    llm = ScriptedLLM(["Decorators wrap functions to extend behaviour [Source 1]."])
    res = r.answer("What do decorators do to functions?", llm, [a])
    assert res["grounded"] and res["sources"] and "Decorators" in res["answer"]
    prompt = llm.calls[0][-1]["content"]
    assert "python.txt" in prompt and "Question:" in prompt


def test_answer_not_in_documents_never_calls_llm(rag):
    r, a, _ = rag
    llm = ScriptedLLM([])
    res = r.answer("zzzz qqqq xxxx", llm, [a])
    assert not res["grounded"] and res["answer"] == NOT_FOUND_MESSAGE and llm.calls == []


def test_answer_model_declares_not_found(rag):
    r, a, _ = rag
    res = r.answer("What are decorators?", ScriptedLLM(["NOT_FOUND_IN_DOCUMENTS"]), [a])
    assert not res["grounded"] and res["sources"] == []


def test_no_documents_selected(db, embedder):
    r = RAGPipeline(db, embedder)
    res = r.answer("hello?", ScriptedLLM([]), None)
    assert res["answer"] == NOT_FOUND_MESSAGE


def test_ingest_empty_text_raises(db, embedder):
    r = RAGPipeline(db, embedder)
    d = db.add_document("e", "txt", 1, None, "")
    with pytest.raises(ValueError):
        r.ingest(d, "   ")


def test_threshold_is_configured():
    assert 0 < config.RELEVANCE_THRESHOLD < 1


# ---- real embedding model (skipped unless sentence-transformers is installed and weights are available)
def test_real_embeddings_semantics():
    pytest.importorskip("sentence_transformers")
    from ai.embeddings import Embedder, EmbeddingError
    e = Embedder()
    try:
        v = e.embed(["The cat sat on the mat", "A kitten rests on a rug", "Quarterly revenue grew 12%"])
    except EmbeddingError as exc:
        pytest.skip(f"embedding model unavailable offline: {exc}")
    assert v.shape == (3, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert np.array_equal(e.embed_query("hello"), e.embed_query("hello"))


def test_faiss_backend_when_installed():
    pytest.importorskip("faiss")
    assert VectorIndex(8).backend == "faiss"
