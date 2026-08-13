import uuid
from types import SimpleNamespace

import pytest

from database.structural_db import QdrantVectorStore


class FakeLLM:
    def __init__(self):
        self.embed_calls = []
        self.embed_many_calls = []

    def embed(self, text):
        self.embed_calls.append(text)
        return self._embedding(text)

    @staticmethod
    def _embedding(text):
        return [
            float("lora" in text.lower()),
            float("quant" in text.lower()),
            1.0,
        ]

    def embed_many(self, texts):
        self.embed_many_calls.append(list(texts))
        return [self._embedding(text) for text in texts]

    def rerank_candidate_questions(self, query, candidates):
        return list(dict.fromkeys(candidate["parent_id"] for candidate in candidates))


def make_store(embedding_dim=3, token=None):
    token = token or uuid.uuid4().hex
    settings = SimpleNamespace(
        QDRANT_URL="http://localhost:6333",
        OPENAI_EMBEDDING_DIM=embedding_dim,
    )
    return QdrantVectorStore(
        settings,
        FakeLLM(),
        curriculum_collection=f"test_curriculum_{token}",
        questions_collection=f"test_questions_{token}",
    )


def delete_store(store):
    for collection in (store.curriculum_collection, store.questions_collection):
        store.client.delete_collection(collection)


def question_point(parent_id, question=None, knowledge=None):
    return SimpleNamespace(
        payload={
            "page_content": question or f"question {parent_id}",
            "parent_id": parent_id,
            "key_knowledge": knowledge or f"knowledge {parent_id}",
        }
    )


def stub_question_retrieval(monkeypatch, store, points, reranker_output):
    query_calls = []
    rerank_calls = []

    def query_points(**kwargs):
        query_calls.append(kwargs)
        return SimpleNamespace(points=points)

    def rerank(query, candidates):
        rerank_calls.append((query, candidates))
        return reranker_output

    monkeypatch.setattr(store.client, "query_points", query_points)
    monkeypatch.setattr(store.llm, "rerank_candidate_questions", rerank)
    monkeypatch.setattr(
        store,
        "_fetch_parent",
        lambda parent_id: {
            "page_content": parent_id,
            "metadata": {"parent_id": parent_id, "source": "paper.pdf"},
        },
    )
    return query_calls, rerank_calls


def test_question_retrieval_resolves_full_parent_section():
    store = make_store()
    try:
        parent_id = "a" * 32
        metadata = {
            "source": "lora.md",
            "section": "Method",
            "seq_id": 1,
            "page_start": 3,
            "page_end": 4,
            "anchor_nodes": ["LoRA"],
        }
        store.upsert_section("LoRA freezes the base model weights.", metadata, parent_id)
        store.upsert_questions(
            [
                {
                    "question": "Which LoRA weights stay frozen?",
                    "key_knowledge": "The base model weights remain frozen.",
                }
            ],
            parent_id,
            "lora.md",
        )

        sections = store.search_candidates_and_fetch_parent(
            "What stays frozen in LoRA?", store.llm, "lora.md"
        )

        assert store.section_exists(parent_id)
        assert len(sections) == 1
        assert sections[0]["page_content"] == "LoRA freezes the base model weights."
        assert sections[0]["metadata"]["matched_knowledge"] == (
            "The base model weights remain frozen."
        )
    finally:
        delete_store(store)


def test_question_retrieval_builds_five_unique_candidates_from_25_hits(monkeypatch):
    store = make_store()
    try:
        parent_ids = [f"parent-{index}" for index in range(1, 7)]
        points = [
            question_point(parent_ids[0], "first question", "first knowledge"),
            question_point(parent_ids[0], "duplicate question", "duplicate knowledge"),
            question_point(""),
            question_point("   "),
            *(question_point(parent_id) for parent_id in parent_ids[1:]),
        ]
        query_calls, rerank_calls = stub_question_retrieval(
            monkeypatch, store, points, []
        )

        sections = store.search_candidates_and_fetch_parent(
            "How does it work?", store.llm, "paper.pdf"
        )

        assert query_calls[0]["limit"] == 25
        conditions = query_calls[0]["query_filter"].must
        assert [(condition.key, condition.match.value) for condition in conditions] == [
            ("type", "question"),
            ("source", "paper.pdf"),
        ]
        candidates = rerank_calls[0][1]
        assert [candidate["parent_id"] for candidate in candidates] == parent_ids[:5]
        assert candidates[0] == {
            "question": "first question",
            "parent_id": parent_ids[0],
            "key_knowledge": "first knowledge",
        }
        assert [section["metadata"]["parent_id"] for section in sections] == parent_ids[:2]
    finally:
        delete_store(store)


def test_question_retrieval_fuses_reranker_first_with_vector_top(monkeypatch):
    store = make_store()
    try:
        precomputed_vector = [0.25, 0.5, 1.0]
        parent_ids = ["vector-top", "reranker-top", "third-parent"]
        query_calls, _ = stub_question_retrieval(
            monkeypatch,
            store,
            [question_point(parent_id) for parent_id in parent_ids],
            [parent_ids[1]],
        )

        sections = store.search_candidates_and_fetch_parent(
            "How does it work?",
            store.llm,
            query_vector=precomputed_vector,
        )

        assert store.llm.embed_calls == []
        assert query_calls[0]["query"] is precomputed_vector
        assert [section["metadata"]["parent_id"] for section in sections] == [
            parent_ids[1],
            parent_ids[0],
        ]
        assert [section["metadata"]["matched_knowledge"] for section in sections] == [
            "knowledge reranker-top",
            "knowledge vector-top",
        ]
    finally:
        delete_store(store)


def test_question_retrieval_deduplicates_matching_reranker_and_fills_next(monkeypatch):
    store = make_store()
    try:
        parent_ids = ["vector-top", "reranker-second", "reranker-third"]
        stub_question_retrieval(
            monkeypatch,
            store,
            [question_point(parent_id) for parent_id in parent_ids],
            [
                parent_ids[0],
                parent_ids[0],
                parent_ids[1],
                parent_ids[2],
            ],
        )

        sections = store.search_candidates_and_fetch_parent(
            "How does it work?", store.llm
        )

        assert [section["metadata"]["parent_id"] for section in sections] == [
            parent_ids[0],
            parent_ids[1],
        ]
        assert [section["metadata"]["matched_knowledge"] for section in sections] == [
            "knowledge vector-top",
            "knowledge reranker-second",
        ]
    finally:
        delete_store(store)


@pytest.mark.parametrize("reranker_output", [[], ["unknown-parent", None]])
def test_question_retrieval_invalid_reranker_falls_back_to_vector_order(
    monkeypatch, reranker_output
):
    store = make_store()
    try:
        parent_ids = ["vector-first", "vector-second", "vector-third"]
        stub_question_retrieval(
            monkeypatch,
            store,
            [question_point(parent_id) for parent_id in parent_ids],
            reranker_output,
        )

        sections = store.search_candidates_and_fetch_parent(
            "How does it work?", store.llm
        )

        assert [section["metadata"]["parent_id"] for section in sections] == parent_ids[:2]
        assert [section["metadata"]["matched_knowledge"] for section in sections] == [
            "knowledge vector-first",
            "knowledge vector-second",
        ]
    finally:
        delete_store(store)


def test_roadmap_steps_are_stored_in_sequence_order():
    store = make_store()
    try:
        metadata = {
            "source": "lora.md",
            "section": "Method",
            "seq_id": 1,
            "page_start": 3,
            "page_end": 4,
        }
        store.upsert_roadmap_step(
            {"seq_id": 1, "title": "Mechanism", "content_focus": "low rank", "concepts": ["LoRA"]},
            "b" * 32,
            metadata,
        )
        store.upsert_roadmap_step(
            {"seq_id": 0, "title": "Motivation", "content_focus": "cost", "concepts": []},
            "b" * 32,
            metadata,
        )

        roadmap = store.get_roadmap("b" * 32)

        assert [step["title"] for step in roadmap] == ["Motivation", "Mechanism"]
    finally:
        delete_store(store)


def test_bulk_curriculum_persistence_batches_roadmap_and_parent(monkeypatch):
    store = make_store()
    try:
        parent_id = "e" * 32
        roadmap_metadata = {
            "source": "lora.md",
            "section": "Method",
            "seq_id": 1,
            "page_start": 3,
            "page_end": 4,
        }
        section_metadata = {
            **roadmap_metadata,
            "content_hash": "current-content",
            "main_entities": ["LoRA"],
            "anchor_nodes": ["LoRA", "Low-Rank Matrix"],
        }
        roadmap_steps = [
            {
                "seq_id": 0,
                "title": "Motivation",
                "content_focus": "why low rank helps",
                "concepts": ["LoRA"],
            },
            {
                "seq_id": 1,
                "title": "Mechanism",
                "content_focus": "low-rank matrices update the model",
                "concepts": ["LoRA", "Low-Rank Matrix"],
            },
        ]
        text = "LoRA freezes the base model weights."
        calls = []
        original_upsert = store.client.upsert

        def capture_upsert(*args, **kwargs):
            calls.append(kwargs)
            return original_upsert(*args, **kwargs)

        monkeypatch.setattr(store.client, "upsert", capture_upsert)
        store.upsert_curriculum_section(
            roadmap_steps, text, roadmap_metadata, section_metadata, parent_id
        )

        assert store.llm.embed_calls == []
        assert store.llm.embed_many_calls == [
            ["why low rank helps", "low-rank matrices update the model", text]
        ]
        assert len(calls) == 1
        assert calls[0]["collection_name"] == store.curriculum_collection
        points = calls[0]["points"]
        assert [point.id for point in points] == [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_step_0")),
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_step_1")),
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_section")),
        ]
        assert [point.payload["type"] for point in points] == [
            "roadmap_step",
            "roadmap_step",
            "section_anchor",
        ]
        assert store.get_roadmap(parent_id) == [
            {
                **roadmap_metadata,
                "parent_id": parent_id,
                "type": "roadmap_step",
                **step,
            }
            for step in roadmap_steps
        ]
        section = store.get_section_exact("lora.md", "Method")
        assert len(section) == 1
        assert section[0]["page_content"] == text
        assert section[0]["metadata"]["content_hash"] == "current-content"
        assert section[0]["metadata"]["anchor_nodes"] == ["LoRA", "Low-Rank Matrix"]
        assert store.client.count(store.questions_collection, exact=True).count == 0
    finally:
        delete_store(store)


def test_bulk_curriculum_parent_still_resolves_from_question_retrieval():
    store = make_store()
    try:
        parent_id = "f" * 32
        metadata = {
            "source": "lora.md",
            "section": "Method",
            "seq_id": 1,
            "page_start": 3,
            "page_end": 4,
            "content_hash": "current-content",
            "anchor_nodes": ["LoRA"],
        }
        text = "LoRA freezes the base model weights."
        store.upsert_curriculum_section(
            [
                {
                    "seq_id": 0,
                    "title": "Mechanism",
                    "content_focus": "low-rank updates",
                    "concepts": ["LoRA"],
                }
            ],
            text,
            metadata,
            metadata,
            parent_id,
        )
        store.upsert_questions(
            [
                {
                    "question": "Which LoRA weights stay frozen?",
                    "key_knowledge": "The base model weights remain frozen.",
                }
            ],
            parent_id,
            "lora.md",
        )

        sections = store.search_candidates_and_fetch_parent(
            "What stays frozen in LoRA?", store.llm, "lora.md"
        )

        assert len(sections) == 1
        assert sections[0]["page_content"] == text
        assert sections[0]["metadata"]["parent_id"] == parent_id
        assert sections[0]["metadata"]["matched_knowledge"] == (
            "The base model weights remain frozen."
        )
    finally:
        delete_store(store)


def test_parent_hash_and_delete_replace_only_parent_points():
    store = make_store()
    try:
        parent_id = "c" * 32
        metadata = {
            "source": "lora.md",
            "section": "Method",
            "seq_id": 1,
            "page_start": 3,
            "page_end": 4,
            "content_hash": "current-content",
        }
        store.upsert_section("LoRA method text.", metadata, parent_id)
        store.upsert_roadmap_step(
            {"seq_id": 0, "title": "Mechanism", "content_focus": "low rank"},
            parent_id,
            metadata,
        )
        store.upsert_questions(
            [{"question": "How does LoRA work?", "key_knowledge": "Low rank."}],
            parent_id,
            "lora.md",
        )
        other_parent_id = "d" * 32
        store.upsert_section("Unrelated section.", metadata, other_parent_id)
        store.upsert_questions(
            [{"question": "What is unrelated?", "key_knowledge": "Separate."}],
            other_parent_id,
            "lora.md",
        )

        assert store.section_exists(parent_id, "current-content")
        assert not store.section_exists(parent_id, "stale-content")

        store.delete_parent(parent_id)

        assert not store.section_exists(parent_id)
        assert store.get_roadmap(parent_id) == []
        assert store.section_exists(other_parent_id)
        assert (
            store.client.count(store.questions_collection, exact=True).count == 1
        )
    finally:
        delete_store(store)


def test_existing_wrong_dimension_is_reported_without_recreating_collection():
    token = uuid.uuid4().hex
    store = make_store(embedding_dim=3, token=token)
    try:
        with pytest.raises(RuntimeError, match="uses vector dimension 3"):
            make_store(embedding_dim=4, token=token)
        assert store.client.count(store.curriculum_collection, exact=True).count == 0
    finally:
        delete_store(store)
