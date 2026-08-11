import uuid
from types import SimpleNamespace

import pytest

from database.structural_db import QdrantVectorStore


class FakeLLM:
    def embed(self, text):
        return [
            float("lora" in text.lower()),
            float("quant" in text.lower()),
            1.0,
        ]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]

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


def test_existing_wrong_dimension_is_reported_without_recreating_collection():
    token = uuid.uuid4().hex
    store = make_store(embedding_dim=3, token=token)
    try:
        with pytest.raises(RuntimeError, match="uses vector dimension 3"):
            make_store(embedding_dim=4, token=token)
        assert store.client.count(store.curriculum_collection, exact=True).count == 0
    finally:
        delete_store(store)
