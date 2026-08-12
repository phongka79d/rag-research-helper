"""Qdrant storage for section anchors, roadmap steps, and HyDE questions."""

from __future__ import annotations

import uuid
from hashlib import sha256
from typing import Any

from qdrant_client import QdrantClient, models

CURRICULUM_COLLECTION = "research_curriculum"
QUESTIONS_COLLECTION = "research_questions"


class QdrantVectorStore:
    """Direct Qdrant operations for the application's two collections."""

    def __init__(
        self,
        settings: Any,
        llm_service: Any,
        curriculum_collection: str = CURRICULUM_COLLECTION,
        questions_collection: str = QUESTIONS_COLLECTION,
    ) -> None:
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.llm = llm_service
        self.embedding_dim = settings.OPENAI_EMBEDDING_DIM
        self.curriculum_collection = curriculum_collection
        self.questions_collection = questions_collection
        self.ensure_collections()

    def ensure_collections(self) -> None:
        collections = {
            self.curriculum_collection: ["source", "section", "type", "parent_id"],
            self.questions_collection: ["source", "type", "parent_id"],
        }
        for collection_name, indexes in collections.items():
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
            else:
                vector_params = self.client.get_collection(
                    collection_name
                ).config.params.vectors
                vector_size = getattr(vector_params, "size", None)
                if vector_size != self.embedding_dim:
                    raise RuntimeError(
                        f"Qdrant collection {collection_name!r} uses vector dimension "
                        f"{vector_size}; configured embeddings use {self.embedding_dim}. "
                        "Existing collection data was not changed."
                    )
            for field in indexes:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as error:
                    if "already exists" not in str(error).lower():
                        raise

    @staticmethod
    def _condition(key: str, value: str) -> models.FieldCondition:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    @classmethod
    def _filter(cls, **values: str) -> models.Filter:
        return models.Filter(
            must=[cls._condition(key, value) for key, value in values.items()]
        )

    @staticmethod
    def _section_point_id(parent_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_section"))

    def section_exists(self, parent_id: str, content_hash: str | None = None) -> bool:
        """Check a parent anchor and optionally its compiled content hash."""
        if content_hash is not None:
            section = self._fetch_parent(parent_id)
            return bool(
                section
                and section["metadata"].get("content_hash") == content_hash
            )
        return bool(
            self.client.count(
                collection_name=self.curriculum_collection,
                count_filter=self._filter(type="section_anchor", parent_id=parent_id),
                exact=True,
            ).count
        )

    def delete_parent(self, parent_id: str) -> None:
        """Remove only the stored Qdrant points belonging to one parent section."""
        selector = models.FilterSelector(filter=self._filter(parent_id=parent_id))
        for collection_name in (self.curriculum_collection, self.questions_collection):
            self.client.delete(
                collection_name=collection_name,
                points_selector=selector,
                wait=True,
            )

    def upsert_section(
        self, text: str, metadata: dict[str, Any], parent_id: str
    ) -> None:
        payload = {
            **metadata,
            "parent_id": parent_id,
            "type": "section_anchor",
            "page_content": text,
            "content_hash": metadata.get("content_hash")
            or sha256(text.encode("utf-8")).hexdigest(),
        }
        self.client.upsert(
            collection_name=self.curriculum_collection,
            points=[
                models.PointStruct(
                    id=self._section_point_id(parent_id),
                    vector=self.llm.embed(text),
                    payload=payload,
                )
            ],
        )

    def upsert_curriculum_section(
        self,
        roadmap_steps: list[dict[str, Any]],
        text: str,
        roadmap_metadata: dict[str, Any],
        section_metadata: dict[str, Any],
        parent_id: str,
    ) -> None:
        """Store one section anchor and its roadmap with one embedding and write."""
        embedding_inputs = [step["content_focus"] for step in roadmap_steps] + [text]
        vectors = self.llm.embed_many(embedding_inputs)
        if len(vectors) != len(embedding_inputs):
            raise RuntimeError(
                "Curriculum embeddings did not match roadmap steps and parent section."
            )

        points = []
        for step, vector in zip(roadmap_steps, vectors[:-1]):
            seq_id = int(step.get("seq_id", 0))
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_step_{seq_id}")),
                    vector=vector,
                    payload={
                        **roadmap_metadata,
                        "parent_id": parent_id,
                        "type": "roadmap_step",
                        "seq_id": seq_id,
                        "title": step["title"],
                        "content_focus": step["content_focus"],
                        "concepts": step.get("concepts", []),
                    },
                )
            )
        points.append(
            models.PointStruct(
                id=self._section_point_id(parent_id),
                vector=vectors[-1],
                payload={
                    **section_metadata,
                    "parent_id": parent_id,
                    "type": "section_anchor",
                    "page_content": text,
                    "content_hash": section_metadata.get("content_hash")
                    or sha256(text.encode("utf-8")).hexdigest(),
                },
            )
        )
        self.client.upsert(collection_name=self.curriculum_collection, points=points)

    def upsert_questions(
        self,
        qa_pairs: list[dict[str, str]],
        parent_id: str,
        source_file: str,
    ) -> None:
        if not qa_pairs:
            return
        questions = [pair["question"] for pair in qa_pairs]
        vectors = self.llm.embed_many(questions)
        if len(vectors) != len(qa_pairs):
            raise RuntimeError("Question embeddings did not match generated questions.")
        points = []
        for index, (pair, vector) in enumerate(zip(qa_pairs, vectors)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_q_{index}")),
                    vector=vector,
                    payload={
                        "page_content": pair["question"],
                        "key_knowledge": pair.get("key_knowledge", ""),
                        "parent_id": parent_id,
                        "source": source_file,
                        "type": "question",
                    },
                )
            )
        self.client.upsert(collection_name=self.questions_collection, points=points)

    def upsert_roadmap_step(
        self,
        step: dict[str, Any],
        parent_id: str,
        metadata: dict[str, Any],
    ) -> None:
        seq_id = int(step.get("seq_id", 0))
        payload = {
            **metadata,
            "parent_id": parent_id,
            "type": "roadmap_step",
            "seq_id": seq_id,
            "title": step["title"],
            "content_focus": step["content_focus"],
            "concepts": step.get("concepts", []),
        }
        self.client.upsert(
            collection_name=self.curriculum_collection,
            points=[
                models.PointStruct(
                    id=str(
                        uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_step_{seq_id}")
                    ),
                    vector=self.llm.embed(step["content_focus"]),
                    payload=payload,
                )
            ],
        )

    def _records_to_sections(
        self, records: list[models.Record]
    ) -> list[dict[str, Any]]:
        sections = []
        for record in records:
            payload = dict(record.payload or {})
            if payload.get("type") != "section_anchor":
                continue
            sections.append(
                {
                    "page_content": payload.get("page_content", ""),
                    "metadata": payload,
                }
            )
        return sections

    def get_section_exact(
        self, target_file: str, target_section: str
    ) -> list[dict[str, Any]]:
        values = {"type": "section_anchor"}
        if target_file:
            values["source"] = target_file
        if target_section:
            values["section"] = target_section
        records, _ = self.client.scroll(
            collection_name=self.curriculum_collection,
            scroll_filter=self._filter(**values),
            limit=1_000,
            with_payload=True,
            with_vectors=False,
        )
        return sorted(
            self._records_to_sections(records),
            key=lambda section: section["metadata"].get("seq_id", 0),
        )

    def get_roadmap(self, parent_id: str) -> list[dict[str, Any]]:
        records, _ = self.client.scroll(
            collection_name=self.curriculum_collection,
            scroll_filter=self._filter(type="roadmap_step", parent_id=parent_id),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        steps = [dict(record.payload or {}) for record in records]
        return sorted(steps, key=lambda step: step.get("seq_id", 0))

    def _fetch_parent(self, parent_id: str) -> dict[str, Any] | None:
        records, _ = self.client.scroll(
            collection_name=self.curriculum_collection,
            scroll_filter=self._filter(type="section_anchor", parent_id=parent_id),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        sections = self._records_to_sections(records)
        return sections[0] if sections else None

    def search_candidates_and_fetch_parent(
        self,
        query: str,
        llm_service: Any,
        target_file: str = "",
    ) -> list[dict[str, Any]]:
        conditions = [self._condition("type", "question")]
        if target_file:
            conditions.append(self._condition("source", target_file))
        results = self.client.query_points(
            collection_name=self.questions_collection,
            query=llm_service.embed(query),
            query_filter=models.Filter(must=conditions),
            limit=5,
            with_payload=True,
        ).points
        if not results:
            return []

        candidates = [
            {
                "question": (record.payload or {}).get("page_content", ""),
                "parent_id": (record.payload or {}).get("parent_id", ""),
                "key_knowledge": (record.payload or {}).get("key_knowledge", ""),
            }
            for record in results
        ]
        parent_ids = llm_service.rerank_candidate_questions(query, candidates)[:2]
        sections = []
        for parent_id in parent_ids:
            section = self._fetch_parent(parent_id)
            if section is None:
                continue
            matched = next(
                (candidate for candidate in candidates if candidate["parent_id"] == parent_id),
                {},
            )
            section["metadata"]["matched_knowledge"] = matched.get(
                "key_knowledge", ""
            )
            sections.append(section)
        return sections
