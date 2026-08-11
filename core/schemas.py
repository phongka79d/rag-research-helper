"""Small validated payloads used by ingestion and retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ALLOWED_RELATIONS = {"PREREQUISITE_OF", "RELATES_TO", "PART_OF", "DESCRIBES"}


class GraphNode(BaseModel):
    name: str
    description: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str = "RELATES_TO"

    @field_validator("relation")
    @classmethod
    def normalize_relation(cls, relation: str) -> str:
        normalized = relation.strip().upper().replace(" ", "_")
        return normalized if normalized in ALLOWED_RELATIONS else "RELATES_TO"


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class RoadmapStep(BaseModel):
    seq_id: int = 0
    title: str
    content_focus: str
    concepts: list[str] = Field(default_factory=list)


class SectionAOTResult(BaseModel):
    main_entities: list[str] = Field(default_factory=list)
    learning_roadmap: list[RoadmapStep] = Field(default_factory=list)
    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)


class HypotheticalQA(BaseModel):
    question: str
    key_knowledge: str
