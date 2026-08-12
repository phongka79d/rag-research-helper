"""Small validated payloads used by ingestion and retrieval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_RELATIONS = {"PREREQUISITE_OF", "RELATES_TO", "PART_OF", "DESCRIBES"}
MAX_GRAPH_VERIFIER_CANDIDATES = 12


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


class GraphEdgeApproval(BaseModel):
    """A verifier approval that can only reference an existing edge candidate."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(strict=True)
    quote: str = Field(min_length=1, max_length=500, strict=True)


class GraphEdgeVerificationResult(BaseModel):
    """Validated response shape for the bounded graph-edge verifier."""

    model_config = ConfigDict(extra="forbid")

    approvals: list[GraphEdgeApproval] = Field(
        max_length=MAX_GRAPH_VERIFIER_CANDIDATES
    )


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
