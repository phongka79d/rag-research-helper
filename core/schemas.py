"""Small validated payloads used by ingestion and retrieval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.relations import CANONICAL_RELATIONS, normalize_relation as canonicalize_relation

ALLOWED_RELATIONS = CANONICAL_RELATIONS
MAX_GRAPH_VERIFIER_CANDIDATES = 12


class GraphNode(BaseModel):
    name: str
    description: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str = "RELATES_TO"
    evidence_id: str = Field(min_length=1)

    @field_validator("relation")
    @classmethod
    def normalize_relation(cls, relation: str) -> str:
        normalized = canonicalize_relation(relation)
        if not normalized:
            raise ValueError("relation must be an upper-snake identifier.")
        return normalized


class GraphEdgeApproval(BaseModel):
    """A verifier approval that can only reference an existing edge candidate."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(strict=True)


class GraphEdgeVerificationResult(BaseModel):
    """Validated response shape for the bounded graph-edge verifier."""

    model_config = ConfigDict(extra="forbid")

    approvals: list[GraphEdgeApproval] = Field(
        max_length=MAX_GRAPH_VERIFIER_CANDIDATES
    )


class GraphEvidenceEdge(BaseModel):
    """A strict graph-only recovery candidate anchored to one evidence span."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, strict=True)
    relation: str = Field(strict=True)
    target: str = Field(min_length=1, strict=True)
    evidence_id: str = Field(min_length=1, strict=True)

    @field_validator("relation")
    @classmethod
    def require_known_relation(cls, relation: str) -> str:
        normalized = relation.strip().upper().replace(" ", "_")
        # Graph-only recovery is deliberately narrower than the main AOT graph:
        # it exists to recover explicit whole-part wording after the normal pass
        # produced no retained edge.  Other relations still use the existing
        # verifier path and must not enter this recovery payload.
        if normalized != "PART_OF":
            raise ValueError("graph recovery supports PART_OF edges only.")
        return normalized


class GraphEvidenceResult(BaseModel):
    """Validated bounded payload returned by the graph-only recovery request."""

    model_config = ConfigDict(extra="forbid")

    edges: list[GraphEvidenceEdge] = Field(max_length=MAX_GRAPH_VERIFIER_CANDIDATES)


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


class SectionPlanResult(BaseModel):
    """Narrow validated payload for the text-model planning request."""

    model_config = ConfigDict(extra="forbid")

    main_entities: list[str] = Field(default_factory=list)
    learning_roadmap: list[RoadmapStep] = Field(default_factory=list)


class SectionGraphResult(BaseModel):
    """Narrow validated payload for the graph-model extraction request."""

    model_config = ConfigDict(extra="forbid")

    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)


class HypotheticalQA(BaseModel):
    question: str
    key_knowledge: str
