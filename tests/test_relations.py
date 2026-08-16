import pytest

from core.relations import (
    CANONICAL_RELATIONS,
    is_concept_endpoint,
    normalize_relation,
    quote_supports_relation,
)
from core.schemas import GraphEdge


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("uses", "USES"),
        ("relies on", "USES"),
        ("extends", "BASED_ON"),
        ("tested_on", "EVALUATED_ON"),
        ("FINE_TUNES", "FINE_TUNES"),
        ("not a relation!", ""),
        ("???", ""),
        ("", ""),
    ],
)
def test_normalize_relation_maps_aliases_and_keeps_novel_names(raw, expected):
    assert normalize_relation(raw) == expected


def test_canonical_table_covers_research_predicates():
    assert {
        "USES",
        "EVALUATED_ON",
        "BASED_ON",
        "APPLIED_TO",
        "HAS_FEATURE",
        "PART_OF",
    } <= CANONICAL_RELATIONS


@pytest.mark.parametrize(
    ("quote", "source", "relation", "target"),
    [
        ("The system uses a technique.", "system", "USES", "technique"),
        ("The model is evaluated on a benchmark.", "model", "EVALUATED_ON", "benchmark"),
        ("The approach is based on self-attention.", "approach", "BASED_ON", "self-attention"),
        ("The key components powering SLMs are quantization.", "quantization", "PART_OF", "SLMs"),
        ("QLoRA fine-tunes LLaMA.", "QLoRA", "FINE_TUNES", "LLaMA"),
        (
            "The model has residual connections.",
            "model",
            "HAS_FEATURE",
            "residual connections",
        ),
        (
            "The model has positional encodings.",
            "model",
            "HAS_FEATURE",
            "positional encodings",
        ),
        (
            "FFN is a component of the layer.",
            "FFN",
            "PART_OF",
            "layer",
        ),
    ],
)
def test_quote_supports_mapped_and_novel_predicates(quote, source, relation, target):
    assert quote_supports_relation(quote, source, relation, target)


def test_quote_rejects_wrong_label_and_cooccurrence():
    assert not quote_supports_relation(
        "The system uses a technique.", "technique", "PART_OF", "system"
    )
    assert not quote_supports_relation(
        "QLoRA and LLaMA appear together.", "QLoRA", "FINE_TUNES", "LLaMA"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "that",
        "where",
        "such",
        "little",
        "their",
        "intelligence that",
        "models such",
        "note",
        "but",
        "since",
    ],
)
def test_concept_endpoint_rejects_clause_and_closed_class(phrase):
    assert not is_concept_endpoint(phrase)


@pytest.mark.parametrize(
    "phrase",
    [
        "residual connections",
        "positional encodings",
        "multi-head attention",
        "FFN",
        "layer",
        "self-attention",
    ],
)
def test_concept_endpoint_keeps_concise_concepts(phrase):
    assert is_concept_endpoint(phrase)


@pytest.mark.parametrize(
    ("quote", "source", "relation", "target"),
    [
        (
            "The choice has little bearing on the result.",
            "choice",
            "HAS_FEATURE",
            "little bearing",
        ),
        (
            "The company has seen 100 million dollars.",
            "company",
            "HAS_FEATURE",
            "100 million dollars",
        ),
        (
            "The method has the advantage of simplicity.",
            "method",
            "HAS_FEATURE",
            "advantage",
        ),
        (
            "The model achieves a point.",
            "model",
            "ACHIEVES",
            "point",
        ),
        (
            "Training requires less compute than before.",
            "Training",
            "REQUIRES",
            "less",
        ),
        (
            "Training requires more data.",
            "Training",
            "REQUIRES",
            "more",
        ),
        (
            "Intelligence that enables agents to plan.",
            "Intelligence that",
            "ENABLES",
            "agents",
        ),
        (
            "Models such enable transfer.",
            "Models such",
            "ENABLES",
            "transfer",
        ),
    ],
)
def test_quote_rejects_near_miss_feature_result_require_enable(
    quote, source, relation, target
):
    assert not quote_supports_relation(quote, source, relation, target)


def test_graph_edge_keeps_novel_relation_and_rejects_garbage():
    edge = GraphEdge(
        source="QLoRA",
        target="LLaMA",
        relation="fine-tunes",
        evidence_id="e0",
    )
    assert edge.relation == "FINE_TUNES"
    with pytest.raises(Exception):
        GraphEdge(source="A", target="B", relation="???", evidence_id="e0")
