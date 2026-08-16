import uuid

from config.settings import Settings
from database.semantic_dag import Neo4jManager


def test_concepts_merge_across_sources_and_traverse():
    manager = Neo4jManager(Settings())
    prefix = f"test-{uuid.uuid4().hex}"
    lora = f"{prefix}-LoRA"
    matrix = f"{prefix}-Matrix"
    qlora = f"{prefix}-QLoRA"
    first_source = {"source": f"{prefix}-a.pdf", "section": "Method"}
    second_source = {"source": f"{prefix}-b.pdf", "section": "Background"}
    try:
        manager.verify_connection()
        manager.save_knowledge_graph(
            nodes=[
                {"name": matrix, "description": "A low-rank matrix."},
                {"name": lora, "description": "An adaptation method."},
            ],
            edges=[
                {"source": matrix, "target": lora, "relation": "PREREQUISITE_OF"}
            ],
            source=first_source,
            main_entities=[lora],
        )
        manager.save_knowledge_graph(
            nodes=[
                {"name": lora, "description": "An adaptation method."},
                {"name": qlora, "description": "A quantized extension."},
            ],
            edges=[
                {"source": lora, "target": qlora, "relation": "extends"}
            ],
            source=second_source,
            main_entities=[qlora],
        )

        visual = manager.get_visual_graph()
        lora_node = next(node for node in visual["nodes"] if node["id"] == lora)
        context = manager.get_graph_context([lora], search_mode="semi_search")
        subgraph = manager.get_concept_subgraph(lora)

        assert set(lora_node["source_locators"]) == {
            f"{first_source['source']}::{first_source['section']}",
            f"{second_source['source']}::{second_source['section']}",
        }
        assert context == [
            {
                "source": matrix,
                "source_desc": "A low-rank matrix.",
                "relation": "PREREQUISITE_OF",
                "target": lora,
                "target_desc": "An adaptation method.",
            }
        ]
        assert subgraph["prerequisites"] == [matrix]
        assert subgraph["related_concepts"] == [qlora]
        test_edges = [
            edge
            for edge in visual["edges"]
            if edge["source"].startswith(prefix) and edge["target"].startswith(prefix)
        ]
        assert {edge["label"] for edge in test_edges} == {
            "PREREQUISITE_OF",
            "RELATES_TO",
        }
    finally:
        with manager.driver.session() as session:
            session.run(
                "MATCH (c:Concept) WHERE c.id STARTS WITH $prefix DETACH DELETE c",
                prefix=prefix,
            )
        manager.close()


def test_visual_graph_filters_relationships_by_source_locator():
    manager = Neo4jManager(Settings())
    prefix = f"test-{uuid.uuid4().hex}"
    first = f"{prefix}-First"
    second = f"{prefix}-Second"
    first_source = {"source": f"{prefix}-a.pdf", "section": "Method"}
    second_source = {"source": f"{prefix}-b.pdf", "section": "Results"}
    first_locator = f"{first_source['source']}::{first_source['section']}"
    second_locator = f"{second_source['source']}::{second_source['section']}"
    try:
        manager.verify_connection()
        manager.save_knowledge_graph(
            nodes=[{"name": first}, {"name": second}],
            edges=[
                {
                    "source": first,
                    "target": second,
                    "relation": "PREREQUISITE_OF",
                }
            ],
            source=first_source,
            main_entities=[],
        )
        manager.save_knowledge_graph(
            nodes=[{"name": first}, {"name": second}],
            edges=[
                {"source": second, "target": first, "relation": "RELATES_TO"}
            ],
            source=second_source,
            main_entities=[],
        )

        first_edges = manager.get_visual_graph(first_locator)["edges"]
        second_edges = manager.get_visual_graph(second_locator)["edges"]

        assert [(edge["source"], edge["label"], edge["target"]) for edge in first_edges] == [
            (first, "PREREQUISITE_OF", second)
        ]
        assert [(edge["source"], edge["label"], edge["target"]) for edge in second_edges] == [
            (second, "RELATES_TO", first)
        ]
        paper_edges = manager.get_visual_graph(source=first_source["source"])["edges"]
        assert [(edge["source"], edge["label"], edge["target"]) for edge in paper_edges] == [
            (first, "PREREQUISITE_OF", second)
        ]

        manager.remove_source_locator(first_source)

        assert manager.get_visual_graph(first_locator)["edges"] == []
        assert manager.get_visual_graph(second_locator)["edges"] == second_edges
    finally:
        with manager.driver.session() as session:
            session.run(
                "MATCH (c:Concept) WHERE c.id STARTS WITH $prefix DETACH DELETE c",
                prefix=prefix,
            )
        manager.close()


def test_remove_source_locator_preserves_concept_shared_by_another_source():
    manager = Neo4jManager(Settings())
    prefix = f"test-{uuid.uuid4().hex}"
    shared = f"{prefix}-Shared"
    first_source = {"source": f"{prefix}-a.pdf", "section": "References"}
    second_source = {"source": f"{prefix}-b.pdf", "section": "Method"}
    first_locator = f"{first_source['source']}::{first_source['section']}"
    second_locator = f"{second_source['source']}::{second_source['section']}"
    try:
        manager.verify_connection()
        for source in (first_source, second_source):
            manager.save_knowledge_graph(
                nodes=[{"name": shared, "description": "Shared concept."}],
                edges=[],
                source=source,
                main_entities=[shared],
            )

        manager.remove_source_locator(first_source)

        remaining = manager.get_visual_graph(second_locator)["nodes"]
        assert remaining == [
            {
                "id": shared,
                "description": "Shared concept.",
                "source_locators": [second_locator],
                "is_main": True,
            }
        ]
        assert manager.get_visual_graph(first_locator)["nodes"] == []
    finally:
        with manager.driver.session() as session:
            session.run(
                "MATCH (c:Concept) WHERE c.id STARTS WITH $prefix DETACH DELETE c",
                prefix=prefix,
            )
        manager.close()


def test_graph_context_filters_relationship_provenance_by_source_prefix():
    manager = Neo4jManager(Settings())
    prefix = f"test-{uuid.uuid4().hex}"
    anchor = f"{prefix}-Anchor"
    first = f"{prefix}-First"
    second = f"{prefix}-Second"
    first_source = {"source": f"{prefix}-paper.pdf", "section": "Method"}
    second_source = {
        "source": f"{prefix}-paper.pdf-extra",
        "section": "Results",
    }
    try:
        manager.verify_connection()
        manager.save_knowledge_graph(
            nodes=[{"name": first}, {"name": anchor}],
            edges=[
                {
                    "source": first,
                    "target": anchor,
                    "relation": "PREREQUISITE_OF",
                }
            ],
            source=first_source,
            main_entities=[],
        )
        manager.save_knowledge_graph(
            nodes=[{"name": second}, {"name": anchor}],
            edges=[
                {
                    "source": second,
                    "target": anchor,
                    "relation": "RELATES_TO",
                }
            ],
            source=second_source,
            main_entities=[],
        )

        def relation_pairs(context):
            return {(item["source"], item["relation"], item["target"]) for item in context}

        expected_first = {(first, "PREREQUISITE_OF", anchor)}
        expected_second = {(second, "RELATES_TO", anchor)}

        assert relation_pairs(
            manager.get_graph_context([anchor], search_mode="search", source=first_source["source"])
        ) == expected_first
        assert relation_pairs(
            manager.get_graph_context([anchor], search_mode="semi_search", source=second_source["source"])
        ) == expected_second
        assert relation_pairs(manager.get_graph_context([anchor], search_mode="semi_search")) == (
            expected_first | expected_second
        )
    finally:
        with manager.driver.session() as session:
            session.run(
                "MATCH (c:Concept) WHERE c.id STARTS WITH $prefix DETACH DELETE c",
                prefix=prefix,
            )
        manager.close()
