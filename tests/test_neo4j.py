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
