"""Neo4j concept relationships for research-paper sections."""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from core.schemas import ALLOWED_RELATIONS


class Neo4jManager:
    """Direct Neo4j access for the application's global Concept graph."""

    def __init__(self, settings: Any) -> None:
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

    def verify_connection(self) -> None:
        try:
            self.driver.verify_connectivity()
        except Exception as error:
            raise RuntimeError(
                "Neo4j connectivity failed. Check NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD."
            ) from error

    def close(self) -> None:
        self.driver.close()

    def get_all_concept_names(self) -> list[str]:
        with self.driver.session() as session:
            records = session.run("MATCH (c:Concept) RETURN c.id AS id ORDER BY id")
            return [record["id"] for record in records]

    @staticmethod
    def _locator(source: dict[str, Any]) -> str:
        return f"{source.get('source', 'Unknown source')}::{source.get('section', 'Unknown section')}"

    @staticmethod
    def _relation(value: Any) -> str:
        relation = str(value or "RELATES_TO").strip().upper().replace(" ", "_")
        return relation if relation in ALLOWED_RELATIONS else "RELATES_TO"

    def save_knowledge_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        source: dict[str, Any],
        main_entities: list[str],
    ) -> None:
        locator = self._locator(source)
        node_map: dict[str, str] = {}
        for node in nodes:
            name = str(node.get("name", "")).strip()
            if name:
                node_map[name] = str(node.get("description", "")).strip()
        for entity in main_entities:
            name = str(entity).strip()
            if name:
                node_map.setdefault(name, "")
        for edge in edges:
            for endpoint in (edge.get("source", ""), edge.get("target", "")):
                name = str(endpoint).strip()
                if name:
                    node_map.setdefault(name, "")

        with self.driver.session() as session:
            for name, description in node_map.items():
                session.run(
                    """
                    MERGE (c:Concept {id: $id})
                    SET c.description = CASE
                        WHEN $description <> '' THEN $description
                        ELSE coalesce(c.description, '')
                    END,
                    c.is_main = coalesce(c.is_main, false) OR $is_main,
                    c.source_locators = CASE
                        WHEN $locator IN coalesce(c.source_locators, [])
                        THEN c.source_locators
                        ELSE coalesce(c.source_locators, []) + $locator
                    END
                    """,
                    id=name,
                    description=description,
                    is_main=name in main_entities,
                    locator=locator,
                )

            for edge in edges:
                source_name = str(edge.get("source", "")).strip()
                target_name = str(edge.get("target", "")).strip()
                if not source_name or not target_name:
                    continue
                relation = self._relation(edge.get("relation"))
                session.run(
                    f"""
                    MATCH (source:Concept {{id: $source}})
                    MATCH (target:Concept {{id: $target}})
                    MERGE (source)-[relationship:{relation}]->(target)
                    SET relationship.source_locators = CASE
                        WHEN $locator IN coalesce(relationship.source_locators, [])
                        THEN relationship.source_locators
                        ELSE coalesce(relationship.source_locators, []) + $locator
                    END
                    """,
                    source=source_name,
                    target=target_name,
                    locator=locator,
                )

    def remove_source_locator(self, source: dict[str, Any]) -> None:
        """Remove one source locator's graph contribution without touching others."""
        locator = self._locator(source)
        with self.driver.session() as session:
            session.run(
                """
                MATCH ()-[relationship]->()
                WHERE $locator IN coalesce(relationship.source_locators, [])
                WITH relationship,
                    [item IN relationship.source_locators WHERE item <> $locator] AS locators
                SET relationship.source_locators = locators
                WITH relationship
                WHERE size(relationship.source_locators) = 0
                DELETE relationship
                """,
                locator=locator,
            )
            session.run(
                """
                MATCH (concept:Concept)
                WHERE $locator IN coalesce(concept.source_locators, [])
                WITH concept,
                    [item IN concept.source_locators WHERE item <> $locator] AS locators
                SET concept.source_locators = locators
                WITH concept
                WHERE size(concept.source_locators) = 0
                DETACH DELETE concept
                """,
                locator=locator,
            )

    def get_graph_context(
        self, node_names: list[str], search_mode: str = "search"
    ) -> list[dict[str, str]]:
        names = [name for name in dict.fromkeys(node_names) if name]
        if not names:
            return []
        if search_mode == "semi_search":
            query = """
                MATCH (source:Concept)-[relationship]->(target:Concept)
                WHERE target.id IN $names
                RETURN DISTINCT source.id AS source,
                    coalesce(source.description, '') AS source_desc,
                    type(relationship) AS relation,
                    target.id AS target,
                    coalesce(target.description, '') AS target_desc
            """
        else:
            query = """
                MATCH path=(anchor:Concept)-[*1..2]-(other:Concept)
                WHERE anchor.id IN $names
                UNWIND relationships(path) AS relationship
                WITH DISTINCT relationship
                RETURN startNode(relationship).id AS source,
                    coalesce(startNode(relationship).description, '') AS source_desc,
                    type(relationship) AS relation,
                    endNode(relationship).id AS target,
                    coalesce(endNode(relationship).description, '') AS target_desc
            """
        with self.driver.session() as session:
            records = session.run(query, names=names)
            return [dict(record) for record in records]

    def get_concept_subgraph(
        self, target_concept: str, max_depth: int = 2
    ) -> dict[str, list[str]]:
        depth = max(1, min(int(max_depth), 5))
        with self.driver.session() as session:
            record = session.run(
                f"""
                MATCH (target:Concept {{id: $target}})
                OPTIONAL MATCH (prerequisite:Concept)-[:PREREQUISITE_OF*1..{depth}]->(target)
                OPTIONAL MATCH (target)-[:PREREQUISITE_OF*1..{depth}]->(next:Concept)
                OPTIONAL MATCH (target)-[:RELATES_TO]-(related:Concept)
                RETURN collect(DISTINCT prerequisite.id) AS prerequisites,
                    collect(DISTINCT next.id) AS leads_to,
                    collect(DISTINCT related.id) AS related_concepts
                """,
                target=target_concept,
            ).single()
        if record is None:
            return {"prerequisites": [], "leads_to": [], "related_concepts": []}
        return {
            key: sorted(item for item in record[key] if item is not None)
            for key in ("prerequisites", "leads_to", "related_concepts")
        }

    def get_visual_graph(self, locator: str | None = None) -> dict[str, list[dict[str, Any]]]:
        node_where = "" if locator is None else "WHERE $locator IN coalesce(c.source_locators, [])"
        edge_where = (
            ""
            if locator is None
            else "WHERE $locator IN coalesce(relationship.source_locators, []) "
            "OR (relationship.source_locators IS NULL "
            "AND $locator IN coalesce(source.source_locators, []) "
            "AND $locator IN coalesce(target.source_locators, []))"
        )
        with self.driver.session() as session:
            node_records = session.run(
                f"""
                MATCH (c:Concept)
                {node_where}
                RETURN c.id AS id,
                    coalesce(c.description, '') AS description,
                    coalesce(c.source_locators, []) AS source_locators,
                    coalesce(c.is_main, false) AS is_main
                ORDER BY id
                """,
                locator=locator,
            )
            edge_records = session.run(
                f"""
                MATCH (source:Concept)-[relationship]->(target:Concept)
                {edge_where}
                RETURN source.id AS source, type(relationship) AS label, target.id AS target
                ORDER BY source, target, label
                """,
                locator=locator,
            )
            return {
                "nodes": [dict(record) for record in node_records],
                "edges": [dict(record) for record in edge_records],
            }
