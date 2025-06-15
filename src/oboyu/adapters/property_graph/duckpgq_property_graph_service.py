"""DuckPGQ-based property graph service implementation.

This module implements property graph operations using DuckDB's PGQ extension
for native graph queries and analytics on the knowledge graph.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from duckdb import DuckDBPyConnection

from oboyu.domain.models.knowledge_graph import Entity, Relation
from oboyu.ports.services.property_graph_service import PropertyGraphError, PropertyGraphService

logger = logging.getLogger(__name__)


class DuckPGQPropertyGraphService(PropertyGraphService):
    """DuckPGQ-based implementation of property graph service."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize DuckPGQ property graph service.

        Args:
            connection: DuckDB database connection

        """
        self.connection = connection
        self._property_graph_initialized = False

    async def initialize_property_graph(self) -> bool:
        """Initialize the DuckPGQ property graph structure."""
        try:
            # Check if PGQ extension is available
            if not await self._check_pgq_extension():
                logger.warning("DuckPGQ extension not available, installing...")
                await self._install_pgq_extension()

            # Drop existing property graph if it exists
            try:
                self.connection.execute("DROP PROPERTY GRAPH IF EXISTS oboyu_kg")
                logger.info("Dropped existing property graph")
            except Exception as e:
                logger.debug(f"No existing property graph to drop: {e}")

            # Create the property graph
            create_pg_sql = """
                CREATE PROPERTY GRAPH oboyu_kg
                VERTEX TABLES (kg_entities LABEL entity)
                EDGE TABLES (
                    kg_relations
                    SOURCE KEY (source_id) REFERENCES kg_entities (id)
                    DESTINATION KEY (target_id) REFERENCES kg_entities (id)
                    LABEL relation
                )
            """

            self.connection.execute(create_pg_sql)
            self._property_graph_initialized = True

            logger.info("DuckPGQ property graph 'oboyu_kg' created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize property graph: {e}")
            raise PropertyGraphError(f"Property graph initialization failed: {e}", "initialize_property_graph")

    async def _check_pgq_extension(self) -> bool:
        """Check if PGQ extension is available."""
        try:
            result = self.connection.execute("SELECT * FROM duckdb_extensions() WHERE extension_name = 'pgq'").fetchone()
            return result is not None and result[1]  # Check if loaded
        except Exception:
            return False

    async def _install_pgq_extension(self) -> None:
        """Install and load the PGQ extension."""
        try:
            self.connection.execute("INSTALL 'pgq'")
            self.connection.execute("LOAD 'pgq'")
            logger.info("DuckPGQ extension installed and loaded successfully")
        except Exception as e:
            logger.error(f"Failed to install PGQ extension: {e}")
            raise PropertyGraphError(f"PGQ extension installation failed: {e}", "_install_pgq_extension")

    async def is_property_graph_available(self) -> bool:
        """Check if DuckPGQ extension and property graph are available."""
        try:
            # Check if extension is available
            if not await self._check_pgq_extension():
                return False

            # Check if property graph exists
            if not self._property_graph_initialized:
                try:
                    # Try to query the property graph to see if it exists
                    self.connection.execute("SELECT COUNT(*) FROM oboyu_kg.entity").fetchone()
                    self._property_graph_initialized = True
                except Exception:
                    return False

            return True
        except Exception:
            return False

    async def find_shortest_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_hops: int = 6,
        relation_types: Optional[List[str]] = None,
    ) -> List[Tuple[Entity, Relation]]:
        """Find shortest path between two entities using graph traversal."""
        try:
            if not await self.is_property_graph_available():
                raise PropertyGraphError("Property graph not available", "find_shortest_path")

            # Build relation type filter
            relation_filter = ""
            if relation_types:
                relation_filter = f"WHERE r.relation_type IN ({','.join(['?' for _ in relation_types])})"

            # DuckPGQ shortest path query
            query = f"""
                SELECT p.path_edges, p.path_vertices
                FROM (
                    SELECT path_edges, path_vertices
                    FROM oboyu_kg
                    MATCH SHORTEST (src:entity)-[r:relation*1..{max_hops}]->(dst:entity)
                    WHERE src.id = ? AND dst.id = ?
                    {relation_filter}
                ) p
                LIMIT 1
            """

            params: List[Any] = [source_entity_id, target_entity_id]
            if relation_types:
                params.extend(relation_types)

            result = self.connection.execute(query, params).fetchone()

            if not result:
                return []

            # Parse the path result and convert to entities/relations
            path_edges = json.loads(result[0]) if result[0] else []
            path_vertices = json.loads(result[1]) if result[1] else []

            # Convert to Entity and Relation objects
            path = []
            for i, edge_id in enumerate(path_edges):
                # Get relation details
                relation_result = self.connection.execute("SELECT * FROM kg_relations WHERE id = ?", (edge_id,)).fetchone()

                if relation_result and i < len(path_vertices) - 1:
                    # Get entity details
                    entity_result = self.connection.execute("SELECT * FROM kg_entities WHERE id = ?", (path_vertices[i + 1],)).fetchone()

                    if entity_result:
                        entity = self._entity_from_row(entity_result)
                        relation = self._relation_from_row(relation_result)
                        path.append((entity, relation))

            return path

        except Exception as e:
            logger.error(f"Failed to find shortest path from {source_entity_id} to {target_entity_id}: {e}")
            raise PropertyGraphError(f"Shortest path query failed: {e}", "find_shortest_path")

    async def find_entity_subgraph(
        self,
        entity_id: str,
        depth: int = 2,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract subgraph around an entity within specified depth."""
        try:
            if not await self.is_property_graph_available():
                raise PropertyGraphError("Property graph not available", "find_entity_subgraph")

            # Build filters
            entity_filter = ""
            if entity_types:
                entity_filter = f"AND e.entity_type IN ({','.join(['?' for _ in entity_types])})"

            relation_filter = ""
            if relation_types:
                relation_filter = f"AND r.relation_type IN ({','.join(['?' for _ in relation_types])})"

            # DuckPGQ subgraph extraction query
            query = f"""
                SELECT DISTINCT e.*, r.*
                FROM oboyu_kg
                MATCH (center:entity)-[r:relation*1..{depth}]-(e:entity)
                WHERE center.id = ? {entity_filter} {relation_filter}
            """

            params: List[Any] = [entity_id]
            if entity_types:
                params.extend(entity_types)
            if relation_types:
                params.extend(relation_types)

            results = self.connection.execute(query, params).fetchall()

            # Parse results into entities and relations
            entities = {}
            relations = {}

            for row in results:
                # Parse entity data (first part of row)
                entity_data = row[:12]  # Assuming 12 entity columns
                entity = self._entity_from_row(entity_data)
                entities[entity.id] = entity

                # Parse relation data (second part of row)
                if len(row) > 12:
                    relation_data = row[12:]
                    relation = self._relation_from_row(relation_data)
                    relations[relation.id] = relation

            return {
                "entities": list(entities.values()),
                "relations": list(relations.values()),
                "center_entity_id": entity_id,
                "depth": depth,
            }

        except Exception as e:
            logger.error(f"Failed to extract subgraph for entity {entity_id}: {e}")
            raise PropertyGraphError(f"Subgraph extraction failed: {e}", "find_entity_subgraph")

    async def get_entity_centrality_scores(
        self,
        entity_type: Optional[str] = None,
        limit: int = 50,
        centrality_type: str = "degree",
    ) -> List[Tuple[str, str, float]]:
        """Calculate centrality scores for entities in the graph."""
        try:
            if not await self.is_property_graph_available():
                raise PropertyGraphError("Property graph not available", "get_entity_centrality_scores")

            # Build entity type filter
            type_filter = ""
            params: List[Any] = []
            if entity_type:
                type_filter = "WHERE e.entity_type = ?"
                params.append(entity_type)

            if centrality_type == "degree":
                # Degree centrality - count of connections
                query = f"""
                    SELECT e.id, e.name, COUNT(*) as centrality
                    FROM oboyu_kg
                    MATCH (e:entity)-[r:relation]-(other:entity)
                    {type_filter}
                    GROUP BY e.id, e.name
                    ORDER BY centrality DESC
                    LIMIT ?
                """
                params.append(limit)

            elif centrality_type == "betweenness":
                # Simplified betweenness using path counting
                query = f"""
                    SELECT e.id, e.name,
                           COUNT(DISTINCT p.path_id) as centrality
                    FROM oboyu_kg
                    MATCH p = (start:entity)-[*]-(e:entity)-[*]-(end:entity)
                    WHERE start.id != e.id AND end.id != e.id AND start.id != end.id
                    {type_filter.replace("e.", "e.")}
                    GROUP BY e.id, e.name
                    ORDER BY centrality DESC
                    LIMIT ?
                """
                params.append(limit)

            else:  # Default to degree centrality
                query = f"""
                    SELECT e.id, e.name, COUNT(*) as centrality
                    FROM oboyu_kg
                    MATCH (e:entity)-[r:relation]-(other:entity)
                    {type_filter}
                    GROUP BY e.id, e.name
                    ORDER BY centrality DESC
                    LIMIT ?
                """
                params.append(limit)

            results = self.connection.execute(query, params).fetchall()
            return [(row[0], row[1], float(row[2])) for row in results]

        except Exception as e:
            logger.error(f"Failed to calculate {centrality_type} centrality: {e}")
            raise PropertyGraphError(f"Centrality calculation failed: {e}", "get_entity_centrality_scores")

    async def find_connected_components(
        self,
        min_component_size: int = 3,
    ) -> List[List[str]]:
        """Find strongly connected components in the knowledge graph."""
        try:
            if not await self.is_property_graph_available():
                raise PropertyGraphError("Property graph not available", "find_connected_components")

            # Use DuckPGQ to find connected components
            query = """
                SELECT component_id, ARRAY_AGG(entity_id) as entities
                FROM (
                    SELECT e.id as entity_id,
                           DENSE_RANK() OVER (ORDER BY MIN(connected.id)) as component_id
                    FROM oboyu_kg
                    MATCH (e:entity)-[*]-(connected:entity)
                    GROUP BY e.id
                ) components
                GROUP BY component_id
                HAVING COUNT(*) >= ?
                ORDER BY COUNT(*) DESC
            """

            results = self.connection.execute(query, (min_component_size,)).fetchall()
            return [json.loads(row[1]) for row in results]

        except Exception as e:
            logger.error(f"Failed to find connected components: {e}")
            raise PropertyGraphError(f"Connected components detection failed: {e}", "find_connected_components")

    async def execute_cypher_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher-like graph query using DuckPGQ."""
        try:
            if not await self.is_property_graph_available():
                raise PropertyGraphError("Property graph not available", "execute_cypher_query")

            # Convert parameters to list if provided
            params = list(parameters.values()) if parameters else []

            results = self.connection.execute(query, params).fetchall()

            # Get column names
            columns = [desc[0] for desc in self.connection.description] if self.connection.description else []

            # Convert to list of dictionaries
            return [dict(zip(columns, row)) for row in results]

        except Exception as e:
            logger.error(f"Failed to execute graph query: {e}")
            raise PropertyGraphError(f"Graph query execution failed: {e}", "execute_cypher_query")

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the property graph."""
        try:
            if not await self.is_property_graph_available():
                raise PropertyGraphError("Property graph not available", "get_graph_statistics")

            stats = {}

            # Basic counts
            entity_result = self.connection.execute("SELECT COUNT(*) FROM oboyu_kg.entity").fetchone()
            relation_result = self.connection.execute("SELECT COUNT(*) FROM oboyu_kg.relation").fetchone()

            entity_count = entity_result[0] if entity_result else 0
            relation_count = relation_result[0] if relation_result else 0

            stats["entity_count"] = entity_count
            stats["relation_count"] = relation_count

            # Entity type distribution
            entity_types = self.connection.execute("""
                SELECT entity_type, COUNT(*)
                FROM kg_entities
                GROUP BY entity_type
                ORDER BY COUNT(*) DESC
            """).fetchall()
            stats["entity_types"] = {row[0]: row[1] for row in entity_types}

            # Relation type distribution
            relation_types = self.connection.execute("""
                SELECT relation_type, COUNT(*)
                FROM kg_relations
                GROUP BY relation_type
                ORDER BY COUNT(*) DESC
            """).fetchall()
            stats["relation_types"] = {row[0]: row[1] for row in relation_types}

            # Graph density
            if entity_count > 1:
                max_edges = entity_count * (entity_count - 1)
                stats["density"] = relation_count / max_edges if max_edges > 0 else 0.0
            else:
                stats["density"] = 0.0

            # Average degree
            if entity_count > 0:
                degree_result = self.connection.execute("""
                    SELECT COUNT(*) * 2 FROM kg_relations
                """).fetchone()
                total_degree = degree_result[0] if degree_result else 0
                stats["average_degree"] = total_degree / entity_count
            else:
                stats["average_degree"] = 0.0

            return stats

        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            raise PropertyGraphError(f"Graph statistics calculation failed: {e}", "get_graph_statistics")

    def _entity_from_row(self, row: Tuple) -> Entity:
        """Convert database row tuple to Entity object."""
        from datetime import datetime

        return Entity(
            id=row[0],
            name=row[1],
            entity_type=row[2],
            definition=row[3],
            properties=json.loads(row[4]) if row[4] else {},
            chunk_id=row[5],
            canonical_name=row[6],
            merged_from=json.loads(row[7]) if row[7] else [],
            merge_confidence=row[8],
            confidence=row[9],
            created_at=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
            updated_at=datetime.fromisoformat(row[11]) if row[11] else datetime.now(),
        )

    def _relation_from_row(self, row: Tuple) -> Relation:
        """Convert database row tuple to Relation object."""
        from datetime import datetime

        return Relation(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation_type=row[3],
            properties=json.loads(row[4]) if row[4] else {},
            chunk_id=row[5],
            confidence=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.now(),
            updated_at=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
        )
