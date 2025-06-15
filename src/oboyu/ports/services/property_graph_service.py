"""Property Graph service interface.

This module defines the port interface for DuckPGQ-based property graph queries
and graph analytics operations on the knowledge graph.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from oboyu.domain.models.knowledge_graph import Entity, Relation


class PropertyGraphService(ABC):
    """Abstract interface for property graph operations using DuckPGQ."""

    @abstractmethod
    async def initialize_property_graph(self) -> bool:
        """Initialize the DuckPGQ property graph structure.

        Returns:
            True if property graph was successfully created

        Raises:
            PropertyGraphError: If property graph initialization fails

        """

    @abstractmethod
    async def is_property_graph_available(self) -> bool:
        """Check if DuckPGQ extension and property graph are available.

        Returns:
            True if property graph functionality is ready

        """

    @abstractmethod
    async def find_shortest_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_hops: int = 6,
        relation_types: Optional[List[str]] = None,
    ) -> List[Tuple[Entity, Relation]]:
        """Find shortest path between two entities using graph traversal.

        Args:
            source_entity_id: Starting entity ID
            target_entity_id: Target entity ID
            max_hops: Maximum number of hops to search
            relation_types: Optional filter for relation types

        Returns:
            List of (entity, relation) tuples representing the path

        Raises:
            PropertyGraphError: If path finding fails

        """

    @abstractmethod
    async def find_entity_subgraph(
        self,
        entity_id: str,
        depth: int = 2,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract subgraph around an entity within specified depth.

        Args:
            entity_id: Central entity ID
            depth: Maximum depth to explore
            entity_types: Optional filter for entity types
            relation_types: Optional filter for relation types

        Returns:
            Dictionary with 'entities' and 'relations' keys containing subgraph

        Raises:
            PropertyGraphError: If subgraph extraction fails

        """

    @abstractmethod
    async def get_entity_centrality_scores(
        self,
        entity_type: Optional[str] = None,
        limit: int = 50,
        centrality_type: str = "degree",
    ) -> List[Tuple[str, str, float]]:
        """Calculate centrality scores for entities in the graph.

        Args:
            entity_type: Optional filter for specific entity type
            limit: Maximum number of results
            centrality_type: Type of centrality ('degree', 'betweenness', 'closeness')

        Returns:
            List of (entity_id, entity_name, centrality_score) tuples

        Raises:
            PropertyGraphError: If centrality calculation fails

        """

    @abstractmethod
    async def find_connected_components(
        self,
        min_component_size: int = 3,
    ) -> List[List[str]]:
        """Find strongly connected components in the knowledge graph.

        Args:
            min_component_size: Minimum size of components to return

        Returns:
            List of component entity ID lists

        Raises:
            PropertyGraphError: If component detection fails

        """

    @abstractmethod
    async def execute_cypher_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher-like graph query using DuckPGQ.

        Args:
            query: Graph query in Cypher-like syntax
            parameters: Optional query parameters

        Returns:
            Query results as list of dictionaries

        Raises:
            PropertyGraphError: If query execution fails

        """

    @abstractmethod
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the property graph.

        Returns:
            Dictionary with graph metrics (nodes, edges, density, etc.)

        Raises:
            PropertyGraphError: If statistics calculation fails

        """


class PropertyGraphError(Exception):
    """Exception raised when property graph operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        """Initialize property graph error.

        Args:
            message: Error description
            operation: Optional operation that failed

        """
        super().__init__(message)
        self.operation = operation
