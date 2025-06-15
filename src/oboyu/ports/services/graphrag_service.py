"""GraphRAG service interface.

This module defines the port interface for GraphRAG (Graph Retrieval Augmented Generation)
operations, combining knowledge graph queries with semantic search for enhanced context.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from oboyu.domain.models.knowledge_graph import Entity, Relation


class GraphRAGService(ABC):
    """Abstract interface for GraphRAG operations."""

    @abstractmethod
    async def expand_query_with_entities(
        self,
        query: str,
        max_entities: int = 10,
        entity_similarity_threshold: float = 0.7,
        expand_depth: int = 1,
    ) -> Dict[str, Any]:
        """Expand a user query with relevant entities from the knowledge graph.

        Args:
            query: User search query
            max_entities: Maximum number of entities to include
            entity_similarity_threshold: Minimum similarity for entity matching
            expand_depth: Depth of entity expansion (1 = direct neighbors)

        Returns:
            Dictionary with expanded entities, relations, and context

        Raises:
            GraphRAGError: If query expansion fails

        """

    @abstractmethod
    async def get_contextual_chunks(
        self,
        entities: List[Entity],
        relations: List[Relation],
        max_chunks: int = 20,
        include_related: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get relevant chunks based on entities and relations.

        Args:
            entities: List of relevant entities
            relations: List of relevant relations
            max_chunks: Maximum number of chunks to return
            include_related: Whether to include chunks from related entities

        Returns:
            List of chunk dictionaries with relevance scores

        Raises:
            GraphRAGError: If chunk retrieval fails

        """

    @abstractmethod
    async def semantic_search_with_graph_context(
        self,
        query: str,
        max_results: int = 10,
        use_graph_expansion: bool = True,
        rerank_with_graph: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search enhanced with knowledge graph context.

        Args:
            query: Search query
            max_results: Maximum number of results
            use_graph_expansion: Whether to expand query with graph entities
            rerank_with_graph: Whether to rerank results using graph relevance

        Returns:
            List of search results with graph-enhanced relevance scores

        Raises:
            GraphRAGError: If search fails

        """

    @abstractmethod
    async def generate_entity_summaries(
        self,
        entities: List[Entity],
        include_relations: bool = True,
        max_summary_length: int = 200,
    ) -> Dict[str, str]:
        """Generate natural language summaries for entities.

        Args:
            entities: List of entities to summarize
            include_relations: Whether to include relation information
            max_summary_length: Maximum summary length in characters

        Returns:
            Dictionary mapping entity IDs to their summaries

        Raises:
            GraphRAGError: If summary generation fails

        """

    @abstractmethod
    async def find_entity_clusters(
        self,
        query_entities: List[Entity],
        clustering_threshold: float = 0.8,
        max_cluster_size: int = 15,
    ) -> List[List[Entity]]:
        """Find clusters of related entities for query context.

        Args:
            query_entities: Starting entities from query expansion
            clustering_threshold: Similarity threshold for clustering
            max_cluster_size: Maximum entities per cluster

        Returns:
            List of entity clusters

        Raises:
            GraphRAGError: If clustering fails

        """

    @abstractmethod
    async def compute_entity_relevance_scores(
        self,
        query: str,
        entities: List[Entity],
        use_centrality: bool = True,
        use_semantic_similarity: bool = True,
    ) -> List[Tuple[Entity, float]]:
        """Compute relevance scores for entities given a query.

        Args:
            query: Search query
            entities: List of entities to score
            use_centrality: Whether to include centrality in scoring
            use_semantic_similarity: Whether to include semantic similarity

        Returns:
            List of (entity, relevance_score) tuples sorted by relevance

        Raises:
            GraphRAGError: If scoring fails

        """

    @abstractmethod
    async def generate_query_explanation(
        self,
        original_query: str,
        expanded_entities: List[Entity],
        selected_chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate explanation of how the query was expanded and processed.

        Args:
            original_query: Original user query
            expanded_entities: Entities found during expansion
            selected_chunks: Final selected chunks

        Returns:
            Human-readable explanation of the search process

        Raises:
            GraphRAGError: If explanation generation fails

        """


class GraphRAGError(Exception):
    """Exception raised when GraphRAG operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        """Initialize GraphRAG error.

        Args:
            message: Error description
            operation: Optional operation that failed

        """
        super().__init__(message)
        self.operation = operation
