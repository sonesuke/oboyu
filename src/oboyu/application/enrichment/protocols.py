"""Protocols for enrichment services."""

from typing import Any, Dict, List, Protocol


class GraphRAGService(Protocol):
    """Protocol for GraphRAG service interface."""

    async def semantic_search_with_graph_context(
        self,
        query: str,
        max_results: int = 10,
        use_graph_expansion: bool = True,
        rerank_with_graph: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search with graph context."""
        ...

    async def expand_query_with_entities(
        self,
        query: str,
        max_entities: int = 10,
        entity_similarity_threshold: float = 0.7,
        expand_depth: int = 1,
    ) -> Dict[str, Any]:
        """Expand query with entities."""
        ...

    kg_repository: Any  # KG repository instance
