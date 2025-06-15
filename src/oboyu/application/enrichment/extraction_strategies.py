"""Extraction strategies for different types of data enrichment.

This module provides various strategies for extracting information from
the knowledge base to enrich CSV data. Each strategy implements a different
approach to finding and extracting relevant information.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd

from .protocols import GraphRAGService

logger = logging.getLogger(__name__)


class BaseExtractionStrategy(ABC):
    """Base class for extraction strategies."""

    def __init__(
        self,
        graphrag_service: Optional[GraphRAGService] = None,
        max_results: int = 5,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize the extraction strategy.

        Args:
            graphrag_service: GraphRAG service instance
            max_results: Maximum number of search results to consider
            confidence_threshold: Minimum confidence threshold for results

        """
        self.graphrag_service = graphrag_service
        self.max_results = max_results
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    async def extract_value(self, query: str, col_config: Dict[str, Any], row: "pd.Series[Any]") -> Optional[str]:
        """Extract a value using the specific strategy.

        Args:
            query: Formatted query string
            col_config: Column configuration from schema
            row: Current row data

        Returns:
            Extracted value or None if not found

        """
        pass


class SearchContentStrategy(BaseExtractionStrategy):
    """Strategy for extracting content from search results."""

    async def extract_value(self, query: str, col_config: Dict[str, Any], row: "pd.Series[Any]") -> Optional[str]:
        """Extract value from search result content.

        This strategy performs a semantic search and extracts content
        based on the specified extraction method.
        """
        if not self.graphrag_service:
            logger.warning("GraphRAG service not available for search_content strategy")
            return None

        try:
            # Perform semantic search with graph context
            results = await self.graphrag_service.semantic_search_with_graph_context(
                query=query,
                max_results=self.max_results,
                use_graph_expansion=True,
                rerank_with_graph=True,
            )

            if not results:
                return None

            # Filter results by confidence threshold
            filtered_results = [r for r in results if r.get("relevance_score", 0.0) >= self.confidence_threshold]

            if not filtered_results:
                return None

            # Extract content based on extraction method
            extraction_method = col_config.get("extraction_method", "first_result")

            if extraction_method == "summarize":
                return await self._summarize_results(filtered_results)
            elif extraction_method == "first_sentence":
                return self._extract_first_sentence(filtered_results[0]["content"])
            elif extraction_method == "pattern_match":
                pattern = col_config.get("extraction_pattern", r".*")
                return self._extract_by_pattern(filtered_results, pattern)
            else:  # first_result (default)
                return self._extract_first_result(filtered_results[0])

        except Exception as e:
            logger.error(f"Error in search_content strategy: {e}")
            return None

    async def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize multiple search results into a concise description."""
        try:
            # Combine top results (limit to prevent token overflow)
            combined_content = ""
            for result in results[:3]:  # Use top 3 results
                content = result.get("content", "")
                # Take first 200 characters of each result
                combined_content += content[:200] + " "

            # Simple summarization: return first sentence or up to 100 characters
            sentences = combined_content.split("。")
            if sentences and len(sentences[0]) > 10:
                return sentences[0] + "。"
            else:
                return combined_content[:100].strip()

        except Exception as e:
            logger.error(f"Error summarizing results: {e}")
            return results[0].get("content", "")[:100] if results else ""

    def _extract_first_sentence(self, content: str) -> str:
        """Extract the first sentence from content."""
        try:
            # Split by Japanese or English sentence endings
            sentences = re.split(r"[。.!?]", content)
            first_sentence = sentences[0].strip()

            # Add appropriate ending punctuation
            if first_sentence:
                if any(char in first_sentence for char in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"):
                    return first_sentence + "。"
                else:
                    return first_sentence + "."

            return content[:100].strip()

        except Exception as e:
            logger.error(f"Error extracting first sentence: {e}")
            return content[:100].strip()

    def _extract_by_pattern(self, results: List[Dict[str, Any]], pattern: str) -> Optional[str]:
        """Extract content matching a specific regex pattern."""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

            for result in results:
                content = result.get("content", "")
                matches = compiled_pattern.findall(content)
                if matches:
                    return matches[0] if isinstance(matches[0], str) else str(matches[0])

            return None

        except Exception as e:
            logger.error(f"Error extracting by pattern '{pattern}': {e}")
            return None

    def _extract_first_result(self, result: Dict[str, Any]) -> str:
        """Extract content from the first search result."""
        content = result.get("content", "")
        # Return first 200 characters
        return content[:200].strip()


class EntityExtractionStrategy(BaseExtractionStrategy):
    """Strategy for extracting entities from the knowledge graph."""

    async def extract_value(self, query: str, col_config: Dict[str, Any], row: "pd.Series[Any]") -> Optional[str]:
        """Extract value using entity extraction from knowledge graph."""
        if not self.graphrag_service:
            logger.warning("GraphRAG service not available for entity_extraction strategy")
            return None

        try:
            # Expand query to get relevant entities
            expansion_result = await self.graphrag_service.expand_query_with_entities(
                query=query,
                max_entities=10,
                entity_similarity_threshold=self.confidence_threshold,
                expand_depth=1,
            )

            expanded_entities = expansion_result.get("expanded_entities", [])
            if not expanded_entities:
                return None

            # Filter entities by specified types
            entity_types = col_config.get("entity_types", [])
            if entity_types:
                filtered_entities = [item for item in expanded_entities if item["entity"].entity_type in entity_types]
            else:
                filtered_entities = expanded_entities

            if not filtered_entities:
                return None

            # Extract value using pattern matching if specified
            extraction_pattern = col_config.get("extraction_pattern")
            if extraction_pattern:
                return self._extract_entity_by_pattern(filtered_entities, extraction_pattern)
            else:
                # Return the name of the most relevant entity
                most_relevant = max(filtered_entities, key=lambda x: x["relevance_score"])
                return most_relevant["entity"].name

        except Exception as e:
            logger.error(f"Error in entity_extraction strategy: {e}")
            return None

    def _extract_entity_by_pattern(self, entities: List[Dict[str, Any]], pattern: str) -> Optional[str]:
        """Extract entity value using pattern matching."""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)

            for entity_item in entities:
                entity_name = entity_item["entity"].name
                matches = compiled_pattern.findall(entity_name)
                if matches:
                    return matches[0] if isinstance(matches[0], str) else str(matches[0])

            return None

        except Exception as e:
            logger.error(f"Error extracting entity by pattern '{pattern}': {e}")
            return None


class GraphRelationsStrategy(BaseExtractionStrategy):
    """Strategy for extracting information through graph relations."""

    async def extract_value(self, query: str, col_config: Dict[str, Any], row: "pd.Series[Any]") -> Optional[str]:
        """Extract value by following graph relations."""
        if not self.graphrag_service:
            logger.warning("GraphRAG service not available for graph_relations strategy")
            return None

        try:
            # First, expand query to get relevant entities
            expansion_result = await self.graphrag_service.expand_query_with_entities(
                query=query,
                max_entities=5,
                entity_similarity_threshold=self.confidence_threshold,
                expand_depth=1,
            )

            expanded_entities = [item["entity"] for item in expansion_result.get("expanded_entities", [])]

            if not expanded_entities:
                return None

            # Get relation types to follow
            relation_types = col_config.get("relation_types", [])
            target_entity_types = col_config.get("target_entity_types", [])

            # Follow relations to find target entities
            kg_repository = self.graphrag_service.kg_repository

            for entity in expanded_entities:
                # Get relations for this entity
                relations = await kg_repository.get_entity_relations(entity.id)

                # Filter by relation types if specified
                if relation_types:
                    relations = [r for r in relations if r.relation_type in relation_types]

                # Get target entities from relations
                for relation in relations:
                    target_entity = await kg_repository.get_entity_by_id(relation.target_id)

                    if target_entity:
                        # Filter by target entity types if specified
                        if not target_entity_types or target_entity.entity_type in target_entity_types:
                            return target_entity.name

            return None

        except Exception as e:
            logger.error(f"Error in graph_relations strategy: {e}")
            return None
