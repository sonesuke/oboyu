"""Oboyu-specific GraphRAG service implementation.

This module implements GraphRAG operations specifically for the Oboyu system,
integrating the knowledge graph with existing semantic search capabilities.
"""

import logging
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer

from oboyu.domain.models.knowledge_graph import Entity, Relation
from oboyu.ports.repositories.kg_repository import KGRepository
from oboyu.ports.services.graphrag_service import GraphRAGError, GraphRAGService
from oboyu.ports.services.property_graph_service import PropertyGraphService

from .chunk_retrieval import ChunkRetrievalHelper
from .entity_analysis import EntityAnalysisHelper
from .query_expansion import QueryExpansionHelper

logger = logging.getLogger(__name__)


class OboyuGraphRAGService(GraphRAGService):
    """Oboyu-specific implementation of GraphRAG service."""

    def __init__(
        self,
        kg_repository: KGRepository,
        property_graph_service: PropertyGraphService,
        embedding_model: SentenceTransformer,
        database_connection: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Oboyu GraphRAG service.

        Args:
            kg_repository: Knowledge graph repository
            property_graph_service: Property graph service for advanced queries
            embedding_model: Sentence transformer for semantic similarity
            database_connection: Database connection for chunk queries

        """
        self.kg_repository = kg_repository
        self.property_graph_service = property_graph_service
        self.embedding_model = embedding_model
        self.db_connection = database_connection
        self.query_helper = QueryExpansionHelper(kg_repository, embedding_model)
        self.chunk_helper = ChunkRetrievalHelper(kg_repository, database_connection)
        self.entity_helper = EntityAnalysisHelper(kg_repository, self.query_helper)

    async def expand_query_with_entities(
        self,
        query: str,
        max_entities: int = 10,
        entity_similarity_threshold: float = 0.7,
        expand_depth: int = 1,
    ) -> Dict[str, Any]:
        """Expand a user query with relevant entities from the knowledge graph."""
        try:
            logger.info(f"Expanding query: '{query}' with max_entities={max_entities}")

            # Extract potential entity names from query using NLP patterns
            potential_entities = self.query_helper.extract_entity_candidates(query)

            # Search for matching entities in the knowledge graph
            matched_entities = []
            for candidate in potential_entities:
                # Search by name pattern
                similar_entities = await self.kg_repository.search_entities_by_name(candidate, limit=5)

                # Filter by similarity threshold
                for entity in similar_entities:
                    similarity = self.query_helper.compute_text_similarity(candidate, entity.name)
                    if similarity >= entity_similarity_threshold:
                        matched_entities.append((entity, similarity))

            # Sort by similarity and take top entities
            matched_entities.sort(key=lambda x: x[1], reverse=True)
            top_entities = [entity for entity, _ in matched_entities[:max_entities]]

            # Expand entities with neighbors if depth > 0
            expanded_entities = set(top_entities)
            relations = []

            if expand_depth > 0 and top_entities:
                for entity in top_entities:
                    try:
                        # Get entity neighbors
                        neighbors = await self.kg_repository.get_entity_neighbors(entity.id, max_hops=expand_depth)
                        expanded_entities.update(neighbors[:3])  # Limit neighbors per entity

                        # Get relations involving this entity
                        entity_relations = await self.query_helper.get_entity_relations(entity.id)
                        relations.extend(entity_relations[:5])  # Limit relations per entity

                    except Exception as e:
                        logger.warning(f"Failed to expand entity {entity.name}: {e}")

            # Get centrality scores if available
            centrality_scores = {}
            if await self.property_graph_service.is_property_graph_available():
                try:
                    all_centrality = await self.property_graph_service.get_entity_centrality_scores(limit=100)
                    centrality_scores = {entity_id: score for entity_id, _, score in all_centrality}
                except Exception as e:
                    logger.warning(f"Failed to get centrality scores: {e}")

            # Compute relevance scores for final entity set
            final_entities = list(expanded_entities)
            entity_scores = await self.query_helper.compute_entity_relevance_scores(query, final_entities, centrality_scores)

            # Sort by relevance
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            scored_entities = entity_scores[:max_entities]

            result = {
                "original_query": query,
                "extracted_candidates": potential_entities,
                "matched_entities": len(matched_entities),
                "expanded_entities": [{"entity": entity, "relevance_score": score} for entity, score in scored_entities],
                "relations": relations,
                "expansion_depth": expand_depth,
            }

            logger.info(f"Query expansion complete: found {len(scored_entities)} relevant entities")
            return result

        except Exception as e:
            logger.error(f"Failed to expand query '{query}': {e}")
            raise GraphRAGError(f"Query expansion failed: {e}", "expand_query_with_entities")

    async def get_contextual_chunks(
        self,
        entities: List[Entity],
        relations: List[Relation],
        max_chunks: int = 20,
        include_related: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get relevant chunks based on entities and relations."""
        return await self.chunk_helper.get_contextual_chunks(entities, relations, max_chunks, include_related)

    async def semantic_search_with_graph_context(
        self,
        query: str,
        max_results: int = 10,
        use_graph_expansion: bool = True,
        rerank_with_graph: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search enhanced with knowledge graph context."""
        try:
            logger.info(f"Performing GraphRAG search for: '{query}'")

            results = []

            if use_graph_expansion:
                # Expand query with knowledge graph entities
                expansion_result = await self.expand_query_with_entities(query, max_entities=8, expand_depth=1)

                expanded_entities = [item["entity"] for item in expansion_result["expanded_entities"]]
                relations = expansion_result["relations"]

                # Get contextual chunks from knowledge graph
                contextual_chunks = await self.chunk_helper.get_contextual_chunks(expanded_entities, relations, max_chunks=max_results * 2)

                # Add graph context to results
                for chunk in contextual_chunks:
                    # Compute relevance score based on entity matches and semantic similarity
                    relevance_score = self.query_helper.compute_chunk_relevance(query, chunk, expanded_entities)

                    result_item = {
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "relevance_score": relevance_score,
                        "graph_entities": chunk["related_entities"],
                        "graph_relations": chunk["related_relations"],
                        "search_type": "graph_enhanced",
                    }
                    results.append(result_item)

            # If graph expansion didn't find enough results, fall back to traditional search
            if len(results) < max_results:
                # TODO: Integrate with existing semantic search when available
                logger.info(f"Graph search found {len(results)} results, may need traditional search fallback")

            # Sort by relevance score
            results.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Rerank with graph context if requested
            if rerank_with_graph and results:
                results = await self._rerank_with_graph_centrality(query, results)

            final_results = results[:max_results]
            logger.info(f"GraphRAG search complete: returning {len(final_results)} results")

            return final_results

        except Exception as e:
            logger.error(f"GraphRAG search failed for query '{query}': {e}")
            raise GraphRAGError(f"GraphRAG search failed: {e}", "semantic_search_with_graph_context")

    async def _rerank_with_graph_centrality(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using graph centrality scores."""
        try:
            if not await self.property_graph_service.is_property_graph_available():
                return results

            # Get centrality scores for entities in results
            entity_ids = set()
            for result in results:
                for entity_info in result.get("graph_entities", []):
                    entity_ids.add(entity_info["id"])

            if not entity_ids:
                return results

            # Calculate centrality scores
            centrality_scores = {}
            try:
                # Get degree centrality for all entities
                all_centrality = await self.property_graph_service.get_entity_centrality_scores(limit=len(entity_ids) * 2)
                centrality_scores = {entity_id: score for entity_id, _, score in all_centrality}
            except Exception as e:
                logger.warning(f"Failed to get centrality scores: {e}")

            # Adjust relevance scores based on centrality
            for result in results:
                centrality_boost = 0.0
                for entity_info in result.get("graph_entities", []):
                    entity_id = entity_info["id"]
                    if entity_id in centrality_scores:
                        # Normalize centrality score and add as boost
                        centrality_boost += min(0.1, centrality_scores[entity_id] / 10.0)

                # Add centrality boost to relevance score
                result["relevance_score"] = min(1.0, result["relevance_score"] + centrality_boost)

            # Re-sort by updated scores
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results

        except Exception as e:
            logger.warning(f"Failed to rerank with graph centrality: {e}")
            return results

    async def generate_entity_summaries(
        self,
        entities: List[Entity],
        include_relations: bool = True,
        max_summary_length: int = 200,
    ) -> Dict[str, str]:
        """Generate natural language summaries for entities."""
        return await self.entity_helper.generate_entity_summaries(entities, include_relations, max_summary_length)

    async def find_entity_clusters(
        self,
        query_entities: List[Entity],
        clustering_threshold: float = 0.8,
        max_cluster_size: int = 15,
    ) -> List[List[Entity]]:
        """Find clusters of related entities for query context."""
        return await self.entity_helper.find_entity_clusters(query_entities, clustering_threshold, max_cluster_size)

    async def compute_entity_relevance_scores(
        self,
        query: str,
        entities: List[Entity],
        use_centrality: bool = True,
        use_semantic_similarity: bool = True,
    ) -> List[Tuple[Entity, float]]:
        """Compute relevance scores for entities given a query."""
        try:
            # Get centrality scores if requested
            centrality_scores = {}
            if use_centrality and await self.property_graph_service.is_property_graph_available():
                try:
                    all_centrality = await self.property_graph_service.get_entity_centrality_scores(limit=len(entities) * 2)
                    centrality_scores = {entity_id: score for entity_id, _, score in all_centrality}
                except Exception as e:
                    logger.warning(f"Failed to get centrality scores: {e}")

            return await self.query_helper.compute_entity_relevance_scores(query, entities, centrality_scores, use_centrality, use_semantic_similarity)

        except Exception as e:
            logger.error(f"Failed to compute entity relevance scores: {e}")
            raise GraphRAGError(f"Entity relevance scoring failed: {e}", "compute_entity_relevance_scores")

    async def generate_query_explanation(
        self,
        original_query: str,
        expanded_entities: List[Entity],
        selected_chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate explanation of how the query was expanded and processed."""
        return self.entity_helper.generate_query_explanation(original_query, expanded_entities, selected_chunks)
