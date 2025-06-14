"""Entity analysis utilities for GraphRAG.

This module provides utilities for analyzing entities, generating summaries,
and finding entity clusters for GraphRAG operations.
"""

import logging
from typing import Any, Dict, List

from oboyu.domain.models.knowledge_graph import Entity
from oboyu.ports.repositories.kg_repository import KGRepository
from oboyu.ports.services.graphrag_service import GraphRAGError

from .query_expansion import QueryExpansionHelper

logger = logging.getLogger(__name__)


class EntityAnalysisHelper:
    """Helper class for entity analysis operations."""

    def __init__(
        self,
        kg_repository: KGRepository,
        query_helper: QueryExpansionHelper,
    ) -> None:
        """Initialize entity analysis helper.

        Args:
            kg_repository: Knowledge graph repository
            query_helper: Query expansion helper for text similarity

        """
        self.kg_repository = kg_repository
        self.query_helper = query_helper

    async def generate_entity_summaries(
        self,
        entities: List[Entity],
        include_relations: bool = True,
        max_summary_length: int = 200,
    ) -> Dict[str, str]:
        """Generate natural language summaries for entities."""
        try:
            summaries = {}

            for entity in entities:
                summary_parts = [f"{entity.name}は{entity.entity_type}です。"]

                # Add definition if available
                if entity.definition:
                    summary_parts.append(entity.definition)

                # Add properties information
                if entity.properties:
                    for key, value in list(entity.properties.items())[:3]:  # Limit properties
                        if isinstance(value, str) and len(value) < 50:
                            summary_parts.append(f"{key}: {value}")

                # Add relation information if requested
                if include_relations:
                    try:
                        relations = await self.query_helper.get_entity_relations(entity.id)
                        if relations:
                            relation_count = len(relations)
                            relation_types = list(set(r.relation_type for r in relations[:5]))
                            summary_parts.append(f"{relation_count}個の関係があり、主な関係タイプは{', '.join(relation_types)}です。")
                    except Exception as e:
                        logger.debug(f"Failed to get relations for entity {entity.name}: {e}")

                # Combine and truncate summary
                full_summary = " ".join(summary_parts)
                if len(full_summary) > max_summary_length:
                    full_summary = full_summary[: max_summary_length - 3] + "..."

                summaries[entity.id] = full_summary

            return summaries

        except Exception as e:
            logger.error(f"Failed to generate entity summaries: {e}")
            raise GraphRAGError(f"Entity summary generation failed: {e}", "generate_entity_summaries")

    async def find_entity_clusters(
        self,
        query_entities: List[Entity],
        clustering_threshold: float = 0.8,
        max_cluster_size: int = 15,
    ) -> List[List[Entity]]:
        """Find clusters of related entities for query context."""
        try:
            if not query_entities:
                return []

            clusters = []
            processed_entities = set()

            for seed_entity in query_entities:
                if seed_entity.id in processed_entities:
                    continue

                # Start new cluster with seed entity
                cluster = [seed_entity]
                processed_entities.add(seed_entity.id)

                # Find related entities using property graph
                try:
                    related_entities = await self.kg_repository.get_entity_neighbors(seed_entity.id, max_hops=1)

                    # Filter related entities by similarity
                    for related_entity in related_entities:
                        if related_entity.id in processed_entities:
                            continue

                        if len(cluster) >= max_cluster_size:
                            break

                        # Compute similarity (simplified clustering)
                        similarity = self.query_helper.compute_text_similarity(seed_entity.name, related_entity.name)

                        if similarity >= clustering_threshold:
                            cluster.append(related_entity)
                            processed_entities.add(related_entity.id)

                except Exception as e:
                    logger.warning(f"Failed to get neighbors for entity {seed_entity.name}: {e}")

                # Add cluster if it has multiple entities
                if len(cluster) > 1:
                    clusters.append(cluster)

            logger.info(f"Found {len(clusters)} entity clusters")
            return clusters

        except Exception as e:
            logger.error(f"Failed to find entity clusters: {e}")
            raise GraphRAGError(f"Entity clustering failed: {e}", "find_entity_clusters")

    def generate_query_explanation(
        self,
        original_query: str,
        expanded_entities: List[Entity],
        selected_chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate explanation of how the query was expanded and processed."""
        try:
            explanation_parts = [f"クエリ「{original_query}」の処理説明:"]

            # Entity expansion explanation
            if expanded_entities:
                entity_names = [entity.name for entity in expanded_entities[:5]]
                explanation_parts.append(f"知識グラフから {len(expanded_entities)} 個の関連エンティティを発見: {', '.join(entity_names)}")

            # Chunk selection explanation
            if selected_chunks:
                explanation_parts.append(f"{len(selected_chunks)} 個の関連文書チャンクを選択しました。")

                # Highlight chunks with high graph relevance
                graph_enhanced_chunks = [chunk for chunk in selected_chunks if chunk.get("search_type") == "graph_enhanced" and chunk.get("graph_entities")]

                if graph_enhanced_chunks:
                    explanation_parts.append(f"そのうち {len(graph_enhanced_chunks)} 個は知識グラフの情報で強化されています。")

            # Overall process explanation
            explanation_parts.append("この結果は、従来のセマンティック検索に知識グラフからの文脈情報を組み合わせて生成されました。")

            return " ".join(explanation_parts)

        except Exception as e:
            logger.error(f"Failed to generate query explanation: {e}")
            return f"クエリ「{original_query}」を処理しましたが、詳細な説明の生成に失敗しました。"
