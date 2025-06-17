"""Utility functions for EDC deduplication service.

This module contains helper functions for the EDC deduplication service
to keep the main service file under the line limit.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import jaconv
import numpy as np

from oboyu.domain.models.knowledge_graph import Entity


class EDCUtils:
    """Utility class for EDC deduplication operations."""

    @staticmethod
    def create_entity_description(entity: Entity, normalized_name: str) -> str:
        """Create comprehensive description for entity embedding.

        Args:
            entity: Entity to describe
            normalized_name: Normalized entity name

        Returns:
            Comprehensive entity description string

        """
        parts = [f"Entity: {normalized_name}"]

        if entity.entity_type:
            parts.append(f"Type: {entity.entity_type}")

        if entity.canonical_name and entity.canonical_name != normalized_name:
            parts.append(f"Canonical Name: {entity.canonical_name}")

        if entity.definition:
            parts.append(f"Definition: {entity.definition}")

        if entity.properties:
            property_strings = []
            for key, value in entity.properties.items():
                if isinstance(value, (str, int, float, bool)):
                    property_strings.append(f"{key}: {value}")
            if property_strings:
                parts.append(f"Properties: {', '.join(property_strings)}")

        return " | ".join(parts)

    @staticmethod
    def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)

        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def are_types_compatible(type1: str, type2: str) -> bool:
        """Check if two entity types are compatible for merging.

        Args:
            type1: First entity type
            type2: Second entity type

        Returns:
            True if types are compatible

        """
        if not type1 or not type2:
            return True

        # Same type
        if type1 == type2:
            return True

        # Compatible type groups
        compatible_groups = [
            {"PERSON", "人物", "PER"},
            {"ORGANIZATION", "ORG", "COMPANY", "組織", "会社"},
            {"LOCATION", "LOC", "PLACE", "場所", "地域"},
            {"MISC", "その他", "MISCELLANEOUS"},
        ]

        for group in compatible_groups:
            if type1.upper() in group and type2.upper() in group:
                return True

        return False

    @staticmethod
    def normalize_for_comparison(name: str) -> str:
        """Normalize entity name for comparison.

        Args:
            name: Original entity name

        Returns:
            Normalized name for comparison

        """
        if not name:
            return ""

        # Convert to hiragana for consistent comparison
        normalized = jaconv.kata2hira(name.lower())

        # Remove common suffixes
        suffixes = [
            r"株式会社$",
            r"有限会社$",
            r"\(株\)$",
            r"会社$",
            r"グループ$",
            r"ホールディングス$",
            r"HD$",
            r"さん$",
            r"氏$",
            r"先生$",
            r"様$",
        ]

        for suffix in suffixes:
            normalized = re.sub(suffix, "", normalized)

        # Remove spaces and punctuation
        normalized = re.sub(r"[^\w]", "", normalized)

        return normalized.strip()

    @staticmethod
    def choose_canonical_name(entities: List[Entity]) -> str:
        """Choose the best canonical name from entity list.

        Args:
            entities: List of entities to choose from

        Returns:
            Best canonical name

        """
        if not entities:
            return ""

        # Prefer entities with existing canonical names
        for entity in sorted(entities, key=lambda e: e.confidence or 0.0, reverse=True):
            if entity.canonical_name:
                return entity.canonical_name

        # Prefer longer, more descriptive names
        candidates = [(entity.name, len(entity.name)) for entity in entities if entity.name]
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]

        return entities[0].name if entities[0].name else ""

    @staticmethod
    def merge_definitions(entities: List[Entity]) -> Optional[str]:
        """Merge entity definitions intelligently.

        Args:
            entities: List of entities to merge definitions from

        Returns:
            Merged definition or None

        """
        definitions = [e.definition for e in entities if e.definition]

        if not definitions:
            return None

        if len(definitions) == 1:
            return definitions[0]

        # Choose the longest, most descriptive definition
        return max(definitions, key=len)

    @staticmethod
    def merge_properties(entities: List[Entity]) -> Dict[str, Any]:
        """Merge entity properties from multiple entities.

        Args:
            entities: List of entities to merge properties from

        Returns:
            Merged properties dictionary

        """
        merged_props: Dict[str, Any] = {}

        for entity in entities:
            if entity.properties:
                for key, value in entity.properties.items():
                    if key not in merged_props:
                        merged_props[key] = value
                    elif merged_props[key] != value:
                        # Handle conflicts by creating a list
                        if not isinstance(merged_props[key], list):
                            merged_props[key] = [merged_props[key]]
                        if value not in merged_props[key]:
                            merged_props[key].append(value)

        return merged_props

    @staticmethod
    def heuristic_verification(entity1: Entity, entity2: Entity, similarity_score: float) -> Tuple[bool, float]:
        """Perform heuristic verification for entity merging.

        Args:
            entity1: First entity
            entity2: Second entity
            similarity_score: Vector similarity score

        Returns:
            Tuple of (should_merge, confidence_score)

        """
        # Start with similarity score as base confidence
        confidence = similarity_score

        # Boost confidence for exact name matches
        if entity1.name and entity2.name:
            name1_norm = EDCUtils.normalize_for_comparison(entity1.name)
            name2_norm = EDCUtils.normalize_for_comparison(entity2.name)

            if name1_norm == name2_norm:
                confidence = min(1.0, confidence + 0.2)
            elif name1_norm in name2_norm or name2_norm in name1_norm:
                confidence = min(1.0, confidence + 0.1)

        # Boost confidence for compatible types
        if EDCUtils.are_types_compatible(entity1.entity_type, entity2.entity_type):
            confidence = min(1.0, confidence + 0.05)
        else:
            confidence = max(0.0, confidence - 0.3)

        # Consider existing confidence scores
        avg_confidence = ((entity1.confidence or 0.5) + (entity2.confidence or 0.5)) / 2
        confidence = confidence * 0.8 + avg_confidence * 0.2

        # Threshold for merging
        should_merge = confidence > 0.7

        return should_merge, confidence
