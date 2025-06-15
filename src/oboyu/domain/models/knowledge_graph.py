"""Domain models for Knowledge Graph entities and relations.

This module defines the core domain models for the Property Graph Index functionality,
including entities, relations, and extraction results from LLM processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class Entity:
    """Represents a knowledge graph entity."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    entity_type: str = ""
    definition: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    canonical_name: Optional[str] = None
    merged_from: List[str] = field(default_factory=list)
    merge_confidence: Optional[float] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """Make Entity hashable for use in sets and dictionaries."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on entity ID."""
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class Relation:
    """Represents a knowledge graph relation between entities."""

    id: str = field(default_factory=lambda: str(uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """Make Relation hashable for use in sets and dictionaries."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on relation ID."""
        if not isinstance(other, Relation):
            return False
        return self.id == other.id


@dataclass
class KnowledgeGraphExtraction:
    """Result of knowledge graph extraction from a text chunk."""

    chunk_id: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    processing_time_ms: Optional[int] = None
    model_used: Optional[str] = None
    confidence: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ProcessingStatus:
    """Status of knowledge graph processing for a chunk."""

    chunk_id: str
    processed_at: datetime = field(default_factory=datetime.now)
    processing_version: str = "1.0.0"
    entity_count: int = 0
    relation_count: int = 0
    processing_time_ms: Optional[int] = None
    model_used: Optional[str] = None
    error_message: Optional[str] = None
    status: str = "completed"  # completed, error, in_progress
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# Predefined entity types for business-oriented knowledge graphs
ENTITY_TYPES = {
    "PERSON": "人物",
    "ORGANIZATION": "組織・団体",
    "COMPANY": "企業",
    "PRODUCT": "製品・サービス",
    "LOCATION": "場所・地域",
    "EVENT": "イベント・出来事",
    "POSITION": "役職・ポジション",
    "TECHNOLOGY": "技術・テクノロジー",
    "CONCEPT": "概念・アイデア",
    "DATE": "日付・時期",
}

# Predefined relation types for business relationships
RELATION_TYPES = {
    # Person-Organization relationships
    "WORKS_AT": "勤務している",
    "CEO_OF": "CEOである",
    "FOUNDER_OF": "創設者である",
    "BOARD_MEMBER_OF": "取締役である",
    "EMPLOYEE_OF": "従業員である",
    # Organization-Organization relationships
    "SUBSIDIARY_OF": "子会社である",
    "PARENT_OF": "親会社である",
    "PARTNER_WITH": "パートナーである",
    "COMPETES_WITH": "競合している",
    "ACQUIRES": "買収している",
    "MERGED_WITH": "合併している",
    # Product/Service relationships
    "DEVELOPS": "開発している",
    "PROVIDES": "提供している",
    "USES": "使用している",
    "MANUFACTURES": "製造している",
    "SELLS": "販売している",
    # Location relationships
    "LOCATED_IN": "所在している",
    "OPERATES_IN": "事業展開している",
    "HEADQUARTERED_IN": "本社がある",
    # Event relationships
    "PARTICIPATES_IN": "参加している",
    "ORGANIZES": "主催している",
    "SPONSORS": "スポンサーである",
    # Temporal relationships
    "FOUNDED_IN": "設立された",
    "ESTABLISHED_IN": "設立された",
    "LAUNCHED_IN": "ローンチされた",
}
