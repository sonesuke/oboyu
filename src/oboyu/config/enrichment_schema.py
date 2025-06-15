"""Schema validation for CSV enrichment configuration.

This module provides Pydantic models for validating enrichment schema
configurations, ensuring proper structure and valid strategy configurations.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class InputColumnSchema(BaseModel):
    """Schema for input CSV column definition."""

    type: Literal["string", "integer", "float", "boolean"] = Field(description="Data type of the column")
    description: str = Field(description="Human-readable description of the column")
    required: bool = Field(default=True, description="Whether this column is required in the input CSV")


class InputSchema(BaseModel):
    """Schema for input CSV structure definition."""

    columns: Dict[str, InputColumnSchema] = Field(description="Dictionary of column name to column schema")
    primary_keys: List[str] = Field(default_factory=list, description="List of column names that serve as primary keys")

    @field_validator("primary_keys")
    @classmethod
    def validate_primary_keys(cls, v: List[str], info: ValidationInfo) -> List[str]:
        """Validate that primary keys exist in columns."""
        if hasattr(info, "data") and "columns" in info.data:
            columns = info.data["columns"]
            for key in v:
                if key not in columns:
                    raise ValueError(f"Primary key '{key}' not found in columns")
        return v


class EnrichmentColumnSchema(BaseModel):
    """Schema for enrichment column configuration."""

    type: Literal["string", "integer", "float", "boolean"] = Field(description="Data type of the enriched column")
    description: str = Field(description="Human-readable description of what this column contains")
    source_strategy: Literal["search_content", "entity_extraction", "graph_relations"] = Field(description="Strategy to use for extracting the value")
    query_template: str = Field(description="Template for generating search queries with placeholders")
    extraction_method: Optional[Literal["first_result", "first_sentence", "summarize", "pattern_match"]] = Field(
        default="first_result", description="Method for extracting content from search results"
    )
    extraction_pattern: Optional[str] = Field(default=None, description="Regex pattern for pattern_match extraction method")
    entity_types: Optional[List[str]] = Field(default_factory=list, description="List of entity types to filter for entity_extraction strategy")
    relation_types: Optional[List[str]] = Field(default_factory=list, description="List of relation types to follow for graph_relations strategy")
    target_entity_types: Optional[List[str]] = Field(default_factory=list, description="List of target entity types for graph_relations strategy")

    @model_validator(mode="after")
    def validate_strategy_config(self) -> "EnrichmentColumnSchema":
        """Validate configuration based on selected strategy."""
        if self.source_strategy == "entity_extraction" and not self.entity_types:
            # Allow empty entity_types for entity_extraction - will use all types
            pass

        if self.source_strategy == "graph_relations":
            if not self.relation_types and not self.target_entity_types:
                raise ValueError("graph_relations strategy requires either relation_types or target_entity_types")

        if self.extraction_method == "pattern_match" and not self.extraction_pattern:
            raise ValueError("pattern_match extraction method requires extraction_pattern")

        return self


class EnrichmentSchema(BaseModel):
    """Schema for enrichment column definitions."""

    columns: Dict[str, EnrichmentColumnSchema] = Field(description="Dictionary of enrichment column name to column schema")


class SearchConfigSchema(BaseModel):
    """Schema for search configuration."""

    search_mode: Literal["vector", "bm25", "hybrid"] = Field(default="hybrid", description="Search mode to use")
    use_graphrag: bool = Field(default=True, description="Whether to use GraphRAG enhancement")
    rerank: bool = Field(default=True, description="Whether to apply reranking to results")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum number of search results to retrieve")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold for results")


class EnrichmentConfigSchema(BaseModel):
    """Complete schema for enrichment configuration."""

    input_schema: InputSchema = Field(description="Schema defining the structure of input CSV data")
    enrichment_schema: EnrichmentSchema = Field(description="Schema defining columns to be enriched")
    search_config: Optional[SearchConfigSchema] = Field(default_factory=SearchConfigSchema, description="Configuration for search behavior")

    @field_validator("enrichment_schema")
    @classmethod
    def validate_enrichment_columns_unique(cls, v: EnrichmentSchema, info: ValidationInfo) -> EnrichmentSchema:
        """Validate that enrichment columns don't conflict with input columns."""
        if hasattr(info, "data") and "input_schema" in info.data:
            input_columns = set(info.data["input_schema"].columns.keys())
            enrichment_columns = set(v.columns.keys())

            conflicts = input_columns & enrichment_columns
            if conflicts:
                raise ValueError(f"Enrichment columns conflict with input columns: {conflicts}")

        return v


def validate_enrichment_config(config_dict: Dict[str, Any]) -> EnrichmentConfigSchema:
    """Validate and parse enrichment configuration dictionary.

    Args:
        config_dict: Dictionary containing enrichment configuration

    Returns:
        Validated EnrichmentConfigSchema instance

    Raises:
        ValueError: If configuration is invalid

    """
    try:
        return EnrichmentConfigSchema.model_validate(config_dict)
    except Exception as e:
        raise ValueError(f"Invalid enrichment configuration: {e}")


def get_enrichment_config_template() -> Dict[str, Any]:
    """Get a template for enrichment configuration.

    Returns:
        Dictionary containing a complete example configuration

    """
    return {
        "input_schema": {
            "columns": {
                "company_name": {"type": "string", "description": "会社名", "required": True},
                "industry": {"type": "string", "description": "業界", "required": False},
            },
            "primary_keys": ["company_name"],
        },
        "enrichment_schema": {
            "columns": {
                "description": {
                    "type": "string",
                    "description": "会社の概要",
                    "source_strategy": "search_content",
                    "query_template": "{company_name} 概要 事業内容",
                    "extraction_method": "summarize",
                },
                "employees": {
                    "type": "integer",
                    "description": "従業員数",
                    "source_strategy": "search_content",
                    "query_template": "{company_name} 従業員数",
                    "extraction_method": "pattern_match",
                    "extraction_pattern": r"(\d+)(?:人|名)",
                },
                "founded_year": {
                    "type": "integer",
                    "description": "設立年",
                    "source_strategy": "graph_relations",
                    "query_template": "{company_name} 設立",
                    "relation_types": ["FOUNDED_IN", "ESTABLISHED_IN"],
                    "target_entity_types": ["DATE", "YEAR"],
                },
            }
        },
        "search_config": {"search_mode": "hybrid", "use_graphrag": True, "rerank": True, "top_k": 5, "similarity_threshold": 0.5},
    }
