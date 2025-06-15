"""Tests for enrichment schema validation."""

import pytest
from pydantic import ValidationError

from oboyu.config.enrichment_schema import (
    EnrichmentConfigSchema,
    get_enrichment_config_template,
    validate_enrichment_config,
)


class TestEnrichmentConfigSchema:
    """Test cases for EnrichmentConfigSchema validation."""

    def test_valid_config(self):
        """Test validation of a valid configuration."""
        config = {
            "input_schema": {
                "columns": {
                    "company_name": {"type": "string", "description": "会社名"},
                    "industry": {"type": "string", "description": "業界"}
                },
                "primary_keys": ["company_name"]
            },
            "enrichment_schema": {
                "columns": {
                    "description": {
                        "type": "string",
                        "description": "会社概要",
                        "source_strategy": "search_content",
                        "query_template": "{company_name} 概要"
                    }
                }
            }
        }
        
        schema = EnrichmentConfigSchema.model_validate(config)
        assert schema.input_schema.columns["company_name"].type == "string"
        assert schema.enrichment_schema.columns["description"].source_strategy == "search_content"

    def test_missing_required_sections(self):
        """Test validation fails when required sections are missing."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            }
            # Missing enrichment_schema
        }
        
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfigSchema.model_validate(config)
        
        assert "enrichment_schema" in str(exc_info.value)

    def test_invalid_source_strategy(self):
        """Test validation fails with invalid source strategy."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            },
            "enrichment_schema": {
                "columns": {
                    "test_col": {
                        "type": "string",
                        "description": "Test",
                        "source_strategy": "invalid_strategy",  # Invalid
                        "query_template": "{name}"
                    }
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfigSchema.model_validate(config)
        
        assert "source_strategy" in str(exc_info.value)

    def test_missing_query_template(self):
        """Test validation fails when query_template is missing."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            },
            "enrichment_schema": {
                "columns": {
                    "test_col": {
                        "type": "string",
                        "description": "Test",
                        "source_strategy": "search_content"
                        # Missing query_template
                    }
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfigSchema.model_validate(config)
        
        assert "query_template" in str(exc_info.value)

    def test_pattern_match_without_pattern(self):
        """Test validation fails when pattern_match extraction method lacks pattern."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            },
            "enrichment_schema": {
                "columns": {
                    "test_col": {
                        "type": "string",
                        "description": "Test",
                        "source_strategy": "search_content",
                        "query_template": "{name}",
                        "extraction_method": "pattern_match"
                        # Missing extraction_pattern
                    }
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfigSchema.model_validate(config)
        
        assert "extraction_pattern" in str(exc_info.value)

    def test_graph_relations_validation(self):
        """Test validation for graph_relations strategy."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            },
            "enrichment_schema": {
                "columns": {
                    "test_col": {
                        "type": "string",
                        "description": "Test",
                        "source_strategy": "graph_relations",
                        "query_template": "{name}"
                        # Missing relation_types and target_entity_types
                    }
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfigSchema.model_validate(config)
        
        assert "relation_types" in str(exc_info.value) or "target_entity_types" in str(exc_info.value)

    def test_conflicting_column_names(self):
        """Test validation fails when enrichment columns conflict with input columns."""
        config = {
            "input_schema": {
                "columns": {
                    "name": {"type": "string", "description": "Name"},
                    "description": {"type": "string", "description": "Input description"}
                }
            },
            "enrichment_schema": {
                "columns": {
                    "description": {  # Conflicts with input column
                        "type": "string",
                        "description": "Enriched description",
                        "source_strategy": "search_content",
                        "query_template": "{name}"
                    }
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfigSchema.model_validate(config)
        
        assert "conflict" in str(exc_info.value).lower()

    def test_invalid_primary_key(self):
        """Test validation fails when primary key doesn't exist in columns."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}},
                "primary_keys": ["invalid_key"]  # Doesn't exist in columns
            },
            "enrichment_schema": {
                "columns": {
                    "description": {
                        "type": "string",
                        "description": "Description",
                        "source_strategy": "search_content",
                        "query_template": "{name}"
                    }
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfigSchema.model_validate(config)
        
        assert "Primary key" in str(exc_info.value)

    def test_search_config_defaults(self):
        """Test that search config has proper defaults."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            },
            "enrichment_schema": {
                "columns": {
                    "description": {
                        "type": "string",
                        "description": "Description",
                        "source_strategy": "search_content",
                        "query_template": "{name}"
                    }
                }
            }
        }
        
        schema = EnrichmentConfigSchema.model_validate(config)
        
        # Check defaults
        assert schema.search_config.search_mode == "hybrid"
        assert schema.search_config.use_graphrag == True
        assert schema.search_config.rerank == True
        assert schema.search_config.top_k == 5
        assert schema.search_config.similarity_threshold == 0.5

    def test_custom_search_config(self):
        """Test custom search configuration."""
        config = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            },
            "enrichment_schema": {
                "columns": {
                    "description": {
                        "type": "string",
                        "description": "Description",
                        "source_strategy": "search_content",
                        "query_template": "{name}"
                    }
                }
            },
            "search_config": {
                "search_mode": "vector",
                "use_graphrag": False,
                "rerank": False,
                "top_k": 10,
                "similarity_threshold": 0.7
            }
        }
        
        schema = EnrichmentConfigSchema.model_validate(config)
        
        assert schema.search_config.search_mode == "vector"
        assert schema.search_config.use_graphrag == False
        assert schema.search_config.rerank == False
        assert schema.search_config.top_k == 10
        assert schema.search_config.similarity_threshold == 0.7


class TestValidateEnrichmentConfig:
    """Test cases for the validate_enrichment_config function."""

    def test_valid_config_dict(self):
        """Test validating a valid configuration dictionary."""
        config_dict = {
            "input_schema": {
                "columns": {"name": {"type": "string", "description": "Name"}}
            },
            "enrichment_schema": {
                "columns": {
                    "description": {
                        "type": "string",
                        "description": "Description",
                        "source_strategy": "search_content",
                        "query_template": "{name}"
                    }
                }
            }
        }
        
        schema = validate_enrichment_config(config_dict)
        assert isinstance(schema, EnrichmentConfigSchema)

    def test_invalid_config_dict(self):
        """Test validating an invalid configuration dictionary."""
        config_dict = {"invalid": "config"}
        
        with pytest.raises(ValueError) as exc_info:
            validate_enrichment_config(config_dict)
        
        assert "Invalid enrichment configuration" in str(exc_info.value)


class TestGetEnrichmentConfigTemplate:
    """Test cases for the template generation function."""

    def test_template_structure(self):
        """Test that the template has the correct structure."""
        template = get_enrichment_config_template()
        
        assert "input_schema" in template
        assert "enrichment_schema" in template
        assert "search_config" in template
        
        # Validate the template itself
        schema = validate_enrichment_config(template)
        assert isinstance(schema, EnrichmentConfigSchema)

    def test_template_example_strategies(self):
        """Test that the template includes examples of all strategies."""
        template = get_enrichment_config_template()
        
        enrichment_columns = template["enrichment_schema"]["columns"]
        
        strategies = [col["source_strategy"] for col in enrichment_columns.values()]
        assert "search_content" in strategies
        assert "graph_relations" in strategies

    def test_template_japanese_content(self):
        """Test that the template includes Japanese content examples."""
        template = get_enrichment_config_template()
        
        # Check for Japanese descriptions
        input_columns = template["input_schema"]["columns"]
        assert any("会社" in col["description"] for col in input_columns.values())
        
        enrichment_columns = template["enrichment_schema"]["columns"]
        assert any("概要" in col["query_template"] for col in enrichment_columns.values())