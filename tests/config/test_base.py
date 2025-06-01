"""Tests for unified configuration management."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oboyu.common.config import ConfigManager
from oboyu.common.config_schema import (
    ConfigSchema,
    CrawlerConfigSchema,
    IndexerConfigSchema,
    QueryConfigSchema,
)
from oboyu.common.paths import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_init_with_default_path(self):
        """Test initialization with default config path."""
        manager = ConfigManager()
        assert manager.config_path == DEFAULT_CONFIG_PATH

    def test_init_with_custom_path(self):
        """Test initialization with custom config path."""
        custom_path = Path("/custom/path/config.yaml")
        manager = ConfigManager(custom_path)
        assert manager.config_path == custom_path

    def test_build_defaults(self):
        """Test default configuration building."""
        manager = ConfigManager()
        defaults = manager._build_defaults()
        
        assert "crawler" in defaults
        assert "indexer" in defaults
        assert "query" in defaults
        
        # Check some specific defaults
        assert defaults["query"]["top_k"] == 10
        assert defaults["query"]["rerank"] is True
        assert defaults["query"]["rerank_model"] == "cl-nagoya/ruri-reranker-small"

    def test_load_config_with_no_file(self):
        """Test loading config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yaml"
            manager = ConfigManager(config_path)
            
            config = manager.load_config()
            
            # Should return defaults
            assert "crawler" in config
            assert "indexer" in config
            assert "query" in config

    def test_load_config_with_file(self):
        """Test loading config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            # Create test config
            test_config = {
                "indexer": {
                    "db_path": "/test/db/path",
                    "batch_size": 64,
                },
                "query": {
                    "top_k": 20,
                }
            }
            
            with open(config_path, "w") as f:
                yaml.dump(test_config, f)
            
            manager = ConfigManager(config_path)
            config = manager.load_config()
            
            # Check merged values
            assert config["indexer"]["db_path"] == "/test/db/path"
            assert config["indexer"]["batch_size"] == 64
            assert config["query"]["top_k"] == 20
            
            # Check defaults are still present
            assert "crawler" in config
            assert config["query"]["rerank"] is True  # Default value

    def test_load_config_with_invalid_yaml(self):
        """Test loading config with invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            
            # Create invalid YAML
            with open(config_path, "w") as f:
                f.write("invalid: yaml: content: [")
            
            manager = ConfigManager(config_path)
            
            # Should warn and use defaults
            with pytest.warns(UserWarning):
                config = manager.load_config()
            
            assert "crawler" in config
            assert "indexer" in config
            assert "query" in config

    def test_get_section(self):
        """Test getting a configuration section."""
        manager = ConfigManager()
        
        crawler_config = manager.get_section("crawler")
        assert isinstance(crawler_config, dict)
        assert "max_workers" in crawler_config
        
        indexer_config = manager.get_section("indexer")
        assert isinstance(indexer_config, dict)
        assert "embedding_model" in indexer_config
        
        query_config = manager.get_section("query")
        assert isinstance(query_config, dict)
        assert "top_k" in query_config

    def test_get_section_returns_copy(self):
        """Test that get_section returns a copy, not reference."""
        manager = ConfigManager()
        
        config1 = manager.get_section("query")
        config2 = manager.get_section("query")
        
        # Modify one copy
        config1["top_k"] = 999
        
        # Other copy should be unchanged
        assert config2["top_k"] != 999

    def test_merge_cli_overrides(self):
        """Test merging CLI overrides with configuration."""
        manager = ConfigManager()
        
        overrides = {
            "top_k": 15,
            "rerank": False,
            "new_option": "value",
        }
        
        merged = manager.merge_cli_overrides("query", overrides)
        
        assert merged["top_k"] == 15
        assert merged["rerank"] is False
        assert merged["new_option"] == "value"
        assert merged["rerank_model"] == "cl-nagoya/ruri-reranker-small"  # Default

    def test_merge_cli_overrides_filters_none(self):
        """Test that None values are filtered from overrides."""
        manager = ConfigManager()
        
        overrides = {
            "top_k": 15,
            "rerank": None,
            "show_scores": None,
        }
        
        merged = manager.merge_cli_overrides("query", overrides)
        
        assert merged["top_k"] == 15
        assert merged["rerank"] is True  # Default, not overridden
        assert merged["show_scores"] is False  # Default, not overridden

    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "save_test.yaml"
            manager = ConfigManager(config_path)
            
            # Modify config
            config = manager.load_config()
            config["query"]["top_k"] = 25
            
            # Save it
            manager.save_config(config)
            
            # Load it again
            with open(config_path) as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config["query"]["top_k"] == 25

    def test_save_config_creates_directory(self):
        """Test that save_config creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.yaml"
            manager = ConfigManager(config_path)
            
            manager.save_config()
            
            assert config_path.exists()
            assert config_path.parent.exists()

    def test_resolve_db_path_precedence(self):
        """Test database path resolution precedence."""
        manager = ConfigManager()
        
        # Test CLI override takes precedence
        cli_path = Path("/cli/db/path")
        resolved = manager.resolve_db_path(cli_path, {"db_path": "/config/path"})
        assert resolved == cli_path
        
        # Test config value when no CLI override
        resolved = manager.resolve_db_path(None, {"db_path": "/config/path"})
        assert resolved == Path("/config/path")
        
        # Test default when neither provided
        resolved = manager.resolve_db_path(None, {})
        assert resolved == DEFAULT_DB_PATH

    def test_get_schema(self):
        """Test getting typed configuration schema."""
        manager = ConfigManager()
        schema = manager.get_schema()
        
        assert isinstance(schema, ConfigSchema)
        assert isinstance(schema.crawler, CrawlerConfigSchema)
        assert isinstance(schema.indexer, IndexerConfigSchema)
        assert isinstance(schema.query, QueryConfigSchema)

    def test_get_section_schema(self):
        """Test getting typed section schema."""
        manager = ConfigManager()
        
        crawler_schema = manager.get_section_schema("crawler")
        assert isinstance(crawler_schema, CrawlerConfigSchema)
        assert crawler_schema.max_workers == 4
        
        indexer_schema = manager.get_section_schema("indexer")
        assert isinstance(indexer_schema, IndexerConfigSchema)
        assert indexer_schema.embedding_model == "cl-nagoya/ruri-v3-30m"
        
        query_schema = manager.get_section_schema("query")
        assert isinstance(query_schema, QueryConfigSchema)
        assert query_schema.top_k == 10

    def test_get_section_schema_invalid(self):
        """Test getting schema for invalid section."""
        manager = ConfigManager()
        
        with pytest.raises(ValueError, match="Invalid configuration section"):
            manager.get_section_schema("invalid_section")

    def test_config_persistence(self):
        """Test that config is loaded only once and cached."""
        manager = ConfigManager()
        
        # First load
        config1 = manager.load_config()
        
        # Second load should return same instance
        config2 = manager.load_config()
        
        # They should be the same object
        assert config1 is config2