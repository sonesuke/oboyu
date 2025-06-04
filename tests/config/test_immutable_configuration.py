"""Tests for immutable configuration system."""

import pytest
import logging
from unittest.mock import Mock

from oboyu.config import (
    ConfigurationBuilder, 
    ConfigurationResolver,
    ConfigSource,
    ConfigValue,
    SearchContext,
)
from oboyu.common.types import SearchMode


class TestConfigurationBuilder:
    """Test the ConfigurationBuilder class."""
    
    def test_configuration_builder_precedence(self):
        """Test that configuration sources follow correct precedence."""
        builder = ConfigurationBuilder()
        
        # Set values in different order to test precedence
        builder.set_default("test.key", "default_value")
        builder.set_from_file("test.key", "file_value")
        builder.set_from_env("test.key", "env_value")
        builder.set_from_cli("test.key", "cli_value")
        
        # CLI should win (highest precedence)
        assert builder.get("test.key") == "cli_value"
        
        config_value = builder.get_with_source("test.key")
        assert config_value.source == ConfigSource.CLI
        assert config_value.value == "cli_value"
    
    def test_configuration_builder_partial_precedence(self):
        """Test precedence when not all sources are set."""
        builder = ConfigurationBuilder()
        
        # Only set file and default
        builder.set_default("test.key", "default_value")
        builder.set_from_file("test.key", "file_value")
        
        # File should win over default
        assert builder.get("test.key") == "file_value"
        
        config_value = builder.get_with_source("test.key")
        assert config_value.source == ConfigSource.FILE
    
    def test_explicit_value_detection(self):
        """Test detection of explicit vs default values."""
        builder = ConfigurationBuilder()
        
        builder.set_default("default.key", "default_value")
        builder.set_from_cli("explicit.key", "explicit_value")
        
        assert not builder.has_explicit_value("default.key")
        assert builder.has_explicit_value("explicit.key")
    
    def test_configuration_merge(self):
        """Test merging configurations from different builders."""
        builder1 = ConfigurationBuilder()
        builder1.set_from_cli("key1", "cli_value")
        builder1.set_from_file("key2", "file_value")
        
        builder2 = ConfigurationBuilder()
        builder2.set_from_file("key1", "other_file_value")  # Lower precedence
        builder2.set_from_cli("key3", "other_cli_value")
        
        builder1.merge_from(builder2)
        
        # CLI from builder1 should be preserved
        assert builder1.get("key1") == "cli_value"
        # New key should be added
        assert builder1.get("key3") == "other_cli_value"
        # File value should be preserved
        assert builder1.get("key2") == "file_value"


class TestConfigurationResolver:
    """Test the ConfigurationResolver class."""
    
    def test_default_configuration(self):
        """Test that system defaults are applied."""
        resolver = ConfigurationResolver()
        
        # Should have default values
        assert resolver.builder.get("search.top_k") == 10
        assert resolver.builder.get("search.use_reranker") == True  # Default for search
        assert resolver.builder.get("indexer.use_reranker") == False  # Default for indexer
    
    def test_file_configuration_loading(self):
        """Test loading configuration from dictionary (file source)."""
        resolver = ConfigurationResolver()
        
        config_dict = {
            "query": {
                "rerank": False,  # Legacy key
                "top_k": 20,
            }
        }
        
        resolver.load_from_dict(config_dict, ConfigSource.FILE)
        
        # Should map legacy key to new key
        assert resolver.builder.get("search.use_reranker") == False
        assert resolver.builder.get("search.top_k") == 20
    
    def test_cli_override_precedence(self):
        """Test that CLI arguments override file configuration."""
        resolver = ConfigurationResolver()
        
        # Load file config first
        config_dict = {
            "query": {
                "rerank": False,
                "top_k": 20,
            }
        }
        resolver.load_from_dict(config_dict, ConfigSource.FILE)
        
        # Override with CLI
        resolver.set_from_cli_args(use_reranker=True, top_k=5)
        
        # CLI should win
        assert resolver.builder.get("search.use_reranker") == True
        assert resolver.builder.get("search.top_k") == 5
        
        # Verify sources
        reranker_config = resolver.builder.get_with_source("search.use_reranker")
        assert reranker_config.source == ConfigSource.CLI
    
    def test_search_config_resolution(self):
        """Test resolving search configuration."""
        resolver = ConfigurationResolver()
        
        # Set some explicit values
        resolver.set_from_cli_args(use_reranker=True, top_k=15)
        
        config = resolver.resolve_search_config("test query", SearchMode.HYBRID)
        
        assert config.query == "test query"
        assert config.mode == SearchMode.HYBRID
        assert config.use_reranker == True
        assert config.top_k == 15
        assert config.reranker_model == "cl-nagoya/ruri-reranker-small"  # Default
        
        # Check sources
        assert "search.use_reranker" in config.sources
        assert config.sources["search.use_reranker"] == ConfigSource.CLI
    
    def test_legacy_configuration_mapping(self):
        """Test mapping of legacy configuration keys."""
        resolver = ConfigurationResolver()
        
        # Test various legacy key formats
        legacy_configs = [
            {"query": {"rerank": True}},  # Legacy rerank
            {"query": {"use_reranker": True}},  # Also should work
            {"query": {"rerank_model": "custom-model"}},  # Legacy model key
        ]
        
        for config_dict in legacy_configs:
            fresh_resolver = ConfigurationResolver()
            fresh_resolver.load_from_dict(config_dict, ConfigSource.FILE)
            
            if "rerank" in str(config_dict) or "use_reranker" in str(config_dict):
                assert fresh_resolver.builder.get("search.use_reranker") == True
            
            if "rerank_model" in str(config_dict):
                assert fresh_resolver.builder.get("search.reranker_model") == "custom-model"


class TestSearchContext:
    """Test the immutable SearchContext class."""
    
    def test_immutable_search_context(self):
        """Test that SearchContext is immutable."""
        context = SearchContext(
            query="test",
            mode=SearchMode.HYBRID,
            top_k=10,
            use_reranker=True
        )
        
        # Should be frozen - cannot modify
        with pytest.raises(Exception):  # dataclass frozen should raise FrozenInstanceError
            context.top_k = 20
    
    def test_explicit_value_override_protection(self):
        """Test that explicit values cannot be overridden."""
        context = SearchContext(
            query="test",
            mode=SearchMode.HYBRID,
            use_reranker=True  # Already explicitly set
        )
        
        # Should raise error when trying to override explicit value
        with pytest.raises(ValueError, match="already explicitly set"):
            context.with_explicit_reranker(False)
    
    def test_context_creation_with_none_values(self):
        """Test creating context with None values (not explicitly set)."""
        context = SearchContext(
            query="test",
            mode=SearchMode.HYBRID,
            # use_reranker not set (None)
        )
        
        # Should allow setting when None
        new_context = context.with_explicit_reranker(True)
        assert new_context.use_reranker == True
        assert context.use_reranker is None  # Original unchanged
    
    def test_context_merge(self):
        """Test merging contexts with proper precedence."""
        context1 = SearchContext(
            query="test",
            mode=SearchMode.HYBRID,
            use_reranker=True,  # Explicit
            top_k=None  # Not set
        )
        
        context2 = SearchContext(
            query="other",  # Different query (ignored in merge)
            mode=SearchMode.VECTOR,  # Different mode (ignored in merge)
            use_reranker=False,  # Different value
            top_k=20  # Set
        )
        
        merged = context1.merge_with(context2)
        
        # context1 values should be preserved
        assert merged.query == "test"
        assert merged.mode == SearchMode.HYBRID
        assert merged.use_reranker == True  # context1's explicit value
        assert merged.top_k == 20  # context2's value (context1 was None)


class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""
    
    def test_user_explicit_reranker_never_overridden(self):
        """Test the main issue: user explicit reranker setting is never overridden."""
        resolver = ConfigurationResolver()
        
        # Simulate file config that would override
        file_config = {
            "query": {"rerank": False},  # File says False
            "indexer": {"use_reranker": False}  # Indexer also says False
        }
        resolver.load_from_dict(file_config, ConfigSource.FILE)
        
        # User explicitly sets reranker via CLI
        resolver.set_from_cli_args(use_reranker=True)
        
        # Resolve configuration
        config = resolver.resolve_search_config("test", SearchMode.HYBRID)
        
        # User's explicit setting should win
        assert config.use_reranker == True
        assert config.sources["search.use_reranker"] == ConfigSource.CLI
    
    def test_configuration_source_tracking(self):
        """Test that configuration sources are properly tracked."""
        resolver = ConfigurationResolver()
        
        # Set up mixed sources
        resolver.load_from_dict({"query": {"top_k": 20}}, ConfigSource.FILE)
        resolver.set_from_cli_args(use_reranker=True)
        
        config = resolver.resolve_search_config("test", SearchMode.HYBRID)
        
        # Verify source tracking
        assert config.sources["search.use_reranker"] == ConfigSource.CLI
        assert config.sources["search.top_k"] == ConfigSource.FILE
        assert "search.reranker_model" in config.sources  # Default
    
    def test_configuration_conflict_detection(self, caplog):
        """Test configuration conflict detection and warnings."""
        resolver = ConfigurationResolver()
        
        # Create conflicting configuration
        config_dict = {
            "query": {"rerank": True},  # Search says True
            "indexer": {"use_reranker": False}  # Indexer says False
        }
        resolver.load_from_dict(config_dict, ConfigSource.FILE)
        
        # Enable logging to capture warnings
        with caplog.at_level(logging.WARNING):
            resolver.resolve_search_config("test", SearchMode.HYBRID)
        
        # Should detect conflict
        assert "conflict" in caplog.text.lower()
    
    @pytest.mark.parametrize("cli_value,file_value,expected", [
        (True, False, True),    # CLI True overrides file False
        (False, True, False),   # CLI False overrides file True
        (None, True, True),     # No CLI, use file True
        (None, False, False),   # No CLI, use file False
        (None, None, True),     # No CLI or file, use default True
    ])
    def test_reranker_precedence_matrix(self, cli_value, file_value, expected):
        """Test reranker configuration precedence in various scenarios."""
        resolver = ConfigurationResolver()
        
        # Set file config if provided
        if file_value is not None:
            resolver.load_from_dict({"query": {"rerank": file_value}}, ConfigSource.FILE)
        
        # Set CLI config if provided
        if cli_value is not None:
            resolver.set_from_cli_args(use_reranker=cli_value)
        
        config = resolver.resolve_search_config("test", SearchMode.HYBRID)
        assert config.use_reranker == expected