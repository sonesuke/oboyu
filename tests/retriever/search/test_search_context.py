"""Tests for SearchContext pattern that prevents setting override."""

import pytest

from oboyu.retriever.search.search_context import ContextBuilder, SearchContext, SettingSource, SystemDefaults


class TestSearchContext:
    """Test SearchContext pattern for explicit setting tracking."""
    
    def test_empty_context_has_no_explicit_settings(self) -> None:
        """Test that empty context has no explicit settings."""
        context = SearchContext()
        
        assert not context.is_explicitly_set('reranker_enabled')
        assert not context.is_explicitly_set('top_k')
        assert context.get_reranker_setting() is None
        assert context.get_top_k_setting() is None
    
    def test_explicit_reranker_setting_is_preserved(self) -> None:
        """Test that explicit reranker setting is never overridden."""
        context = SearchContext()
        
        # Set reranker explicitly
        context.set_reranker(True, SettingSource.CLI_ARGUMENT)
        
        # Verify it's marked as explicit
        assert context.is_explicitly_set('reranker_enabled')
        assert context.get_reranker_setting() is True
        assert context.get_setting_source('reranker_enabled') == SettingSource.CLI_ARGUMENT
    
    def test_explicit_false_reranker_is_preserved(self) -> None:
        """Test that explicitly disabled reranker is preserved."""
        context = SearchContext()
        
        # Explicitly disable reranker
        context.set_reranker(False, SettingSource.FUNCTION_CALL)
        
        # Verify explicit False is preserved
        assert context.is_explicitly_set('reranker_enabled')
        assert context.get_reranker_setting() is False
        assert context.get_setting_source('reranker_enabled') == SettingSource.FUNCTION_CALL
    
    def test_explicit_top_k_setting_is_preserved(self) -> None:
        """Test that explicit top_k setting is never overridden."""
        context = SearchContext()
        
        # Set top_k explicitly
        context.set_top_k(25, SettingSource.CLI_ARGUMENT)
        
        # Verify it's marked as explicit
        assert context.is_explicitly_set('top_k')
        assert context.get_top_k_setting() == 25
        assert context.get_setting_source('top_k') == SettingSource.CLI_ARGUMENT
    
    def test_context_builder_fluent_interface(self) -> None:
        """Test ContextBuilder fluent interface."""
        context = (ContextBuilder()
                  .with_reranker(True, SettingSource.CLI_ARGUMENT)
                  .with_top_k(20, SettingSource.FUNCTION_CALL)
                  .build())
        
        assert context.is_explicitly_set('reranker_enabled')
        assert context.is_explicitly_set('top_k')
        assert context.get_reranker_setting() is True
        assert context.get_top_k_setting() == 20
    
    def test_context_builder_from_cli_args(self) -> None:
        """Test ContextBuilder with CLI arguments."""
        context = (ContextBuilder()
                  .from_cli_args(top_k=15, use_reranker=True)
                  .build())
        
        assert context.is_explicitly_set('reranker_enabled')
        assert context.is_explicitly_set('top_k')
        assert context.get_reranker_setting() is True
        assert context.get_top_k_setting() == 15
        assert context.get_setting_source('reranker_enabled') == SettingSource.CLI_ARGUMENT
        assert context.get_setting_source('top_k') == SettingSource.CLI_ARGUMENT
    
    def test_context_builder_from_function_args(self) -> None:
        """Test ContextBuilder with function arguments."""
        context = (ContextBuilder()
                  .from_function_args(top_k=30, use_reranker=False)
                  .build())
        
        assert context.is_explicitly_set('reranker_enabled')
        assert context.is_explicitly_set('top_k')
        assert context.get_reranker_setting() is False
        assert context.get_top_k_setting() == 30
        assert context.get_setting_source('reranker_enabled') == SettingSource.FUNCTION_CALL
        assert context.get_setting_source('top_k') == SettingSource.FUNCTION_CALL
    
    def test_context_builder_ignores_none_values(self) -> None:
        """Test that ContextBuilder ignores None values."""
        context = (ContextBuilder()
                  .from_cli_args(top_k=None, use_reranker=True)
                  .build())
        
        assert context.is_explicitly_set('reranker_enabled')
        assert not context.is_explicitly_set('top_k')
        assert context.get_reranker_setting() is True
        assert context.get_top_k_setting() is None
    
    def test_get_all_explicit_settings(self) -> None:
        """Test getting all explicit settings."""
        context = (ContextBuilder()
                  .with_reranker(True, SettingSource.CLI_ARGUMENT)
                  .with_top_k(25, SettingSource.FUNCTION_CALL)
                  .build())
        
        settings = context.get_all_explicit_settings()
        expected = {
            'reranker_enabled': True,
            'top_k': 25,
        }
        assert settings == expected


class TestSystemDefaults:
    """Test SystemDefaults class."""
    
    def test_system_defaults_values(self) -> None:
        """Test that system defaults return expected values."""
        assert SystemDefaults.get_reranker_default() is False
        assert SystemDefaults.get_top_k_default() == 10
        assert SystemDefaults.get_vector_weight_default() == 0.7
        assert SystemDefaults.get_bm25_weight_default() == 0.3


class TestSearchContextIntegration:
    """Integration tests demonstrating the fix for issue 137."""
    
    def test_explicit_reranker_never_overridden(self) -> None:
        """Test that explicit reranker setting is never lost.
        
        This test demonstrates the fix for issue 137 where user-specified
        reranker settings were getting silently overridden by default values.
        """
        # User explicitly enables reranker via CLI
        context = ContextBuilder().from_cli_args(use_reranker=True).build()
        
        # Context should preserve explicit setting through entire pipeline
        assert context.is_explicitly_set('reranker_enabled')
        assert context.get_reranker_setting() is True
        assert context.get_setting_source('reranker_enabled') == SettingSource.CLI_ARGUMENT
        
        # This setting should NEVER be overridden by any default value
        # The search orchestrator will check is_explicitly_set() and use
        # the explicit value instead of falling back to config defaults
    
    def test_explicit_false_reranker_never_overridden(self) -> None:
        """Test that explicitly disabled reranker is never overridden."""
        # User explicitly disables reranker
        context = ContextBuilder().from_function_args(use_reranker=False).build()
        
        # Even explicit False should be preserved
        assert context.is_explicitly_set('reranker_enabled')
        assert context.get_reranker_setting() is False
        assert context.get_setting_source('reranker_enabled') == SettingSource.FUNCTION_CALL
    
    def test_multiple_explicit_settings_preserved(self) -> None:
        """Test that multiple explicit settings are all preserved."""
        context = (ContextBuilder()
                  .from_cli_args(use_reranker=True, top_k=50)
                  .build())
        
        # Both settings should be preserved
        assert context.is_explicitly_set('reranker_enabled')
        assert context.is_explicitly_set('top_k')
        assert context.get_reranker_setting() is True
        assert context.get_top_k_setting() == 50
        
        # Sources should be tracked
        assert context.get_setting_source('reranker_enabled') == SettingSource.CLI_ARGUMENT
        assert context.get_setting_source('top_k') == SettingSource.CLI_ARGUMENT