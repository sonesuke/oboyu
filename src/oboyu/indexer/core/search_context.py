"""Search context for tracking explicit configuration values."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SettingSource(Enum):
    """Source of a configuration setting."""
    
    CLI_ARGUMENT = "cli_argument"
    FUNCTION_CALL = "function_call"
    CONFIG_FILE = "config_file"
    SYSTEM_DEFAULT = "system_default"


@dataclass
class SearchContext:
    """Search context that holds ONLY explicitly set values - no defaults.
    
    This class implements the Context Pattern to ensure user-specified
    settings are never overridden by default values.
    """
    
    _explicit_settings: Dict[str, Any] = field(default_factory=dict)
    _setting_sources: Dict[str, SettingSource] = field(default_factory=dict)
    
    def set_reranker(self, enabled: bool, source: SettingSource) -> None:
        """Explicitly set reranker - never overridden.
        
        Args:
            enabled: Whether to enable reranker
            source: Source of this setting
            
        """
        self._explicit_settings['reranker_enabled'] = enabled
        self._setting_sources['reranker_enabled'] = source
        logger.info(f"🔧 Reranker EXPLICITLY set: {enabled} (source: {source.value})")
    
    def set_top_k(self, value: int, source: SettingSource) -> None:
        """Explicitly set top_k - never overridden.
        
        Args:
            value: Top k value
            source: Source of this setting
            
        """
        self._explicit_settings['top_k'] = value
        self._setting_sources['top_k'] = source
        logger.debug(f"🔧 Top-k EXPLICITLY set: {value} (source: {source.value})")
    
    def set_vector_weight(self, value: float, source: SettingSource) -> None:
        """Explicitly set vector weight - never overridden.
        
        Args:
            value: Vector weight value
            source: Source of this setting
            
        """
        self._explicit_settings['vector_weight'] = value
        self._setting_sources['vector_weight'] = source
        logger.debug(f"🔧 Vector weight EXPLICITLY set: {value} (source: {source.value})")
    
    def set_bm25_weight(self, value: float, source: SettingSource) -> None:
        """Explicitly set BM25 weight - never overridden.
        
        Args:
            value: BM25 weight value
            source: Source of this setting
            
        """
        self._explicit_settings['bm25_weight'] = value
        self._setting_sources['bm25_weight'] = source
        logger.debug(f"🔧 BM25 weight EXPLICITLY set: {value} (source: {source.value})")
    
    def is_explicitly_set(self, key: str) -> bool:
        """Check if setting was explicitly provided by user.
        
        Args:
            key: Setting key to check
            
        Returns:
            True if setting was explicitly set
            
        """
        return key in self._explicit_settings
    
    def get_reranker_setting(self) -> Optional[bool]:
        """Get explicit reranker setting (None if not set).
        
        Returns:
            Explicit reranker setting or None
            
        """
        return self._explicit_settings.get('reranker_enabled')
    
    def get_top_k_setting(self) -> Optional[int]:
        """Get explicit top_k setting (None if not set).
        
        Returns:
            Explicit top_k setting or None
            
        """
        return self._explicit_settings.get('top_k')
    
    def get_vector_weight_setting(self) -> Optional[float]:
        """Get explicit vector weight setting (None if not set).
        
        Returns:
            Explicit vector weight setting or None
            
        """
        return self._explicit_settings.get('vector_weight')
    
    def get_bm25_weight_setting(self) -> Optional[float]:
        """Get explicit BM25 weight setting (None if not set).
        
        Returns:
            Explicit BM25 weight setting or None
            
        """
        return self._explicit_settings.get('bm25_weight')
    
    def get_setting_source(self, key: str) -> Optional[SettingSource]:
        """Get the source of a setting.
        
        Args:
            key: Setting key
            
        Returns:
            Source of the setting or None if not set
            
        """
        return self._setting_sources.get(key)
    
    def get_all_explicit_settings(self) -> Dict[str, Any]:
        """Get all explicitly set settings.
        
        Returns:
            Dictionary of all explicit settings
            
        """
        return self._explicit_settings.copy()
    
    def log_final_settings(self, final_settings: Dict[str, Any]) -> None:
        """Log final settings with their sources for debugging.
        
        Args:
            final_settings: Dictionary of final settings used
            
        """
        logger.info("📋 Final search settings:")
        for key, value in final_settings.items():
            source = self.get_setting_source(key)
            if source:
                logger.info(f"  {key}: {value} (EXPLICIT from {source.value})")
            else:
                logger.info(f"  {key}: {value} (DEFAULT)")


class ContextBuilder:
    """Builder for creating SearchContext with fluent interface."""
    
    def __init__(self) -> None:
        """Initialize the context builder."""
        self.context = SearchContext()
    
    def with_reranker(self, enabled: bool, source: SettingSource = SettingSource.FUNCTION_CALL) -> "ContextBuilder":
        """Set reranker setting.
        
        Args:
            enabled: Whether to enable reranker
            source: Source of this setting
            
        Returns:
            Self for chaining
            
        """
        self.context.set_reranker(enabled, source)
        return self
    
    def with_top_k(self, value: int, source: SettingSource = SettingSource.FUNCTION_CALL) -> "ContextBuilder":
        """Set top_k setting.
        
        Args:
            value: Top k value
            source: Source of this setting
            
        Returns:
            Self for chaining
            
        """
        self.context.set_top_k(value, source)
        return self
    
    def with_vector_weight(self, value: float, source: SettingSource = SettingSource.FUNCTION_CALL) -> "ContextBuilder":
        """Set vector weight setting.
        
        Args:
            value: Vector weight value
            source: Source of this setting
            
        Returns:
            Self for chaining
            
        """
        self.context.set_vector_weight(value, source)
        return self
    
    def with_bm25_weight(self, value: float, source: SettingSource = SettingSource.FUNCTION_CALL) -> "ContextBuilder":
        """Set BM25 weight setting.
        
        Args:
            value: BM25 weight value
            source: Source of this setting
            
        Returns:
            Self for chaining
            
        """
        self.context.set_bm25_weight(value, source)
        return self
    
    def from_cli_args(
        self,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> "ContextBuilder":
        """Set multiple settings from CLI arguments.
        
        Args:
            top_k: Optional top_k value
            use_reranker: Optional reranker setting
            vector_weight: Optional vector weight
            bm25_weight: Optional BM25 weight
            
        Returns:
            Self for chaining
            
        """
        if top_k is not None:
            self.with_top_k(top_k, SettingSource.CLI_ARGUMENT)
        if use_reranker is not None:
            self.with_reranker(use_reranker, SettingSource.CLI_ARGUMENT)
        if vector_weight is not None:
            self.with_vector_weight(vector_weight, SettingSource.CLI_ARGUMENT)
        if bm25_weight is not None:
            self.with_bm25_weight(bm25_weight, SettingSource.CLI_ARGUMENT)
        return self
    
    def from_function_args(
        self,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> "ContextBuilder":
        """Set multiple settings from function arguments.
        
        Args:
            top_k: Optional top_k value
            use_reranker: Optional reranker setting
            vector_weight: Optional vector weight
            bm25_weight: Optional BM25 weight
            
        Returns:
            Self for chaining
            
        """
        if top_k is not None:
            self.with_top_k(top_k, SettingSource.FUNCTION_CALL)
        if use_reranker is not None:
            self.with_reranker(use_reranker, SettingSource.FUNCTION_CALL)
        if vector_weight is not None:
            self.with_vector_weight(vector_weight, SettingSource.FUNCTION_CALL)
        if bm25_weight is not None:
            self.with_bm25_weight(bm25_weight, SettingSource.FUNCTION_CALL)
        return self
    
    def build(self) -> SearchContext:
        """Build the final context.
        
        Returns:
            Configured SearchContext
            
        """
        return self.context


# Convenience class for defaults (only used when no explicit setting)
class SystemDefaults:
    """System default values - only used when no explicit setting provided."""
    
    @staticmethod
    def get_reranker_default() -> bool:
        """Get default reranker setting."""
        return False
    
    @staticmethod
    def get_top_k_default() -> int:
        """Get default top_k setting."""
        return 10
    
    @staticmethod
    def get_vector_weight_default() -> float:
        """Get default vector weight setting."""
        return 0.7
    
    @staticmethod
    def get_bm25_weight_default() -> float:
        """Get default BM25 weight setting."""
        return 0.3
