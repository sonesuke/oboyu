"""Tests for the configuration handling functionality."""

from oboyu.crawler.config import CrawlerConfig, load_default_config


class TestCrawlerConfig:
    """Test cases for the CrawlerConfig class."""
    def test_default_config(self) -> None:
        """Test loading default configuration."""
        config = load_default_config()
        # Check default values
        assert config.depth == 10
        assert "*.txt" in config.include_patterns
        assert "*/node_modules/*" in config.exclude_patterns
        assert config.max_workers == 4  # Default worker count
    
    def test_config_from_dict(self) -> None:
        """Test loading configuration from dictionary."""
        config_dict = {
            "crawler": {
                "depth": 5,
                "include_patterns": ["*.csv"],
                "exclude_patterns": ["*/temp/*"],
                "max_workers": 8,
            }
        }
        config = CrawlerConfig(config_dict=config_dict)
        # Check values from dict
        assert config.depth == 5
        assert config.include_patterns == ["*.csv"]
        assert config.exclude_patterns == ["*/temp/*"]
        assert config.max_workers == 8
    
    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Test with invalid values
        config_dict = {
            "crawler": {
                "depth": -5,  # Invalid: negative - will be replaced with default
                "include_patterns": "not-a-list",  # Invalid: not a list - will be replaced
                "exclude_patterns": None,  # Invalid: None - will be replaced
                "japanese_encodings": {},  # Invalid: dict instead of list - will be replaced
            }
        }
        # This should not raise an exception
        config = CrawlerConfig(config_dict=config_dict)
        # All values should now be valid defaults
        assert config.depth == 10  # Default
        assert "*.txt" in config.include_patterns  # Default include pattern
        assert "*/node_modules/*" in config.exclude_patterns  # Default exclude pattern
    
    def test_partial_config(self) -> None:
        """Test partial configuration override."""
        # Only override some values
        config_dict = {
            "crawler": {
                "depth": 5,
                "include_patterns": ["*.csv"],
                # Other values not specified - will be set to defaults during validation
            }
        }
        # Set the defaults first, then update with our partial config
        default_config = CrawlerConfig()
        config = CrawlerConfig(config_dict=config_dict)
        # Check overridden values
        assert config.depth == 5
        assert config.include_patterns == ["*.csv"]
        # Check default values for non-overridden
        # The validation should have added these fields with default values
        assert len(config.exclude_patterns) > 0
        assert "*/node_modules/*" in config.exclude_patterns
