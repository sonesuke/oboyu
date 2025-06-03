"""Integration tests for MCP server search filtering functionality."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from oboyu.mcp.server import search
from oboyu.common.types import SearchFilters


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        yield str(db_path)


class TestMCPSearchFiltering:
    """Integration tests for search filtering through MCP server."""

    def test_search_without_filters(self, temp_db: str) -> None:
        """Test search without any filters (backward compatibility)."""
        result = search(
            query="test query",
            mode="hybrid",
            top_k=5,
            db_path=temp_db
        )
        
        # Should not raise an error and return expected structure
        assert "results" in result
        assert "stats" in result
        assert isinstance(result["results"], list)

    def test_search_with_empty_filters(self, temp_db: str) -> None:
        """Test search with empty filters dict."""
        result = search(
            query="test query",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters={}
        )
        
        # Should work the same as no filters
        assert "results" in result
        assert "stats" in result

    def test_search_with_date_range_filter(self, temp_db: str) -> None:
        """Test search with date range filter."""
        filters = {
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "field": "modified_at"
            }
        }
        
        result = search(
            query="test query",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        # Should not raise an error
        assert "results" in result
        assert "stats" in result

    def test_search_with_path_filter(self, temp_db: str) -> None:
        """Test search with path filter."""
        filters = {
            "path_filter": {
                "include_patterns": ["*/docs/*", "*.md"],
                "exclude_patterns": ["*/test/*", "*.log"]
            }
        }
        
        result = search(
            query="test query", 
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        # Should not raise an error
        assert "results" in result
        assert "stats" in result

    def test_search_with_combined_filters(self, temp_db: str) -> None:
        """Test search with both date range and path filters."""
        filters = {
            "date_range": {
                "start": "2024-06-01",
                "field": "modified_at"
            },
            "path_filter": {
                "include_patterns": ["*/docs/*"],
                "exclude_patterns": ["*/archived/*"]
            }
        }
        
        result = search(
            query="test query",
            mode="hybrid", 
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        # Should not raise an error
        assert "results" in result
        assert "stats" in result

    def test_search_with_invalid_filters(self, temp_db: str) -> None:
        """Test search with invalid filters should not crash."""
        filters = {
            "date_range": {
                "start": "invalid-date",
                "field": "invalid_field"
            }
        }
        
        result = search(
            query="test query",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        # Should handle invalid filters gracefully and continue without filters
        assert "results" in result
        assert "stats" in result

    def test_search_filters_integration_with_snippet_config(self, temp_db: str) -> None:
        """Test that filters work correctly with snippet configuration."""
        filters = {
            "date_range": {
                "start": "2024-01-01"
            }
        }
        
        snippet_config = {
            "max_length": 200,
            "show_surrounding_context": True
        }
        
        result = search(
            query="test query",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters,
            snippet_config=snippet_config
        )
        
        # Should handle both filters and snippet config
        assert "results" in result
        assert "stats" in result

    def test_search_filters_with_all_modes(self, temp_db: str) -> None:
        """Test that filters work with all search modes."""
        filters = {
            "path_filter": {
                "include_patterns": ["*.md"]
            }
        }
        
        # Test vector mode
        result_vector = search(
            query="test query",
            mode="vector",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        assert "results" in result_vector
        
        # Test BM25 mode
        result_bm25 = search(
            query="test query",
            mode="bm25",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        assert "results" in result_bm25
        
        # Test hybrid mode
        result_hybrid = search(
            query="test query",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        assert "results" in result_hybrid

    def test_search_filters_with_language_filter(self, temp_db: str) -> None:
        """Test that search filters work correctly with language filtering."""
        filters = {
            "date_range": {
                "start": "2024-01-01"
            }
        }
        
        result = search(
            query="テストクエリ",
            mode="hybrid",
            top_k=5,
            language="ja",
            db_path=temp_db,
            filters=filters
        )
        
        # Should handle both language filter and search filters
        assert "results" in result
        assert "stats" in result
        assert result["stats"]["language_filter"] == "ja"


class TestSearchFiltersValidation:
    """Test validation of search filters."""

    def test_date_range_filter_validation(self) -> None:
        """Test date range filter validation."""
        # Valid filter
        valid_filter = {
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "field": "modified_at"
            }
        }
        filters = SearchFilters.from_dict(valid_filter)
        assert filters.date_range is not None

    def test_path_filter_validation(self) -> None:
        """Test path filter validation."""
        # Valid filter
        valid_filter = {
            "path_filter": {
                "include_patterns": ["*/docs/*"],
                "exclude_patterns": ["*/test/*"]
            }
        }
        filters = SearchFilters.from_dict(valid_filter)
        assert filters.path_filter is not None

    def test_invalid_date_range_filter(self) -> None:
        """Test invalid date range filter raises appropriate error."""
        invalid_filter = {
            "date_range": {
                "start": "2024-12-31",
                "end": "2024-01-01"  # End before start
            }
        }
        
        with pytest.raises(ValueError):
            SearchFilters.from_dict(invalid_filter)

    def test_empty_path_filter_allowed(self) -> None:
        """Test that empty path filters are allowed for backward compatibility."""
        filter_with_empty_path = {
            "path_filter": {
                # No include or exclude patterns - this is now allowed
            }
        }
        
        # This should not raise an error
        filters = SearchFilters.from_dict(filter_with_empty_path)
        assert filters.path_filter is not None
        assert filters.path_filter.include_patterns is None
        assert filters.path_filter.exclude_patterns is None


class TestSearchFiltersExamples:
    """Test examples from the GitHub issue specification."""

    def test_japanese_date_range_example(self, temp_db: str) -> None:
        """Test the Japanese date range example from the issue."""
        filters = {
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "field": "modified_at"
            }
        }
        
        result = search(
            query="機械学習アルゴリズム",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        assert "results" in result
        assert "stats" in result

    def test_api_documentation_path_example(self, temp_db: str) -> None:
        """Test the API documentation path filtering example from the issue."""
        filters = {
            "path_filter": {
                "include_patterns": ["*/docs/*", "*/api/*", "*.md"],
                "exclude_patterns": ["*/test/*", "*/temp/*", "*.log"]
            }
        }
        
        result = search(
            query="API documentation",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        assert "results" in result
        assert "stats" in result

    def test_combined_filtering_example(self, temp_db: str) -> None:
        """Test the combined filtering example from the issue."""
        filters = {
            "date_range": {
                "start": "2024-06-01",
                "field": "modified_at"
            },
            "path_filter": {
                "include_patterns": ["*/documentation/*"],
                "exclude_patterns": ["*/archived/*"]
            }
        }
        
        result = search(
            query="設計パターン",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        assert "results" in result
        assert "stats" in result

    def test_recent_documents_example(self, temp_db: str) -> None:
        """Test the recent documents example from the issue."""
        filters = {
            "date_range": {
                "start": "2024-05-01",
                "field": "modified_at"
            },
            "path_filter": {
                "include_patterns": ["*/docs/*"]
            }
        }
        
        result = search(
            query="システム設計",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        assert "results" in result
        assert "stats" in result

    def test_project_specific_search_example(self, temp_db: str) -> None:
        """Test the project-specific search example from the issue."""
        filters = {
            "path_filter": {
                "include_patterns": ["*/backend/*", "*/api/*"],
                "exclude_patterns": ["*/test/*", "*/deprecated/*"]
            }
        }
        
        result = search(
            query="API implementation",
            mode="hybrid",
            top_k=5,
            db_path=temp_db,
            filters=filters
        )
        
        assert "results" in result
        assert "stats" in result