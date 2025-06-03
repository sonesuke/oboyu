"""Tests for search filter functionality."""

import pytest
from datetime import datetime
from typing import Dict, Any

from oboyu.common.types import DateRangeFilter, PathFilter, SearchFilters


class TestDateRangeFilter:
    """Test date range filter functionality."""

    def test_valid_date_range_filter(self) -> None:
        """Test creating a valid date range filter."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        filter_obj = DateRangeFilter(start=start_date, end=end_date, field="modified_at")
        
        assert filter_obj.start == start_date
        assert filter_obj.end == end_date
        assert filter_obj.field == "modified_at"

    def test_invalid_date_field(self) -> None:
        """Test that invalid date field raises ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="String should match pattern"):
            DateRangeFilter(field="invalid_field")

    def test_start_after_end_date(self) -> None:
        """Test that start date after end date raises ValueError."""
        start_date = datetime(2024, 12, 31)
        end_date = datetime(2024, 1, 1)
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            DateRangeFilter(start=start_date, end=end_date)

    def test_only_start_date(self) -> None:
        """Test filter with only start date."""
        start_date = datetime(2024, 1, 1)
        filter_obj = DateRangeFilter(start=start_date)
        
        assert filter_obj.start == start_date
        assert filter_obj.end is None

    def test_only_end_date(self) -> None:
        """Test filter with only end date."""
        end_date = datetime(2024, 12, 31)
        filter_obj = DateRangeFilter(end=end_date)
        
        assert filter_obj.start is None
        assert filter_obj.end == end_date


class TestPathFilter:
    """Test path filter functionality."""

    def test_valid_path_filter_include_only(self) -> None:
        """Test creating a path filter with include patterns only."""
        include_patterns = ["*/docs/*", "*.md"]
        filter_obj = PathFilter(include_patterns=include_patterns)
        
        assert filter_obj.include_patterns == include_patterns
        assert filter_obj.exclude_patterns is None

    def test_valid_path_filter_exclude_only(self) -> None:
        """Test creating a path filter with exclude patterns only."""
        exclude_patterns = ["*/test/*", "*.log"]
        filter_obj = PathFilter(exclude_patterns=exclude_patterns)
        
        assert filter_obj.include_patterns is None
        assert filter_obj.exclude_patterns == exclude_patterns

    def test_valid_path_filter_both(self) -> None:
        """Test creating a path filter with both include and exclude patterns."""
        include_patterns = ["*/docs/*"]
        exclude_patterns = ["*/test/*"]
        filter_obj = PathFilter(include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        
        assert filter_obj.include_patterns == include_patterns
        assert filter_obj.exclude_patterns == exclude_patterns

    def test_empty_path_filter(self) -> None:
        """Test that empty path filter is now allowed for backward compatibility."""
        # This should no longer raise an error
        filter_obj = PathFilter()
        assert filter_obj.include_patterns is None
        assert filter_obj.exclude_patterns is None

    def test_path_matching_include_only(self) -> None:
        """Test path matching with include patterns only."""
        filter_obj = PathFilter(include_patterns=["*/docs/*", "*.md"])
        
        # Should match
        assert filter_obj.matches("/project/docs/readme.txt")
        assert filter_obj.matches("/project/test.md")
        
        # Should not match
        assert not filter_obj.matches("/project/src/main.py")
        assert not filter_obj.matches("/project/test/unit.py")

    def test_path_matching_exclude_only(self) -> None:
        """Test path matching with exclude patterns only."""
        filter_obj = PathFilter(exclude_patterns=["*/test/*", "*.log"])
        
        # Should match (not excluded)
        assert filter_obj.matches("/project/docs/readme.txt")
        assert filter_obj.matches("/project/src/main.py")
        
        # Should not match (excluded)
        assert not filter_obj.matches("/project/test/unit.py")
        assert not filter_obj.matches("/project/app.log")

    def test_path_matching_both_patterns(self) -> None:
        """Test path matching with both include and exclude patterns."""
        filter_obj = PathFilter(
            include_patterns=["*/docs/*"],
            exclude_patterns=["*/test/*"]
        )
        
        # Should match (included and not excluded)
        assert filter_obj.matches("/project/docs/readme.txt")
        
        # Should not match (not included)
        assert not filter_obj.matches("/project/src/main.py")
        
        # Should not match (excluded, even if it would be included)
        assert not filter_obj.matches("/project/docs/test/example.txt")


class TestSearchFilters:
    """Test combined search filters functionality."""

    def test_empty_search_filters(self) -> None:
        """Test empty search filters."""
        filters = SearchFilters()
        
        assert filters.date_range is None
        assert filters.path_filter is None
        assert not filters.has_filters()

    def test_search_filters_with_date_range(self) -> None:
        """Test search filters with date range only."""
        date_range = DateRangeFilter(start=datetime(2024, 1, 1))
        filters = SearchFilters(date_range=date_range)
        
        assert filters.date_range == date_range
        assert filters.path_filter is None
        assert filters.has_filters()

    def test_search_filters_with_path_filter(self) -> None:
        """Test search filters with path filter only."""
        path_filter = PathFilter(include_patterns=["*.md"])
        filters = SearchFilters(path_filter=path_filter)
        
        assert filters.date_range is None
        assert filters.path_filter == path_filter
        assert filters.has_filters()

    def test_search_filters_with_both(self) -> None:
        """Test search filters with both date range and path filter."""
        date_range = DateRangeFilter(start=datetime(2024, 1, 1))
        path_filter = PathFilter(include_patterns=["*.md"])
        filters = SearchFilters(date_range=date_range, path_filter=path_filter)
        
        assert filters.date_range == date_range
        assert filters.path_filter == path_filter
        assert filters.has_filters()

    def test_parse_date_string_formats(self) -> None:
        """Test parsing various date string formats."""
        # ISO date format
        date1 = SearchFilters._parse_date("2024-01-01")
        assert date1 == datetime(2024, 1, 1)
        
        # ISO datetime format
        date2 = SearchFilters._parse_date("2024-01-01T12:00:00")
        assert date2 == datetime(2024, 1, 1, 12, 0, 0)
        
        # Space-separated datetime format
        date3 = SearchFilters._parse_date("2024-01-01 12:00:00")
        assert date3 == datetime(2024, 1, 1, 12, 0, 0)

    def test_parse_date_invalid_format(self) -> None:
        """Test parsing invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            SearchFilters._parse_date("invalid-date")

    def test_parse_date_invalid_type(self) -> None:
        """Test parsing invalid date type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date type"):
            SearchFilters._parse_date(123)

    def test_from_dict_date_range_only(self) -> None:
        """Test creating SearchFilters from dict with date range only."""
        filter_dict = {
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "field": "created_at"
            }
        }
        
        filters = SearchFilters.from_dict(filter_dict)
        
        assert filters.date_range is not None
        assert filters.date_range.start == datetime(2024, 1, 1)
        assert filters.date_range.end == datetime(2024, 12, 31)
        assert filters.date_range.field == "created_at"
        assert filters.path_filter is None

    def test_from_dict_path_filter_only(self) -> None:
        """Test creating SearchFilters from dict with path filter only."""
        filter_dict = {
            "path_filter": {
                "include_patterns": ["*/docs/*", "*.md"],
                "exclude_patterns": ["*/test/*"]
            }
        }
        
        filters = SearchFilters.from_dict(filter_dict)
        
        assert filters.date_range is None
        assert filters.path_filter is not None
        assert filters.path_filter.include_patterns == ["*/docs/*", "*.md"]
        assert filters.path_filter.exclude_patterns == ["*/test/*"]

    def test_from_dict_both_filters(self) -> None:
        """Test creating SearchFilters from dict with both filters."""
        filter_dict = {
            "date_range": {
                "start": "2024-01-01"
            },
            "path_filter": {
                "include_patterns": ["*.md"]
            }
        }
        
        filters = SearchFilters.from_dict(filter_dict)
        
        assert filters.date_range is not None
        assert filters.date_range.start == datetime(2024, 1, 1)
        assert filters.path_filter is not None
        assert filters.path_filter.include_patterns == ["*.md"]

    def test_to_sql_conditions_date_range_only(self) -> None:
        """Test converting date range filter to SQL conditions."""
        date_range = DateRangeFilter(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            field="modified_at"
        )
        filters = SearchFilters(date_range=date_range)
        
        conditions, parameters = filters.to_sql_conditions()
        
        assert len(conditions) == 2
        assert "c.modified_at >= ?" in conditions
        assert "c.modified_at <= ?" in conditions
        assert len(parameters) == 2
        assert parameters[0] == "2024-01-01T00:00:00"
        assert parameters[1] == "2024-12-31T00:00:00"

    def test_to_sql_conditions_path_filter_only(self) -> None:
        """Test converting path filter to SQL conditions."""
        path_filter = PathFilter(
            include_patterns=["*/docs/*"],
            exclude_patterns=["*/test/*", "*.log"]
        )
        filters = SearchFilters(path_filter=path_filter)
        
        conditions, parameters = filters.to_sql_conditions()
        
        # Should have 1 include condition + 2 exclude conditions
        assert len(conditions) == 3
        assert "(c.path LIKE ?)" in conditions
        assert "c.path NOT LIKE ?" in conditions
        assert len(parameters) == 3
        assert "%/docs/%" in parameters
        assert "%/test/%" in parameters
        assert "%.log" in parameters

    def test_to_sql_conditions_both_filters(self) -> None:
        """Test converting both filters to SQL conditions."""
        date_range = DateRangeFilter(start=datetime(2024, 1, 1))
        path_filter = PathFilter(include_patterns=["*.md"])
        filters = SearchFilters(date_range=date_range, path_filter=path_filter)
        
        conditions, parameters = filters.to_sql_conditions()
        
        # Should have 1 date condition + 1 path condition
        assert len(conditions) == 2
        assert any("c.modified_at >=" in cond for cond in conditions)
        assert any("c.path LIKE" in cond for cond in conditions)
        assert len(parameters) == 2

    def test_to_sql_conditions_empty_filters(self) -> None:
        """Test converting empty filters to SQL conditions."""
        filters = SearchFilters()
        
        conditions, parameters = filters.to_sql_conditions()
        
        assert len(conditions) == 0
        assert len(parameters) == 0