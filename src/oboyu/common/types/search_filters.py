"""Search filter data structures and validation.

This module provides filtering capabilities for search operations including
date range filtering and path pattern filtering.
"""

import fnmatch
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class DateRangeFilter(BaseModel):
    """Enhanced date range filter for search operations with validation."""
    
    start: Optional[datetime] = Field(default=None, description="Start date for filtering")
    end: Optional[datetime] = Field(default=None, description="End date for filtering")
    field: str = Field(default="modified_at", pattern=r'^(created_at|modified_at)$', description="Date field to filter on")
    
    @model_validator(mode='after')
    def validate_date_range(self) -> 'DateRangeFilter':
        """Validate date range consistency."""
        if self.start and self.end and self.start > self.end:
            raise ValueError("Start date must be before end date")
        return self
    
    @field_validator('start', 'end')
    @classmethod
    def validate_dates(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Validate date values."""
        if v is not None:
            # Ensure date is not in the future
            if v > datetime.now():
                raise ValueError("Date cannot be in the future")
        return v


class PathFilter(BaseModel):
    """Enhanced path pattern filter for search operations with validation."""
    
    include_patterns: Optional[List[str]] = Field(default=None, description="Patterns to include")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="Patterns to exclude")
    
    # Allow empty patterns for backward compatibility - validation only when actually used
    
    @field_validator('include_patterns', 'exclude_patterns')
    @classmethod
    def validate_pattern_lists(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate pattern lists."""
        if v is not None:
            # Remove empty patterns and normalize
            cleaned = [pattern.strip() for pattern in v if pattern.strip()]
            if not cleaned:
                return None
            
            # Validate pattern format
            for pattern in cleaned:
                if not pattern or pattern.isspace():
                    raise ValueError("Pattern cannot be empty or whitespace")
                # Basic validation for dangerous patterns
                if '..' in pattern:
                    raise ValueError("Patterns with '..' are not allowed for security reasons")
            
            return cleaned
        return v
    
    def matches(self, path: str) -> bool:
        """Check if a path matches this filter.
        
        Args:
            path: File path to check
            
        Returns:
            True if path matches filter criteria

        """
        # If path is excluded, return False
        if self.exclude_patterns:
            for pattern in self.exclude_patterns:
                if fnmatch.fnmatch(path, pattern):
                    return False
        
        # If no include patterns, path is included (but may be excluded above)
        if not self.include_patterns:
            return True
            
        # Check if path matches any include pattern
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
                
        return False


class SearchFilters(BaseModel):
    """Enhanced combined search filters for multiple filter types with validation."""
    
    date_range: Optional[DateRangeFilter] = Field(default=None, description="Date range filter")
    path_filter: Optional[PathFilter] = Field(default=None, description="Path pattern filter")
    language_filter: Optional[str] = Field(default=None, pattern=r'^[a-z]{2}$', description="Language filter")
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum score threshold")
    file_types: Optional[List[str]] = Field(default=None, description="File type filters")
    
    @field_validator('file_types')
    @classmethod
    def validate_file_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate file type extensions."""
        if v is not None:
            normalized = []
            for ext in v:
                if not ext.strip():
                    continue
                if not ext.startswith('.'):
                    ext = '.' + ext
                normalized.append(ext.lower())
            return normalized if normalized else None
        return v
    
    # Removed the validator requiring at least one filter for backward compatibility
    
    @classmethod
    def from_dict(cls, filter_dict: Dict[str, Any]) -> "SearchFilters":
        """Create SearchFilters from dictionary representation.
        
        Args:
            filter_dict: Dictionary containing filter parameters
            
        Returns:
            SearchFilters instance
            
        Raises:
            ValueError: If filter parameters are invalid

        """
        date_range = None
        path_filter = None
        
        # Parse date range filter
        if "date_range" in filter_dict:
            date_range_dict = filter_dict["date_range"]
            start_date = None
            end_date = None
            
            if "start" in date_range_dict:
                start_date = cls._parse_date(date_range_dict["start"])
            if "end" in date_range_dict:
                end_date = cls._parse_date(date_range_dict["end"])
                
            field = date_range_dict.get("field", "modified_at")
            date_range = DateRangeFilter(start=start_date, end=end_date, field=field)
        
        # Parse path filter
        if "path_filter" in filter_dict:
            path_filter_dict = filter_dict["path_filter"]
            include_patterns = path_filter_dict.get("include_patterns")
            exclude_patterns = path_filter_dict.get("exclude_patterns")
            path_filter = PathFilter(include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        
        return cls(date_range=date_range, path_filter=path_filter)
    
    @staticmethod
    def _parse_date(date_input: Union[str, datetime]) -> datetime:
        """Parse date from string or datetime object.
        
        Args:
            date_input: Date string (ISO format) or datetime object
            
        Returns:
            Parsed datetime object
            
        Raises:
            ValueError: If date format is invalid

        """
        if isinstance(date_input, datetime):
            return date_input
            
        if isinstance(date_input, str):
            # Try common date formats
            formats = [
                "%Y-%m-%d",           # 2024-01-01
                "%Y-%m-%dT%H:%M:%S",  # 2024-01-01T12:00:00
                "%Y-%m-%d %H:%M:%S",  # 2024-01-01 12:00:00
                "%Y-%m-%dT%H:%M:%S.%f",  # 2024-01-01T12:00:00.123456
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_input, fmt)
                except ValueError:
                    continue
                    
            # Try ISO format parsing
            try:
                return datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            except ValueError:
                pass
                
            raise ValueError(f"Invalid date format: {date_input}. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        raise ValueError(f"Invalid date type: {type(date_input)}. Must be string or datetime")
    
    def has_filters(self) -> bool:
        """Check if any filters are active.
        
        Returns:
            True if any filters are set

        """
        return (
            self.date_range is not None
            or self.path_filter is not None
            or self.language_filter is not None
            or self.min_score is not None
            or self.file_types is not None
        )
    
    def to_sql_conditions(self) -> tuple[List[str], List[Any]]:
        """Convert filters to SQL WHERE conditions and parameters.
        
        Returns:
            Tuple of (conditions_list, parameters_list)

        """
        conditions = []
        parameters = []
        
        # Add date range conditions
        if self.date_range:
            if self.date_range.start:
                conditions.append(f"c.{self.date_range.field} >= ?")
                parameters.append(self.date_range.start.isoformat())
            
            if self.date_range.end:
                conditions.append(f"c.{self.date_range.field} <= ?")
                parameters.append(self.date_range.end.isoformat())
        
        # Add path filtering conditions
        if self.path_filter:
            path_conditions = []
            
            # Include patterns
            if self.path_filter.include_patterns:
                include_conditions = []
                for pattern in self.path_filter.include_patterns:
                    # Convert shell-style wildcards to SQL LIKE patterns
                    sql_pattern = pattern.replace('*', '%').replace('?', '_')
                    include_conditions.append("c.path LIKE ?")
                    parameters.append(sql_pattern)
                
                if include_conditions:
                    path_conditions.append(f"({' OR '.join(include_conditions)})")
            
            # Exclude patterns
            if self.path_filter.exclude_patterns:
                for pattern in self.path_filter.exclude_patterns:
                    sql_pattern = pattern.replace('*', '%').replace('?', '_')
                    path_conditions.append("c.path NOT LIKE ?")
                    parameters.append(sql_pattern)
            
            if path_conditions:
                conditions.extend(path_conditions)
        
        return conditions, parameters
