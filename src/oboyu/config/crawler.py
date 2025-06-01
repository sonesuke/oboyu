"""Consolidated crawler configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class CrawlerConfig:
    """Crawler configuration."""

    # Basic settings
    depth: int = 10
    max_workers: int = 4
    respect_gitignore: bool = True
    
    # File filtering
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    
    # Processing settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    follow_symlinks: bool = False
    timeout: int = 30
    
    # Content settings
    min_doc_length: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200
    encoding: str = "utf-8"
    use_japanese_tokenizer: bool = True
    
    # Directories and extensions
    exclude_dirs: List[str] = None
    include_extensions: List[str] = None

    def __post_init__(self) -> None:
        """Post-initialization to set default lists."""
        if self.include_patterns is None:
            self.include_patterns = [
                "*.txt",
                "*.md", 
                "*.html",
                "*.py",
                "*.java",
            ]
            
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "*/node_modules/*",
                "*/.venv/*",
                "*/__pycache__/*",
                "*/.git/*",
            ]
            
        if self.exclude_dirs is None:
            self.exclude_dirs = [
                "__pycache__",
                ".git", 
                "node_modules",
                ".venv",
                "venv",
            ]
            
        if self.include_extensions is None:
            self.include_extensions = [
                ".py", ".md", ".txt", ".yaml", ".yml", 
                ".json", ".toml", ".cfg", ".ini", ".rst", ".ipynb"
            ]

    def validate(self) -> None:
        """Validate configuration values."""
        if self.depth <= 0:
            self.depth = 10
            
        if self.max_workers <= 0:
            self.max_workers = 4
            
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = min(200, self.chunk_size // 2)


# Default configuration
DEFAULT_CONFIG = {
    "crawler": CrawlerConfig()
}