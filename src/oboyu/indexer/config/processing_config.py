"""Document processing configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ProcessingConfig:
    """Document processing configuration."""

    # Document chunking settings
    chunk_size: int = 1024
    chunk_overlap: int = 256

    # Processing settings
    max_workers: int = 4

    # Database settings
    db_path: Union[str, Path] = "index.db"

    # VSS (Vector Similarity Search) settings
    ef_construction: int = 128
    ef_search: int = 64
    m: int = 16
    m0: Optional[int] = None

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if self.m0 is None:
            self.m0 = 2 * self.m

        # Ensure db_path is a Path object
        self.db_path = Path(self.db_path)
