"""Simplified progress tracking system for Oboyu CLI.

This module provides a clean progress pipeline architecture that
replaces the complex callback-based system.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from oboyu.cli.hierarchical_logger import HierarchicalLogger


@dataclass
class ProgressStage:
    """Represents a single progress stage."""

    name: str
    description: str
    total: Optional[int] = None
    current: int = 0
    operation_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _last_update_time: float = field(default=0.0)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        """Calculate processing rate (items per second)."""
        if self.current > 0 and self.elapsed > 0:
            return self.current / self.elapsed
        return 0.0

    @property
    def eta_seconds(self) -> float:
        """Calculate estimated time to completion in seconds."""
        if self.rate > 0 and self.total and self.current < self.total:
            return (self.total - self.current) / self.rate
        return 0.0

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total and self.total > 0:
            return (self.current / self.total) * 100
        return 0.0


class ProgressPipeline:
    """Simple progress pipeline with clear stage definitions."""

    def __init__(self, logger: HierarchicalLogger) -> None:
        """Initialize the progress pipeline.

        Args:
            logger: HierarchicalLogger instance for output

        """
        self.logger = logger
        self.stages: Dict[str, ProgressStage] = {}
        self.active_stage: Optional[str] = None

    def add_stage(self, name: str, description: str, total: Optional[int] = None) -> None:
        """Add a progress stage to the pipeline.

        Args:
            name: Unique name for the stage
            description: Human-readable description
            total: Total number of items (if known)

        """
        self.stages[name] = ProgressStage(
            name=name,
            description=description,
            total=total,
        )

    def start_stage(self, name: str) -> None:
        """Start a specific stage.

        Args:
            name: Name of the stage to start

        """
        if name not in self.stages:
            return

        stage = self.stages[name]

        # Complete any active stage
        if self.active_stage and self.active_stage != name:
            self.complete_stage(self.active_stage)

        # Start the new stage
        stage.operation_id = self.logger.start_operation(stage.description)
        stage.start_time = time.time()
        self.active_stage = name

    def update(self, stage_name: str, current: int, total: Optional[int] = None) -> None:
        """Update progress for a specific stage.

        Args:
            stage_name: Name of the stage to update
            current: Current progress value
            total: Total items (updates stage total if provided)

        """
        if stage_name not in self.stages:
            return

        stage = self.stages[stage_name]

        # Update total if provided
        if total is not None:
            stage.total = total

        # Update current progress
        stage.current = current

        # Start stage if not active
        if not stage.operation_id:
            self.start_stage(stage_name)

        # Frequent updates for smooth real-time feedback
        should_update = (
            current == 1  # First item
            or (total is not None and current >= total)  # Completion
            or (total is not None and stage.elapsed > 1.0 and current % max(total // 40, 25) == 0)  # Every 2.5% or 25 items after 1s
            or (time.time() - getattr(stage, "_last_update_time", 0) > 1.5)  # Every 1.5 seconds max
        )

        # Format progress message only when updating
        if stage.operation_id and should_update:
            message = self._format_progress_message(stage)
            self.logger.update_operation(stage.operation_id, message)
            stage._last_update_time = time.time()

        # Auto-complete when reaching total
        if stage.total and current >= stage.total:
            # Ensure final progress message shows 100% before completion
            if stage.operation_id:
                final_message = self._format_progress_message(stage)
                self.logger.update_operation(stage.operation_id, final_message)
            self.complete_stage(stage_name)

    def complete_stage(self, name: str) -> None:
        """Mark a stage as complete.

        Args:
            name: Name of the stage to complete

        """
        if name not in self.stages:
            return

        stage = self.stages[name]

        if stage.operation_id:
            self.logger.complete_operation(stage.operation_id)
            stage.operation_id = None

        if self.active_stage == name:
            self.active_stage = None

    def _format_progress_message(self, stage: ProgressStage) -> str:
        """Format a progress message for display.

        Args:
            stage: ProgressStage to format

        Returns:
            Formatted progress message

        """
        parts = [stage.description]

        # Add progress counts
        if stage.total:
            parts.append(f"{stage.current}/{stage.total}")
            parts.append(f"({stage.progress_percent:.0f}%)")

        # Add rate and ETA for longer running operations
        if stage.elapsed > 2 and stage.rate > 0:
            # Format rate based on stage type
            rate_unit = stage.metadata.get("rate_unit", "items/sec")
            parts.append(f"{stage.rate:.0f} {rate_unit}")

            # Add ETA if meaningful
            if stage.eta_seconds > 0:
                parts.append(f"ETA: {stage.eta_seconds:.0f}s")

        return " ".join(parts)


class IndexerProgressAdapter:
    """Adapter to convert indexer callbacks to pipeline updates."""

    # Stage configurations with descriptions and metadata
    STAGE_CONFIG: Dict[str, Dict[str, Union[str, Dict[int, str]]]] = {
        "crawling": {
            "description": "Scanning directories...",
            "rate_unit": "files/sec",
        },
        "processing": {
            "description": "Reading and chunking files...",
            "rate_unit": "files/sec",
        },
        "embedding": {
            "description": "Generating embeddings...",
            "rate_unit": "batches/sec",
        },
        "storing": {
            "description": "Storing chunks in database...",
            "rate_unit": "chunks/sec",
        },
        "storing_embeddings": {
            "description": "Storing embeddings in database...",
            "rate_unit": "embeddings/sec",
        },
        "bm25_indexing": {
            "description": "Building BM25 search index...",
            "substages": {
                1: "Tokenizing documents... (this may take a while for Japanese text)",
                2: "Building vocabulary...",
                3: "Filtering low-frequency terms...",
                4: "Storing index in database...",
            },
        },
        "bm25_tokenizing": {
            "description": "Tokenizing chunks for BM25 index...",
            "rate_unit": "chunks/sec",
        },
        "bm25_vocabulary": {
            "description": "Building vocabulary...",
            "rate_unit": "terms/sec",
        },
        "bm25_filtering": {
            "description": "Filtering low-frequency terms...",
            "rate_unit": "terms/sec",
        },
        "bm25_store_vocabulary": {
            "description": "Storing vocabulary terms...",
            "rate_unit": "terms/sec",
        },
        "bm25_store_inverted_index": {
            "description": "Storing inverted index entries...",
            "rate_unit": "entries/sec",
        },
        "bm25_store_document_stats": {
            "description": "Storing document statistics...",
            "rate_unit": "docs/sec",
        },
        "bm25_creating_indexes": {
            "description": "Creating database indexes...",
            "substages": {
                1: "Creating term index...",
                2: "Creating term-chunk index...",
            },
        },
    }

    def __init__(self, pipeline: ProgressPipeline, scan_operation_id: str) -> None:
        """Initialize the adapter.

        Args:
            pipeline: ProgressPipeline instance
            scan_operation_id: ID of the initial scan operation

        """
        self.pipeline = pipeline
        self.scan_operation_id = scan_operation_id
        self._last_stage: Optional[str] = None
        self._initialize_stages()

    def _initialize_stages(self) -> None:
        """Initialize all known stages in the pipeline."""
        for stage_name, config in self.STAGE_CONFIG.items():
            description = config["description"]
            if isinstance(description, str):
                self.pipeline.add_stage(
                    name=stage_name,
                    description=description,
                )
            # Add metadata
            stage = self.pipeline.stages[stage_name]
            if "rate_unit" in config:
                stage.metadata["rate_unit"] = config["rate_unit"]
            if "substages" in config:
                stage.metadata["substages"] = config["substages"]

    def callback(self, stage: str, current: int, total: int) -> None:
        """Delegate stage progress to pipeline.

        Args:
            stage: Stage name from indexer
            current: Current progress
            total: Total items

        """
        # Map BM25 storage stage names
        if stage == "bm25_storing_creating_indexes":
            # Map to the correct stage name
            self.pipeline.update("bm25_creating_indexes", current, total)
        elif stage.startswith("bm25_storing_"):
            # Convert "bm25_storing_vocabulary" to "bm25_store_vocabulary"
            mapped_stage = stage.replace("bm25_storing_", "bm25_store_")
            if mapped_stage in self.STAGE_CONFIG:
                self.pipeline.update(mapped_stage, current, total)
        elif stage in self.STAGE_CONFIG:
            # Handle regular stages
            self.pipeline.update(stage, current, total)

            # Special handling for BM25 indexing substages
            if stage == "bm25_indexing" and current > 0:
                config = self.STAGE_CONFIG[stage]
                if "substages" in config and current in config["substages"]:
                    # Update the stage description for substages
                    stage_obj = self.pipeline.stages[stage]
                    if stage_obj.operation_id:
                        substage_desc = config["substages"][current]
                        self.pipeline.logger.update_operation(stage_obj.operation_id, substage_desc)

        # Handle initial scan completion
        if stage == "crawling" and current >= total and self.scan_operation_id:
            self.pipeline.logger.complete_operation(self.scan_operation_id)
            self.scan_operation_id = ""

        self._last_stage = stage


def create_indexer_progress_callback(logger: HierarchicalLogger, scan_operation_id: str) -> Callable[[str, int, int], None]:
    """Create a simplified progress callback for the indexer.

    Args:
        logger: HierarchicalLogger instance
        scan_operation_id: ID of the initial scan operation

    Returns:
        Progress callback function

    """
    pipeline = ProgressPipeline(logger)
    adapter = IndexerProgressAdapter(pipeline, scan_operation_id)
    return adapter.callback
