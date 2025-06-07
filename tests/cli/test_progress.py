"""Tests for the progress tracking system."""

import time
from unittest.mock import Mock, patch

import pytest

from oboyu.cli.hierarchical_logger import HierarchicalLogger
from oboyu.cli.progress import (
    IndexerProgressAdapter,
    ProgressPipeline,
    ProgressStage,
    create_indexer_progress_callback,
)


class TestProgressStage:
    """Test ProgressStage class."""

    def test_progress_stage_init(self):
        """Test ProgressStage initialization."""
        stage = ProgressStage("test", "Test stage", total=100)
        assert stage.name == "test"
        assert stage.description == "Test stage"
        assert stage.total == 100
        assert stage.current == 0
        assert stage.operation_id is None

    def test_progress_stage_properties(self):
        """Test ProgressStage properties."""
        stage = ProgressStage("test", "Test stage", total=100)
        stage.current = 50
        
        # Test elapsed time (should be > 0)
        assert stage.elapsed > 0
        
        # Test rate calculation
        time.sleep(0.01)  # Small delay to ensure elapsed > 0
        stage.current = 10
        assert stage.rate > 0
        
        # Test progress percentage
        assert stage.progress_percent == 10.0
        
        # Test ETA
        assert stage.eta_seconds > 0


class TestProgressPipeline:
    """Test ProgressPipeline class."""

    @pytest.fixture
    def mock_logger(self):
        """Mock HierarchicalLogger."""
        return Mock(spec=HierarchicalLogger)

    @pytest.fixture
    def pipeline(self, mock_logger):
        """Create ProgressPipeline instance."""
        return ProgressPipeline(mock_logger)

    def test_add_stage(self, pipeline):
        """Test adding stages to pipeline."""
        pipeline.add_stage("test_stage", "Test Stage", total=100)
        
        assert "test_stage" in pipeline.stages
        stage = pipeline.stages["test_stage"]
        assert stage.name == "test_stage"
        assert stage.description == "Test Stage"
        assert stage.total == 100

    def test_start_stage(self, pipeline, mock_logger):
        """Test starting a stage."""
        pipeline.add_stage("test_stage", "Test Stage", total=100)
        mock_logger.start_operation.return_value = "op_id_123"
        
        pipeline.start_stage("test_stage")
        
        mock_logger.start_operation.assert_called_once_with("Test Stage")
        assert pipeline.stages["test_stage"].operation_id == "op_id_123"
        assert pipeline.active_stage == "test_stage"

    def test_update_stage(self, pipeline, mock_logger):
        """Test updating stage progress."""
        pipeline.add_stage("test_stage", "Test Stage")
        mock_logger.start_operation.return_value = "op_id_123"
        
        # Update should auto-start the stage
        pipeline.update("test_stage", 50, 100)
        
        mock_logger.start_operation.assert_called_once()
        mock_logger.update_operation.assert_called_once()
        
        stage = pipeline.stages["test_stage"]
        assert stage.current == 50
        assert stage.total == 100

    def test_complete_stage(self, pipeline, mock_logger):
        """Test completing a stage."""
        pipeline.add_stage("test_stage", "Test Stage")
        pipeline.stages["test_stage"].operation_id = "op_id_123"
        pipeline.active_stage = "test_stage"
        
        pipeline.complete_stage("test_stage")
        
        mock_logger.complete_operation.assert_called_once_with("op_id_123")
        assert pipeline.stages["test_stage"].operation_id is None
        assert pipeline.active_stage is None

    def test_auto_complete_on_total(self, pipeline, mock_logger):
        """Test auto-completion when reaching total."""
        pipeline.add_stage("test_stage", "Test Stage", total=100)
        mock_logger.start_operation.return_value = "op_id_123"
        
        # Update to total should auto-complete
        pipeline.update("test_stage", 100, 100)
        
        mock_logger.complete_operation.assert_called_once_with("op_id_123")

    def test_completion_shows_100_percent(self, pipeline, mock_logger):
        """Test that completion always shows 100% progress before marking complete."""
        pipeline.add_stage("test_stage", "Test Stage", total=100)
        mock_logger.start_operation.return_value = "op_id_123"
        
        # Update to total should show final progress message with 100%
        pipeline.update("test_stage", 100, 100)
        
        # Should have called update_operation twice:
        # 1. When auto-starting the stage 
        # 2. Final progress message with 100% before completion
        assert mock_logger.update_operation.call_count == 2
        
        # Check the final call shows 100%
        final_call_args = mock_logger.update_operation.call_args_list[-1]
        final_message = final_call_args[0][1]  # Second argument is the message
        assert "100%" in final_message
        
        # Should then mark as complete
        mock_logger.complete_operation.assert_called_once_with("op_id_123")


class TestIndexerProgressAdapter:
    """Test IndexerProgressAdapter class."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock ProgressPipeline."""
        mock = Mock(spec=ProgressPipeline)
        mock.stages = {}
        return mock

    @pytest.fixture
    def adapter(self, mock_pipeline):
        """Create IndexerProgressAdapter instance."""
        # Mock the stages attribute to prevent initialization errors
        stage_mock = Mock()
        stage_mock.metadata = {}
        mock_pipeline.stages = {stage_name: stage_mock for stage_name in IndexerProgressAdapter.STAGE_CONFIG}
        return IndexerProgressAdapter(mock_pipeline, "scan_op_123")

    def test_callback_regular_stage(self, adapter, mock_pipeline):
        """Test callback for regular stages."""
        adapter.callback("crawling", 50, 100)
        
        mock_pipeline.update.assert_called_once_with("crawling", 50, 100)

    def test_callback_bm25_storage_stage(self, adapter, mock_pipeline):
        """Test callback for BM25 storage stages."""
        adapter.callback("bm25_storing_vocabulary", 25, 50)
        
        mock_pipeline.update.assert_called_once_with("bm25_store_vocabulary", 25, 50)

    def test_callback_bm25_index_creation(self, adapter, mock_pipeline):
        """Test callback for BM25 index creation."""
        adapter.callback("bm25_storing_creating_indexes", 1, 2)
        
        # This stage should be mapped to bm25_creating_indexes
        mock_pipeline.update.assert_called_once_with("bm25_creating_indexes", 1, 2)

    def test_callback_crawling_completion(self, adapter, mock_pipeline):
        """Test callback completes scan operation on crawling completion."""
        # Mock the logger
        mock_logger = Mock()
        mock_pipeline.logger = mock_logger
        
        adapter.callback("crawling", 100, 100)
        
        # Should complete the scan operation
        mock_logger.complete_operation.assert_called_once_with("scan_op_123")


class TestCreateIndexerProgressCallback:
    """Test create_indexer_progress_callback function."""

    @patch("oboyu.cli.progress.ProgressPipeline")
    @patch("oboyu.cli.progress.IndexerProgressAdapter")
    def test_create_callback(self, mock_adapter_class, mock_pipeline_class):
        """Test creating indexer progress callback."""
        mock_logger = Mock(spec=HierarchicalLogger)
        mock_pipeline = Mock()
        mock_adapter = Mock()
        
        mock_pipeline_class.return_value = mock_pipeline
        mock_adapter_class.return_value = mock_adapter
        
        callback = create_indexer_progress_callback(mock_logger, "scan_op_123")
        
        # Should create pipeline and adapter
        mock_pipeline_class.assert_called_once_with(mock_logger)
        mock_adapter_class.assert_called_once_with(mock_pipeline, "scan_op_123")
        
        # Should return the adapter callback
        assert callback == mock_adapter.callback