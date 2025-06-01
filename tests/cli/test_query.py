"""Tests for the query command and interactive mode."""

from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from oboyu.cli.query import InteractiveQuerySession
from oboyu.indexer.indexer import SearchResult


@pytest.fixture
def mock_indexer():
    """Create a mock indexer for testing."""
    indexer = MagicMock()
    indexer.search.return_value = [
        SearchResult(
            chunk_id="chunk1",
            path="/test/doc1.md",
            title="Test Document 1",
            content="This is test content for document 1",
            chunk_index=0,
            language="en",
            metadata={},
            score=0.95,
        ),
        SearchResult(
            chunk_id="chunk2",
            path="/test/doc2.md",
            title="Test Document 2",
            content="This is test content for document 2",
            chunk_index=0,
            language="en",
            metadata={},
            score=0.85,
        ),
    ]
    indexer.get_statistics.return_value = {
        "total_documents": 100,
        "total_chunks": 500,
        "unique_files": 100,
        "db_size_mb": 50.5,
    }
    return indexer


@pytest.fixture
def interactive_session(mock_indexer):
    """Create an interactive query session for testing."""
    config = {
        "mode": "hybrid",
        "top_k": 5,
        "explain": False,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "use_reranker": False,
        "db_path": "/test/db.db",
    }
    console = Console(force_terminal=True, width=80)
    return InteractiveQuerySession(mock_indexer, config, console)


class TestInteractiveQuerySession:
    """Test cases for InteractiveQuerySession."""

    def test_initialization(self, interactive_session):
        """Test that the session initializes correctly."""
        assert interactive_session.indexer is not None
        assert interactive_session.config["mode"] == "hybrid"
        assert interactive_session.config["top_k"] == 5
        assert isinstance(interactive_session.session, PromptSession)

    def test_is_command(self, interactive_session):
        """Test command detection."""
        assert interactive_session._is_command("/help")
        assert interactive_session._is_command("/exit")
        assert interactive_session._is_command("/mode vector")
        assert interactive_session._is_command("/topk 10")
        assert not interactive_session._is_command("help")  # without slash
        assert not interactive_session._is_command("machine learning")
        assert not interactive_session._is_command("search query")

    def test_process_command_exit(self, interactive_session):
        """Test exit commands."""
        assert not interactive_session._process_command("/exit")
        assert not interactive_session._process_command("/quit")
        assert not interactive_session._process_command("/q")

    def test_process_command_mode(self, interactive_session):
        """Test mode change command."""
        assert interactive_session._process_command("/mode vector")
        assert interactive_session.config["mode"] == "vector"
        
        assert interactive_session._process_command("/mode bm25")
        assert interactive_session.config["mode"] == "bm25"
        
        assert interactive_session._process_command("/mode hybrid")
        assert interactive_session.config["mode"] == "hybrid"

    def test_process_command_topk(self, interactive_session):
        """Test top-k change command."""
        assert interactive_session._process_command("/topk 10")
        assert interactive_session.config["top_k"] == 10
        
        assert interactive_session._process_command("/top-k 20")
        assert interactive_session.config["top_k"] == 20

    def test_process_command_weights(self, interactive_session):
        """Test weights change command."""
        assert interactive_session._process_command("/weights 0.8 0.2")
        assert interactive_session.config["vector_weight"] == 0.8
        assert interactive_session.config["bm25_weight"] == 0.2

    def test_process_command_rerank(self, interactive_session):
        """Test rerank toggle command."""
        assert interactive_session._process_command("/rerank on")
        assert interactive_session.config["use_reranker"] is True
        
        assert interactive_session._process_command("/rerank off")
        assert interactive_session.config["use_reranker"] is False

    def test_process_query(self, interactive_session, mock_indexer):
        """Test query processing."""
        interactive_session._process_query("test query")
        
        mock_indexer.search.assert_called_once_with(
            "test query",
            limit=5,
            mode="hybrid",
        )

    def test_show_stats(self, interactive_session, mock_indexer):
        """Test statistics display."""
        # This should not raise an exception
        interactive_session._show_stats()
        mock_indexer.get_stats.assert_called_once()

    @patch("subprocess.run")
    def test_clear_command(self, mock_run, interactive_session):
        """Test clear screen command."""
        interactive_session._process_command("/clear")
        mock_run.assert_called_once_with(["/usr/bin/clear"], check=False)

    def test_invalid_commands(self, interactive_session):
        """Test handling of invalid commands."""
        # Invalid mode
        assert interactive_session._process_command("/mode invalid")
        assert interactive_session.config["mode"] == "hybrid"  # Should not change
        
        # Invalid topk
        assert interactive_session._process_command("/topk abc")
        assert interactive_session.config["top_k"] == 5  # Should not change
        
        # Invalid weights
        assert interactive_session._process_command("/weights 1.5 0.5")
        assert interactive_session.config["vector_weight"] == 0.7  # Should not change
        
        # Invalid rerank
        assert interactive_session._process_command("/rerank maybe")
        assert interactive_session.config["use_reranker"] is False  # Should not change


def test_interactive_session_run():
    """Test the interactive session run loop with simulated input."""
    # Set up test environment
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            mock_indexer = MagicMock()
            mock_indexer.search.return_value = []
            mock_indexer.get_stats.return_value = {
                "total_documents": 0,
                "total_chunks": 0,
                "unique_files": 0,
                "db_size_mb": 0,
            }
            
            config = {
                "mode": "hybrid",
                "top_k": 5,
                "explain": False,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "use_reranker": False,
                "db_path": "/test/db.db",
            }
            
            console = Console(force_terminal=False, width=80, force_jupyter=False)
            session = InteractiveQuerySession(mock_indexer, config, console)
            
            # Override the prompt session to use our pipe input
            session.session = PromptSession(input=pipe_input, output=DummyOutput())
            
            # Send commands to the session
            pipe_input.send_text("/mode vector\n")
            pipe_input.send_text("/exit\n")
            
            # Run the session (it should exit after processing the commands)
            session.run()
            
            # Verify the mode was changed
            assert session.config["mode"] == "vector"