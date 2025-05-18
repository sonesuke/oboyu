"""Tests for the Oboyu CLI index manage commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from oboyu.cli.index import manage_app

runner = CliRunner()


@pytest.fixture
def mock_indexer():
    """Fixture for mocking the Indexer class."""
    with patch("oboyu.cli.index.Indexer") as mock_indexer_class:
        mock_indexer = mock_indexer_class.return_value
        yield mock_indexer


def test_clear_command(mock_indexer):
    """Test the clear command."""
    # Create a test context to provide to the command
    ctx = {"config_data": {"indexer": {"db_path": "test.db"}}}
    
    # Run the command with force option to bypass confirmation
    result = runner.invoke(manage_app, ["--force"], obj=ctx)
    
    # Print debug information
    print(f"Exit code: {result.exit_code}")
    print(f"Stdout: {result.stdout}")
    print(f"Exception: {result.exception}")
    
    # Check the command succeeded
    assert result.exit_code == 0
    
    # Check the clear_index method was called
    mock_indexer.clear_index.assert_called_once()
    
    # Check the success message is displayed
    assert "Index database cleared successfully!" in result.stdout


def test_clear_command_confirmation_yes(mock_indexer):
    """Test the clear command with confirmation (yes)."""
    # Create a test context
    ctx = {"config_data": {"indexer": {"db_path": "test.db"}}}
    
    # Run the command with confirmation (simulate user entering 'y')
    result = runner.invoke(manage_app, [], input="y\n", obj=ctx)
    
    # Check the command succeeded
    assert result.exit_code == 0
    
    # Check the warning message is displayed
    assert "Warning" in result.stdout
    
    # Check the clear_index method was called
    mock_indexer.clear_index.assert_called_once()


def test_clear_command_confirmation_no(mock_indexer):
    """Test the clear command with confirmation (no)."""
    # Create a test context
    ctx = {"config_data": {"indexer": {"db_path": "test.db"}}}
    
    # Run the command with confirmation (simulate user entering 'n')
    result = runner.invoke(manage_app, [], input="n\n", obj=ctx)
    
    # Check the command succeeded
    assert result.exit_code == 0
    
    # Check the warning message is displayed
    assert "Warning" in result.stdout
    
    # Check the operation was cancelled
    assert "Operation cancelled" in result.stdout
    
    # Check the clear_index method was not called
    mock_indexer.clear_index.assert_not_called()


def test_clear_command_with_db_path(mock_indexer):
    """Test the clear command with custom database path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "custom.db"
        
        # Create a test context
        ctx = {"config_data": {}}
        
        # Run the command with db_path and force options
        result = runner.invoke(manage_app, ["--db-path", str(db_path), "--force"], obj=ctx)
        
        # Check the command succeeded
        assert result.exit_code == 0
        
        # Check the db_path message is displayed
        assert f"Using explicitly specified database path:" in result.stdout
        assert str(db_path) in result.stdout
        
        # Check the clear_index method was called
        mock_indexer.clear_index.assert_called_once()