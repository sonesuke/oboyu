"""Tests for the Oboyu CLI clear command."""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from oboyu.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_config_manager():
    """Fixture for mocking the ConfigManager."""
    with patch("oboyu.cli.main.ConfigManager") as mock_config_manager_class:
        mock_config_manager = mock_config_manager_class.return_value
        mock_config_manager.get_section.return_value = {"db_path": "test.db"}
        mock_config_manager.resolve_db_path.return_value = Path("test.db")
        yield mock_config_manager


def test_clear_command_force(mock_config_manager):
    """Test the clear command with force option."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        wal_path = Path(temp_dir) / "test.db-wal"
        shm_path = Path(temp_dir) / "test.db-shm"
        
        # Create test database files
        db_path.touch()
        wal_path.touch()
        shm_path.touch()
        
        mock_config_manager.resolve_db_path.return_value = db_path
        
        # Create a test context with config manager
        ctx = {"config_manager": mock_config_manager, "config_data": {"indexer": {"db_path": str(db_path)}}}
        
        # Run the command with force option to bypass confirmation
        result = runner.invoke(app, ["clear", "--force"], obj=ctx)
        
        # Check the command succeeded
        assert result.exit_code == 0
        
        # Check the success message is displayed
        assert "Index database cleared successfully!" in result.stdout
        
        # Check files were deleted
        assert not db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()


def test_clear_command_confirmation_yes(mock_config_manager):
    """Test the clear command with confirmation (yes)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        db_path.touch()
        
        mock_config_manager.resolve_db_path.return_value = db_path
        
        # Create a test context
        ctx = {"config_manager": mock_config_manager, "config_data": {"indexer": {"db_path": str(db_path)}}}
        
        # Run the command with confirmation (simulate user entering 'y')
        result = runner.invoke(app, ["clear"], input="y\n", obj=ctx)
        
        # Check the command succeeded
        assert result.exit_code == 0
        
        # Check the warning message is displayed
        assert "Warning" in result.stdout
        
        # Check file was deleted
        assert not db_path.exists()


def test_clear_command_confirmation_no(mock_config_manager):
    """Test the clear command with confirmation (no)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        db_path.touch()
        
        mock_config_manager.resolve_db_path.return_value = db_path
        
        # Create a test context
        ctx = {"config_manager": mock_config_manager, "config_data": {"indexer": {"db_path": str(db_path)}}}
        
        # Run the command with confirmation (simulate user entering 'n')
        result = runner.invoke(app, ["clear"], input="n\n", obj=ctx)
        
        # Check the command succeeded
        assert result.exit_code == 0
        
        # Check the warning message is displayed
        assert "Warning" in result.stdout
        
        # Check the operation was cancelled
        assert "Operation cancelled" in result.stdout
        
        # Check file was not deleted
        assert db_path.exists()


def test_clear_command_with_db_path(mock_config_manager):
    """Test the clear command with custom database path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "custom.db"
        db_path.touch()
        
        mock_config_manager.resolve_db_path.return_value = db_path
        
        # Create a test context
        ctx = {"config_manager": mock_config_manager, "config_data": {}}
        
        # Run the command with db_path and force options
        result = runner.invoke(app, ["clear", "--db-path", str(db_path), "--force"], obj=ctx)
        
        # Check the command succeeded
        assert result.exit_code == 0
        
        # Check the db_path message is displayed
        assert f"Using database:" in result.stdout
        assert str(db_path) in result.stdout
        
        # Check file was deleted
        assert not db_path.exists()


def test_clear_command_no_files():
    """Test the clear command when no database files exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "nonexistent.db"
        
        with patch("oboyu.cli.main.ConfigManager") as mock_config_manager_class:
            mock_config_manager = mock_config_manager_class.return_value
            mock_config_manager.get_section.return_value = {"db_path": str(db_path)}
            mock_config_manager.resolve_db_path.return_value = db_path
            
            # Create a test context
            ctx = {"config_manager": mock_config_manager, "config_data": {}}
            
            # Run the command with force option
            result = runner.invoke(app, ["clear", "--force"], obj=ctx)
            
            # Check the command succeeded
            assert result.exit_code == 0
            
            # Check the appropriate message is displayed
            assert "No database files found to delete" in result.stdout
            assert "Index database was already clear" in result.stdout


def test_clear_command_partial_files():
    """Test the clear command when only some database files exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        wal_path = Path(temp_dir) / "test.db-wal"
        
        # Create only the main database file and WAL file
        db_path.touch()
        wal_path.touch()
        
        with patch("oboyu.cli.main.ConfigManager") as mock_config_manager_class:
            mock_config_manager = mock_config_manager_class.return_value
            mock_config_manager.get_section.return_value = {"db_path": str(db_path)}
            mock_config_manager.resolve_db_path.return_value = db_path
            
            # Create a test context
            ctx = {"config_manager": mock_config_manager, "config_data": {}}
            
            # Run the command with force option
            result = runner.invoke(app, ["clear", "--force"], obj=ctx)
            
            # Check the command succeeded
            assert result.exit_code == 0
            
            # Check the success message is displayed
            assert "Deleted 2 database file(s)" in result.stdout
            assert "Index database cleared successfully!" in result.stdout
            
            # Check files were deleted
            assert not db_path.exists()
            assert not wal_path.exists()