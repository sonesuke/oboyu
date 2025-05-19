"""Tests for the MCP CLI module."""

import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from oboyu.cli.main import app
from oboyu.cli.mcp import app as mcp_app


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@patch("oboyu.mcp.context.mcp.run")
def test_mcp_command_default_options(mock_run, runner):
    """Test the MCP command with default options."""
    # Call the MCP command
    result = runner.invoke(mcp_app, [])
    
    # Check that the command exited successfully
    assert result.exit_code == 0
    
    # Verify mcp.run was called with the default transport
    mock_run.assert_called_once_with("stdio")


@patch("oboyu.mcp.context.mcp.run")
def test_mcp_command_with_transport(mock_run, runner):
    """Test the MCP command with transport option."""
    # Call the MCP command with transport option
    result = runner.invoke(mcp_app, ["--transport", "sse", "--port", "8000"])
    
    # Check that the command exited successfully
    assert result.exit_code == 0
    
    # Verify mcp.run was called with the specified transport
    # Note: The implementation now only passes the transport to run()
    mock_run.assert_called_once_with("sse")


@patch("oboyu.mcp.context.mcp.run")
def test_mcp_command_with_debug(mock_run, runner):
    """Test the MCP command with debug option."""
    # Call the MCP command with debug option
    result = runner.invoke(mcp_app, ["--debug"])
    
    # Check that the command exited successfully
    assert result.exit_code == 0
    
    # Verify mcp.run was called with the default transport
    # Note: The implementation now only passes the transport to run()
    mock_run.assert_called_once_with("stdio")


@patch("oboyu.mcp.context.mcp.run")
def test_mcp_command_with_db_path(mock_run, runner):
    """Test the MCP command with db_path option."""
    # Call the MCP command with db_path option
    result = runner.invoke(mcp_app, ["--db-path", "/custom/path.db"])
    
    # Check that the command exited successfully
    assert result.exit_code == 0
    
    # Verify db_path_global was set correctly
    from oboyu.mcp.context import db_path_global
    assert db_path_global.value == "/custom/path.db"
    
    # Should still use default transport
    mock_run.assert_called_once_with("stdio")


@patch("oboyu.mcp.context.mcp.run")
def test_mcp_command_error_handling(mock_run, runner):
    """Test error handling in the MCP command."""
    # Setup the mock to raise an exception
    mock_run.side_effect = Exception("Test error")
    
    # Call the MCP command
    result = runner.invoke(mcp_app, [])
    
    # Check that the command failed
    assert result.exit_code == 1
    
    # Verify error message was displayed
    assert "Error starting MCP server: Test error" in result.stdout