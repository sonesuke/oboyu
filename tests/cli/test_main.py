"""Tests for the Oboyu CLI main module."""

from typer.testing import CliRunner

from oboyu.cli.main import app

runner = CliRunner()


def test_version_command() -> None:
    """Test the version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Oboyu version:" in result.stdout


def test_help_command() -> None:
    """Test the help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    
    # Check for index and query subcommands
    assert "index" in result.stdout
    assert "query" in result.stdout


def test_index_help() -> None:
    """Test the index help command."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    
    # Check for recursive option (the exact format might vary)
    assert "recursive" in result.stdout.lower()


def test_query_help() -> None:
    """Test the query help command."""
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    
    # Check for important option terms (the exact format might vary)
    assert "mode" in result.stdout.lower()