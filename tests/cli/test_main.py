"""Tests for the Oboyu CLI main module."""

from typer.testing import CliRunner

from oboyu.cli.main import app

runner = CliRunner()


def test_version_command() -> None:
    """Test the version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Oboyu version:" in result.stdout


def test_global_db_path_option() -> None:
    """Test the global database path option is passed to subcommands."""
    # We can't fully test command execution since that would require setting up
    # a real database, but we can check the option is recognized
    result = runner.invoke(app, ["--db-path", "/path/to/custom.db", "--help"])
    assert result.exit_code == 0


def test_help_command() -> None:
    """Test the help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout

    # Check for main commands in new structure
    assert "index" in result.stdout
    assert "search" in result.stdout  # Changed from query to search
    assert "build-kg" in result.stdout
    assert "deduplicate" in result.stdout

    # Check for global database path option
    assert "--db-path" in result.stdout


def test_index_help() -> None:
    """Test the index help command."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout

    # Check for recursive option (the exact format might vary)
    assert "recursive" in result.stdout.lower()

    # Check for db-path option
    assert "--db-path" in result.stdout


def test_search_help() -> None:
    """Test the search help command."""
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout

    # Check for important option terms (the exact format might vary)
    assert "mode" in result.stdout.lower()
    assert "--db-path" in result.stdout
    assert "--no-graph" in result.stdout  # New option for disabling GraphRAG


def test_build_kg_help() -> None:
    """Test the build-kg help command."""
    result = runner.invoke(app, ["build-kg", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    
    # Check for build-kg specific options
    assert "--full" in result.stdout
    assert "--batch-size" in result.stdout


def test_deduplicate_help() -> None:
    """Test the deduplicate help command."""
    result = runner.invoke(app, ["deduplicate", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    
    # Check for deduplicate specific options
    assert "--type" in result.stdout
    assert "--similarity" in result.stdout
