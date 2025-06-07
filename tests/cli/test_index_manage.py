"""Tests for the Oboyu CLI index manage commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from oboyu.cli.manage import app as manage_app
from oboyu.common.config import ConfigManager

runner = CliRunner()


@pytest.fixture
def mock_indexer():
    """Fixture for mocking the Indexer class."""
    with patch("oboyu.cli.index.Indexer") as mock_indexer_class:
        mock_indexer = mock_indexer_class.return_value
        yield mock_indexer


# Clear command tests have been moved to test_clear.py
# since the clear command is now a top-level command