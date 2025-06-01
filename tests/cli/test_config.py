"""Tests for the Oboyu CLI configuration."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from oboyu.cli.config import create_default_config, load_config
from oboyu.common.paths import DEFAULT_DB_PATH


def test_load_config_nonexistent() -> None:
    """Test loading a non-existent configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_path = Path(tmpdir) / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(nonexistent_path)


def test_load_config_invalid() -> None:
    """Test loading an invalid configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = Path(tmpdir) / "invalid.yaml"
        with open(invalid_path, "w") as f:
            f.write("invalid: yaml: :")

        with pytest.raises(ValueError):
            load_config(invalid_path)


def test_load_config_valid() -> None:
    """Test loading a valid configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_path = Path(tmpdir) / "valid.yaml"
        config_data = {
            "crawler": {
                "depth": 5,
                "include_patterns": ["*.txt"],
            },
            "indexer": {
                "chunk_size": 512,
                "embedding_model": "test-model",
            },
            "query": {
                "top_k": 10,
                "vector_weight": 0.8,
            },
        }

        with open(valid_path, "w") as f:
            yaml.dump(config_data, f)

        loaded_config = load_config(valid_path)
        assert loaded_config["crawler"]["depth"] == 5
        assert loaded_config["crawler"]["include_patterns"] == ["*.txt"]
        assert loaded_config["indexer"]["chunk_size"] == 512
        assert loaded_config["indexer"]["embedding_model"] == "test-model"
        assert loaded_config["query"]["top_k"] == 10
        assert loaded_config["query"]["vector_weight"] == 0.8


def test_create_default_config() -> None:
    """Test creating a default configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config = create_default_config(config_path)

        # Check file was created
        assert config_path.exists()

        # Check loaded config matches what we expect
        loaded_config = load_config(config_path)
        assert "crawler" in loaded_config
        assert "indexer" in loaded_config
        assert "query" in loaded_config

        # Check specific values
        assert loaded_config["crawler"]["depth"] == 10
        assert "*.txt" in loaded_config["crawler"]["include_patterns"]
        assert loaded_config["indexer"]["chunk_size"] == 1000
        assert loaded_config["query"]["top_k"] == 10
        
        # Note: db_path is not in the indexer defaults - it must be explicitly provided