"""Global test configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator
import pytest
import os

@pytest.fixture(scope="function")
def temp_db_path() -> Generator[Path, None, None]:
    """Provide a temporary database path for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        yield db_path


@pytest.fixture(scope="function")
def temp_working_dir() -> Generator[Path, None, None]:
    """Provide a temporary working directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(autouse=True)
def clean_project_root():
    """Automatically clean up database files from project root after each test."""
    # This fixture runs automatically for every test
    yield
    
    # Cleanup after test
    project_root = Path(__file__).parent.parent
    db_files = [
        "test.db", "test.db.wal",
        "index.db", "index.db.wal", 
        "oboyu.db", "oboyu.db.wal"
    ]
    
    for db_file in db_files:
        db_path = project_root / db_file
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture(autouse=True)
def isolate_config_for_tests(monkeypatch):
    """Isolate configuration for tests to prevent interference."""
    # Set a temporary config directory for tests
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("XDG_CONFIG_HOME", temp_dir)
        yield