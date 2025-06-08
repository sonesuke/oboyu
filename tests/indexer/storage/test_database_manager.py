"""Tests for DatabaseManager."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from oboyu.indexer.storage.database_manager import DatabaseManager
from oboyu.indexer.storage.index_manager import HNSWIndexParams


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        yield Path(tmp.name)
    # Cleanup
    try:
        Path(tmp.name).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def database_manager(temp_db_path):
    """Create a DatabaseManager instance."""
    return DatabaseManager(
        db_path=temp_db_path,
        embedding_dimensions=256,
        hnsw_params=HNSWIndexParams(
            ef_construction=128,
            ef_search=64,
            m=16,
        ),
    )


def test_database_manager_init(database_manager, temp_db_path):
    """Test DatabaseManager initialization."""
    assert database_manager.db_path == temp_db_path
    assert database_manager.embedding_dimensions == 256
    assert database_manager.hnsw_params.ef_construction == 128
    assert database_manager.hnsw_params.ef_search == 64
    assert database_manager.hnsw_params.m == 16
    assert database_manager._is_initialized is False
    assert database_manager.conn is None


def test_database_manager_init_default_params():
    """Test DatabaseManager initialization with default parameters."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        manager = DatabaseManager(db_path=tmp.name)
        
        assert manager.embedding_dimensions == 256
        assert manager.hnsw_params.ef_construction == 128
        assert manager.hnsw_params.ef_search == 64
        assert manager.hnsw_params.m == 16
        assert manager.auto_vacuum is True
        assert manager.enable_experimental_features is True


@patch("oboyu.indexer.storage.database_manager.MigrationManager")
@patch("oboyu.indexer.storage.database_manager.IndexManager")
def test_initialize_success(mock_index, mock_migration):
    """Test successful database initialization using state manager."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        # Create database manager
        with patch("oboyu.indexer.storage.database_manager.DatabaseStateManager") as mock_state_manager:
            mock_conn = MagicMock()
            
            # Mock state manager behavior
            mock_state_instance = MagicMock()
            mock_state_manager.return_value = mock_state_instance
            mock_state_instance.ensure_initialized.return_value = mock_conn
            
            # Create database manager with mocked state manager
            database_manager = DatabaseManager(db_path=tmp.name)
            
            # Initialize the database manager
            database_manager.initialize()
            
            # Verify initialization status
            assert database_manager._is_initialized is True
            assert database_manager.conn is mock_conn
            
            # Verify state manager was used for initialization
            mock_state_instance.ensure_initialized.assert_called_once()
            
            # Verify managers were initialized with the connection
            mock_migration.assert_called_once_with(mock_conn, database_manager.schema)
            mock_index.assert_called_once_with(mock_conn, database_manager.schema)


@patch("duckdb.connect")
def test_initialize_already_initialized(mock_connect, database_manager):
    """Test initialization when already initialized."""
    database_manager._is_initialized = True
    
    database_manager.initialize()
    
    # Should not connect again
    mock_connect.assert_not_called()


@patch("duckdb.connect")
def test_initialize_failure(mock_connect, database_manager):
    """Test initialization failure."""
    mock_connect.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception, match="Connection failed"):
        database_manager.initialize()
    
    assert database_manager._is_initialized is False
    assert database_manager.conn is None


def test_get_connection_not_initialized(database_manager):
    """Test getting connection when not initialized."""
    with pytest.raises(RuntimeError, match="Database not initialized"):
        database_manager.get_connection()


@patch("duckdb.connect")
def test_get_connection_initialized(mock_connect, database_manager):
    """Test getting connection when initialized."""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    with patch("oboyu.indexer.storage.database_manager.MigrationManager"):
        with patch("oboyu.indexer.storage.database_manager.IndexManager"):
            database_manager.initialize()
    
    conn = database_manager.get_connection()
    assert conn is mock_conn


@patch("duckdb.connect")
def test_close(mock_connect, database_manager):
    """Test closing database connection."""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    with patch("oboyu.indexer.storage.database_manager.MigrationManager"):
        with patch("oboyu.indexer.storage.database_manager.IndexManager"):
            database_manager.initialize()
    
    database_manager.close()
    
    mock_conn.close.assert_called_once()
    assert database_manager.conn is None
    assert database_manager._is_initialized is False


def test_backup_database_file_exists(database_manager, temp_db_path):
    """Test backing up existing database."""
    # Create the database file
    temp_db_path.touch()
    
    backup_path = temp_db_path.parent / "backup.db"
    
    success = database_manager.backup_database(backup_path)
    
    assert success is True
    assert backup_path.exists()
    
    # Cleanup
    backup_path.unlink()


def test_backup_database_file_not_exists(database_manager, temp_db_path):
    """Test backing up non-existent database."""
    # Ensure the database file doesn't exist
    if temp_db_path.exists():
        temp_db_path.unlink()
    
    backup_path = temp_db_path.parent / "backup_nonexistent.db"
    
    success = database_manager.backup_database(backup_path)
    
    assert success is False


@patch("duckdb.connect")
def test_ensure_hnsw_index_no_manager(mock_connect, database_manager):
    """Test ensure_hnsw_index when index manager is not initialized."""
    database_manager.index_manager = None
    
    # Should not raise error
    database_manager.ensure_hnsw_index()


@patch("duckdb.connect")
def test_ensure_hnsw_index_exists(mock_connect, database_manager):
    """Test ensure_hnsw_index when index already exists."""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    with patch("oboyu.indexer.storage.database_manager.MigrationManager"):
        with patch("oboyu.indexer.storage.database_manager.IndexManager") as mock_index_cls:
            mock_index_manager = MagicMock()
            mock_index_manager.hnsw_index_exists.return_value = True
            mock_index_cls.return_value = mock_index_manager
            
            database_manager.initialize()
            database_manager.ensure_hnsw_index()
    
    # Should check but not create
    mock_index_manager.hnsw_index_exists.assert_called_once()
    mock_index_manager.create_hnsw_index.assert_not_called()


@patch("duckdb.connect")
def test_context_manager(mock_connect, database_manager):
    """Test using DatabaseManager as context manager."""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    with patch("oboyu.indexer.storage.database_manager.MigrationManager"):
        with patch("oboyu.indexer.storage.database_manager.IndexManager"):
            with database_manager as db:
                assert db is database_manager
                assert database_manager._is_initialized is True
    
    # Should be closed after context
    mock_conn.close.assert_called_once()
    assert database_manager._is_initialized is False