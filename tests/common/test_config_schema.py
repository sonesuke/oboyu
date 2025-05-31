"""Tests for configuration schema definitions."""

from pathlib import Path

import pytest

from oboyu.common.config_schema import (
    ConfigSchema,
    CrawlerConfigSchema,
    IndexerConfigSchema,
    QueryConfigSchema,
)


class TestCrawlerConfigSchema:
    """Test CrawlerConfigSchema."""

    def test_default_values(self):
        """Test default values for crawler config."""
        config = CrawlerConfigSchema()
        
        assert config.max_workers == 4
        assert config.timeout == 30
        assert config.max_depth == 3
        assert config.exclude_dirs == ["__pycache__", ".git", "node_modules"]
        assert ".py" in config.include_extensions
        assert config.min_doc_length == 50
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.encoding == "utf-8"
        assert config.use_japanese_tokenizer is True

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "max_workers": 10,
            "timeout": 60,
            "chunk_size": 2000,
            "extra_field": "ignored",
        }
        
        config = CrawlerConfigSchema.from_dict(data)
        
        assert config.max_workers == 10
        assert config.timeout == 60
        assert config.chunk_size == 2000
        # Defaults for unspecified fields
        assert config.max_depth == 3

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = CrawlerConfigSchema(max_workers=8, chunk_size=1500)
        data = config.to_dict()
        
        assert data["max_workers"] == 8
        assert data["chunk_size"] == 1500
        assert data["timeout"] == 30  # Default value
        assert "exclude_dirs" in data
        assert isinstance(data["exclude_dirs"], list)


class TestIndexerConfigSchema:
    """Test IndexerConfigSchema."""

    def test_default_values(self):
        """Test default values for indexer config."""
        config = IndexerConfigSchema()
        
        assert config.embedding_model == "cl-nagoya/ruri-v3-30m"
        assert config.batch_size == 128
        assert config.max_length == 8192
        assert config.normalize_embeddings is True
        assert config.show_progress is True
        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.75
        assert config.use_japanese_tokenizer is True
        assert config.n_probe == 10
        assert config.db_path is None

    def test_from_dict_with_path(self):
        """Test creating from dictionary with path conversion."""
        data = {
            "embedding_model": "custom-model",
            "batch_size": 64,
            "db_path": "/path/to/db",
        }
        
        config = IndexerConfigSchema.from_dict(data)
        
        assert config.embedding_model == "custom-model"
        assert config.batch_size == 64
        assert config.db_path == Path("/path/to/db")

    def test_from_dict_without_path(self):
        """Test creating from dictionary without db_path."""
        data = {
            "embedding_model": "custom-model",
            "batch_size": 64,
        }
        
        config = IndexerConfigSchema.from_dict(data)
        
        assert config.embedding_model == "custom-model"
        assert config.batch_size == 64
        assert config.db_path is None

    def test_to_dict_with_path(self):
        """Test converting to dictionary with path."""
        config = IndexerConfigSchema(
            embedding_model="test-model",
            db_path=Path("/test/path")
        )
        data = config.to_dict()
        
        assert data["embedding_model"] == "test-model"
        assert data["db_path"] == "/test/path"  # Converted to string

    def test_to_dict_without_path(self):
        """Test converting to dictionary without path."""
        config = IndexerConfigSchema(embedding_model="test-model")
        data = config.to_dict()
        
        assert data["embedding_model"] == "test-model"
        assert "db_path" not in data


class TestQueryConfigSchema:
    """Test QueryConfigSchema."""

    def test_default_values(self):
        """Test default values for query config."""
        config = QueryConfigSchema()
        
        assert config.top_k == 10
        assert config.rerank is True
        assert config.rerank_model == "cl-nagoya/ruri-reranker-small"
        assert config.show_scores is False
        assert config.interactive is False

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "top_k": 20,
            "rerank": False,
            "show_scores": True,
        }
        
        config = QueryConfigSchema.from_dict(data)
        
        assert config.top_k == 20
        assert config.rerank is False
        assert config.show_scores is True
        # Defaults for unspecified
        assert config.rerank_model == "cl-nagoya/ruri-reranker-small"

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = QueryConfigSchema(top_k=15, interactive=True)
        data = config.to_dict()
        
        assert data["top_k"] == 15
        assert data["interactive"] is True
        assert data["rerank"] is True  # Default value


class TestConfigSchema:
    """Test complete ConfigSchema."""

    def test_default_values(self):
        """Test default values for complete config."""
        config = ConfigSchema()
        
        assert isinstance(config.crawler, CrawlerConfigSchema)
        assert isinstance(config.indexer, IndexerConfigSchema)
        assert isinstance(config.query, QueryConfigSchema)
        
        # Check some nested values
        assert config.crawler.max_workers == 4
        assert config.indexer.embedding_model == "cl-nagoya/ruri-v3-30m"
        assert config.query.top_k == 10

    def test_from_dict(self):
        """Test creating from nested dictionary."""
        data = {
            "crawler": {
                "max_workers": 8,
                "chunk_size": 1500,
            },
            "indexer": {
                "batch_size": 48,
                "db_path": "/custom/path",
            },
            "query": {
                "top_k": 25,
                "rerank": False,
            },
        }
        
        config = ConfigSchema.from_dict(data)
        
        assert config.crawler.max_workers == 8
        assert config.crawler.chunk_size == 1500
        assert config.indexer.batch_size == 48
        assert config.indexer.db_path == Path("/custom/path")
        assert config.query.top_k == 25
        assert config.query.rerank is False

    def test_from_dict_with_missing_sections(self):
        """Test creating from dictionary with missing sections."""
        data = {
            "query": {
                "top_k": 30,
            }
        }
        
        config = ConfigSchema.from_dict(data)
        
        # Query section is updated
        assert config.query.top_k == 30
        
        # Other sections use defaults
        assert config.crawler.max_workers == 4
        assert config.indexer.embedding_model == "cl-nagoya/ruri-v3-30m"

    def test_to_dict(self):
        """Test converting to nested dictionary."""
        config = ConfigSchema()
        config.crawler.max_workers = 12
        config.indexer.batch_size = 64
        config.query.top_k = 20
        
        data = config.to_dict()
        
        assert data["crawler"]["max_workers"] == 12
        assert data["indexer"]["batch_size"] == 64
        assert data["query"]["top_k"] == 20
        
        # Check structure
        assert isinstance(data["crawler"], dict)
        assert isinstance(data["indexer"], dict)
        assert isinstance(data["query"], dict)