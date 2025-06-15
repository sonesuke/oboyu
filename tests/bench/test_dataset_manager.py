"""Tests for benchmark dataset manager."""

import pytest
import tempfile
from pathlib import Path

# Import dataset manager modules
import sys
from pathlib import Path

# Add bench directory to path
bench_path = Path(__file__).parent.parent.parent / "bench"
sys.path.insert(0, str(bench_path))

from bench.datasets.dataset_manager import (
    DatasetManager,
    EvaluationDataset,
    Document,
    Query,
    DatasetInfo,
    load_dataset,
    validate_dataset,
)


class TestDocument:
    """Test Document class."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            doc_id="test_doc_1",
            title="Test Document",
            content="This is test content.",
            metadata={"category": "test"},
            language="ja",
        )
        
        assert doc.doc_id == "test_doc_1"
        assert doc.title == "Test Document"
        assert doc.content == "This is test content."
        assert doc.metadata["category"] == "test"
        assert doc.language == "ja"
    
    def test_document_serialization(self):
        """Test document to/from dict conversion."""
        doc = Document(
            doc_id="test_doc_1",
            title="Test Document",
            content="This is test content.",
        )
        
        # Test to_dict
        doc_dict = doc.to_dict()
        assert doc_dict["doc_id"] == "test_doc_1"
        assert doc_dict["title"] == "Test Document"
        assert doc_dict["content"] == "This is test content."
        
        # Test from_dict
        restored_doc = Document.from_dict(doc_dict)
        assert restored_doc.doc_id == doc.doc_id
        assert restored_doc.title == doc.title
        assert restored_doc.content == doc.content


class TestQuery:
    """Test Query class."""
    
    def test_query_creation(self):
        """Test creating a query."""
        query = Query(
            query_id="test_query_1",
            query_text="Test query",
            relevant_docs={"doc1", "doc2"},
            metadata={"difficulty": "easy"},
            language="ja",
        )
        
        assert query.query_id == "test_query_1"
        assert query.query_text == "Test query"
        assert query.relevant_docs == {"doc1", "doc2"}
        assert query.metadata["difficulty"] == "easy"
        assert query.language == "ja"
    
    def test_query_serialization(self):
        """Test query to/from dict conversion."""
        query = Query(
            query_id="test_query_1",
            query_text="Test query",
            relevant_docs={"doc1", "doc2"},
        )
        
        # Test to_dict
        query_dict = query.to_dict()
        assert query_dict["query_id"] == "test_query_1"
        assert query_dict["query_text"] == "Test query"
        assert set(query_dict["relevant_docs"]) == {"doc1", "doc2"}
        
        # Test from_dict
        restored_query = Query.from_dict(query_dict)
        assert restored_query.query_id == query.query_id
        assert restored_query.query_text == query.query_text
        assert restored_query.relevant_docs == query.relevant_docs


class TestEvaluationDataset:
    """Test EvaluationDataset class."""
    
    def create_test_dataset(self) -> EvaluationDataset:
        """Create a test dataset."""
        info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            version="1.0",
            language="ja",
            num_documents=2,
            num_queries=2,
        )
        
        documents = [
            Document(doc_id="doc1", title="Doc 1", content="Content 1"),
            Document(doc_id="doc2", title="Doc 2", content="Content 2"),
        ]
        
        queries = [
            Query(query_id="query1", query_text="Query 1", relevant_docs={"doc1"}),
            Query(query_id="query2", query_text="Query 2", relevant_docs={"doc2"}),
        ]
        
        return EvaluationDataset(info=info, documents=documents, queries=queries)
    
    def test_dataset_creation(self):
        """Test creating an evaluation dataset."""
        dataset = self.create_test_dataset()
        
        assert dataset.info.name == "test_dataset"
        assert len(dataset.documents) == 2
        assert len(dataset.queries) == 2
    
    def test_dataset_validation(self):
        """Test dataset validation."""
        dataset = self.create_test_dataset()
        
        # Valid dataset should pass
        issues = dataset.validate()
        assert len(issues) == 0
        
        # Test duplicate document IDs
        dataset.documents.append(Document(doc_id="doc1", title="Duplicate", content="Duplicate"))
        issues = dataset.validate()
        assert len(issues) > 0
        assert any("Duplicate document IDs" in issue for issue in issues)
    
    def test_dataset_serialization(self):
        """Test dataset save/load."""
        dataset = self.create_test_dataset()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_dataset.json"
            
            # Save dataset
            dataset.save(filepath)
            assert filepath.exists()
            
            # Load dataset
            loaded_dataset = EvaluationDataset.load(filepath)
            
            # Compare
            assert loaded_dataset.info.name == dataset.info.name
            assert len(loaded_dataset.documents) == len(dataset.documents)
            assert len(loaded_dataset.queries) == len(dataset.queries)
    
    def test_dataset_filtering(self):
        """Test dataset query filtering."""
        dataset = self.create_test_dataset()
        
        # Filter to 1 query
        filtered = dataset.filter_queries(max_queries=1)
        assert len(filtered.queries) == 1
        assert len(filtered.documents) == 2  # Documents should remain the same
        assert filtered.info.num_queries == 1
        assert "filtered" in filtered.info.name
    
    def test_dataset_lookup(self):
        """Test document and query lookup methods."""
        dataset = self.create_test_dataset()
        
        # Test document lookup
        doc = dataset.get_document_by_id("doc1")
        assert doc is not None
        assert doc.doc_id == "doc1"
        
        missing_doc = dataset.get_document_by_id("missing")
        assert missing_doc is None
        
        # Test query lookup
        query = dataset.get_query_by_id("query1")
        assert query is not None
        assert query.query_id == "query1"
        
        missing_query = dataset.get_query_by_id("missing")
        assert missing_query is None


class TestDatasetManager:
    """Test DatasetManager class."""
    
    def test_manager_creation(self):
        """Test creating a dataset manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(cache_dir=Path(temp_dir))
            assert manager.cache_dir.exists()
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        manager = DatasetManager()
        datasets = manager.list_available_datasets()
        
        # Should include synthetic dataset at minimum
        assert "synthetic" in datasets
        assert isinstance(datasets, list)
    
    def test_load_synthetic_dataset(self):
        """Test loading the synthetic dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(cache_dir=Path(temp_dir))
            
            # Load synthetic dataset
            dataset = manager.load_dataset("synthetic")
            
            assert dataset.info.name == "synthetic"
            assert len(dataset.documents) > 0
            assert len(dataset.queries) > 0
            
            # Validate the dataset
            issues = dataset.validate()
            assert len(issues) == 0
    
    def test_load_with_query_limit(self):
        """Test loading dataset with query limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(cache_dir=Path(temp_dir))
            
            # Load with limit
            dataset = manager.load_dataset("synthetic", max_queries=5)
            assert len(dataset.queries) <= 5
    
    def test_caching(self):
        """Test dataset caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(cache_dir=Path(temp_dir))
            
            # First load should create cache
            dataset1 = manager.load_dataset("synthetic")
            cache_file = manager.cache_dir / "synthetic.json"
            assert cache_file.exists()
            
            # Second load should use cache
            dataset2 = manager.load_dataset("synthetic")
            
            # Should be equivalent
            assert dataset1.info.name == dataset2.info.name
            assert len(dataset1.documents) == len(dataset2.documents)
            assert len(dataset1.queries) == len(dataset2.queries)
    
    def test_force_download(self):
        """Test force download functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(cache_dir=Path(temp_dir))
            
            # Load once to create cache
            dataset1 = manager.load_dataset("synthetic")
            cache_file = manager.cache_dir / "synthetic.json"
            original_mtime = cache_file.stat().st_mtime
            
            # Force reload should recreate cache
            dataset2 = manager.load_dataset("synthetic", force_download=True)
            new_mtime = cache_file.stat().st_mtime
            
            # Cache file should be updated
            assert new_mtime >= original_mtime
    
    def test_invalid_dataset(self):
        """Test loading invalid dataset."""
        manager = DatasetManager()
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            manager.load_dataset("invalid_dataset_name")
    
    def test_dataset_validation_integration(self):
        """Test dataset validation through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(cache_dir=Path(temp_dir))
            
            # Load a valid dataset
            dataset = manager.load_dataset("synthetic")
            
            # Validate through manager
            is_valid = manager.validate_dataset(dataset)
            assert is_valid is True


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_dataset_function(self):
        """Test load_dataset convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = load_dataset("synthetic", cache_dir=Path(temp_dir))
            
            assert dataset.info.name == "synthetic"
            assert len(dataset.documents) > 0
            assert len(dataset.queries) > 0
    
    def test_validate_dataset_function(self):
        """Test validate_dataset convenience function."""
        # Create a simple valid dataset
        info = DatasetInfo(
            name="test", description="Test", version="1.0",
            language="ja", num_documents=1, num_queries=1
        )
        documents = [Document(doc_id="doc1", title="Test", content="Test")]
        queries = [Query(query_id="q1", query_text="Test", relevant_docs={"doc1"})]
        dataset = EvaluationDataset(info=info, documents=documents, queries=queries)
        
        # Should be valid
        is_valid = validate_dataset(dataset)
        assert is_valid is True


class TestDatasetTypes:
    """Test different dataset types."""
    
    def test_all_dataset_types(self):
        """Test that all registered dataset types can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(cache_dir=Path(temp_dir))
            
            available_datasets = manager.list_available_datasets()
            
            for dataset_name in available_datasets:
                try:
                    dataset = manager.load_dataset(dataset_name, max_queries=2)
                    
                    # Basic checks
                    assert dataset.info.name == dataset_name
                    assert len(dataset.documents) >= 0
                    assert len(dataset.queries) >= 0
                    
                    # Validation should pass
                    issues = dataset.validate()
                    assert len(issues) == 0, f"Dataset {dataset_name} validation failed: {issues}"
                    
                except Exception as e:
                    pytest.fail(f"Failed to load dataset {dataset_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])