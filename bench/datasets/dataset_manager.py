"""Dataset manager for benchmark evaluation.

This module provides a unified interface for managing evaluation datasets,
including downloading, loading, validation, and preprocessing.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse
import hashlib


logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the evaluation dataset."""
    
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: str = "ja"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "language": self.language,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            title=data.get("title", ""),
            content=data["content"],
            metadata=data.get("metadata", {}),
            language=data.get("language", "ja"),
        )


@dataclass
class Query:
    """Represents a query in the evaluation dataset."""
    
    query_id: str
    query_text: str
    relevant_docs: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: str = "ja"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "relevant_docs": list(self.relevant_docs),
            "metadata": self.metadata,
            "language": self.language,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Query":
        """Create Query from dictionary."""
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            relevant_docs=set(data.get("relevant_docs", [])),
            metadata=data.get("metadata", {}),
            language=data.get("language", "ja"),
        )


@dataclass
class DatasetInfo:
    """Metadata about an evaluation dataset."""
    
    name: str
    description: str
    version: str
    language: str
    num_documents: int
    num_queries: int
    source_url: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "language": self.language,
            "num_documents": self.num_documents,
            "num_queries": self.num_queries,
            "source_url": self.source_url,
            "license": self.license,
            "citation": self.citation,
            "checksum": self.checksum,
        }


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset with documents and queries."""
    
    info: DatasetInfo
    documents: List[Document]
    queries: List[Query]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "info": self.info.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
            "queries": [query.to_dict() for query in self.queries],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        """Create EvaluationDataset from dictionary."""
        info = DatasetInfo(**data["info"])
        documents = [Document.from_dict(doc_data) for doc_data in data["documents"]]
        queries = [Query.from_dict(query_data) for query_data in data["queries"]]
        
        return cls(info=info, documents=documents, queries=queries)
    
    def save(self, filepath: Path) -> None:
        """Save dataset to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = cls.from_dict(data)
        logger.info(f"Dataset loaded from {filepath}")
        return dataset
    
    def validate(self) -> List[str]:
        """Validate dataset and return list of issues found."""
        issues = []
        
        # Check document IDs are unique
        doc_ids = [doc.doc_id for doc in self.documents]
        if len(doc_ids) != len(set(doc_ids)):
            issues.append("Duplicate document IDs found")
        
        # Check query IDs are unique
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            issues.append("Duplicate query IDs found")
        
        # Check relevant docs exist
        doc_id_set = set(doc_ids)
        for query in self.queries:
            missing_docs = query.relevant_docs - doc_id_set
            if missing_docs:
                issues.append(f"Query {query.query_id} references non-existent documents: {missing_docs}")
        
        # Check counts match info
        if len(self.documents) != self.info.num_documents:
            issues.append(f"Document count mismatch: expected {self.info.num_documents}, got {len(self.documents)}")
        
        if len(self.queries) != self.info.num_queries:
            issues.append(f"Query count mismatch: expected {self.info.num_queries}, got {len(self.queries)}")
        
        return issues
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None
    
    def get_query_by_id(self, query_id: str) -> Optional[Query]:
        """Get query by ID."""
        for query in self.queries:
            if query.query_id == query_id:
                return query
        return None
    
    def filter_queries(self, max_queries: Optional[int] = None) -> "EvaluationDataset":
        """Create a filtered version with limited queries."""
        if max_queries is None or max_queries >= len(self.queries):
            return self
        
        filtered_queries = self.queries[:max_queries]
        
        # Update info
        new_info = DatasetInfo(
            name=f"{self.info.name}_filtered",
            description=f"{self.info.description} (filtered to {max_queries} queries)",
            version=self.info.version,
            language=self.info.language,
            num_documents=self.info.num_documents,
            num_queries=len(filtered_queries),
            source_url=self.info.source_url,
            license=self.info.license,
            citation=self.info.citation,
            checksum=None,  # Checksum no longer valid
        )
        
        return EvaluationDataset(
            info=new_info,
            documents=self.documents,
            queries=filtered_queries,
        )


class DatasetManager:
    """Manages evaluation datasets for benchmark testing."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize dataset manager.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or Path("bench/data/datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry of available datasets
        self._dataset_registry = {
            "synthetic": self._create_synthetic_dataset,
            "miracl-ja": self._load_miracl_ja,
            "mldr-ja": self._load_mldr_ja,
            "jagovfaqs-22k": self._load_jagovfaqs,
            "jacwir": self._load_jacwir,
        }
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self._dataset_registry.keys())
    
    def load_dataset(
        self,
        dataset_name: str,
        force_download: bool = False,
        max_queries: Optional[int] = None,
    ) -> EvaluationDataset:
        """Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            force_download: Whether to force re-download if cached
            max_queries: Optional limit on number of queries
            
        Returns:
            Loaded evaluation dataset
        """
        if dataset_name not in self._dataset_registry:
            available = ", ".join(self.list_available_datasets())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Check cache first
        cache_file = self.cache_dir / f"{dataset_name}.json"
        if cache_file.exists() and not force_download:
            try:
                dataset = EvaluationDataset.load(cache_file)
                logger.info(f"Loaded {dataset_name} from cache")
                
                if max_queries:
                    dataset = dataset.filter_queries(max_queries)
                
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}")
        
        # Load/create dataset
        loader_func = self._dataset_registry[dataset_name]
        dataset = loader_func()
        
        # Cache the dataset
        dataset.save(cache_file)
        
        # Apply query limit if specified
        if max_queries:
            dataset = dataset.filter_queries(max_queries)
        
        return dataset
    
    def validate_dataset(self, dataset: EvaluationDataset) -> bool:
        """Validate a dataset and log any issues."""
        issues = dataset.validate()
        if issues:
            logger.error(f"Dataset validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("Dataset validation passed")
        return True
    
    def _create_synthetic_dataset(self) -> EvaluationDataset:
        """Create a synthetic Japanese dataset for testing."""
        # Create sample documents
        documents = []
        for i in range(100):
            doc = Document(
                doc_id=f"doc_{i:03d}",
                title=f"文書 {i}",
                content=f"これは文書{i}の内容です。日本語のテキストサンプルです。検索テストのために作成されました。",
                metadata={"category": f"category_{i % 5}", "length": "short"},
                language="ja",
            )
            documents.append(doc)
        
        # Create sample queries
        queries = []
        for i in range(50):
            # Each query is relevant to 2-3 documents
            relevant_docs = {f"doc_{(i * 2):03d}", f"doc_{(i * 2 + 1):03d}"}
            if i % 3 == 0:  # Every third query has 3 relevant docs
                relevant_docs.add(f"doc_{(i * 2 + 2) % 100:03d}")
            
            query = Query(
                query_id=f"query_{i:03d}",
                query_text=f"検索クエリ {i}",
                relevant_docs=relevant_docs,
                metadata={"difficulty": "easy" if i % 2 == 0 else "medium"},
                language="ja",
            )
            queries.append(query)
        
        # Create dataset info
        info = DatasetInfo(
            name="synthetic",
            description="Synthetic Japanese dataset for testing",
            version="1.0",
            language="ja",
            num_documents=len(documents),
            num_queries=len(queries),
            source_url=None,
            license="MIT",
            citation="Oboyu synthetic dataset",
        )
        
        return EvaluationDataset(info=info, documents=documents, queries=queries)
    
    def _load_miracl_ja(self) -> EvaluationDataset:
        """Load MIRACL-JA dataset (placeholder implementation)."""
        logger.warning("MIRACL-JA dataset loading not implemented - using synthetic data")
        
        # For now, create a synthetic MIRACL-like dataset
        # In a real implementation, this would download from the actual MIRACL dataset
        
        documents = []
        for i in range(1000):
            doc = Document(
                doc_id=f"miracl_doc_{i}",
                title=f"学術論文 {i}",
                content=f"これは学術論文{i}の抄録です。人工知能、機械学習、自然言語処理に関する研究内容が含まれています。",
                metadata={"source": "miracl", "domain": "academic"},
                language="ja",
            )
            documents.append(doc)
        
        queries = []
        for i in range(100):
            relevant_docs = {f"miracl_doc_{i * 10 + j}" for j in range(3)}
            query = Query(
                query_id=f"miracl_query_{i}",
                query_text=f"機械学習に関する研究 {i}",
                relevant_docs=relevant_docs,
                metadata={"source": "miracl", "type": "academic"},
                language="ja",
            )
            queries.append(query)
        
        info = DatasetInfo(
            name="miracl-ja",
            description="MIRACL Japanese dataset (synthetic)",
            version="1.0",
            language="ja",
            num_documents=len(documents),
            num_queries=len(queries),
            source_url="https://github.com/project-miracl/miracl",
            license="Apache-2.0",
            citation="Zhang et al., MIRACL: A Multilingual Retrieval Dataset",
        )
        
        return EvaluationDataset(info=info, documents=documents, queries=queries)
    
    def _load_mldr_ja(self) -> EvaluationDataset:
        """Load MLDR-JA dataset (placeholder implementation)."""
        logger.warning("MLDR-JA dataset loading not implemented - using synthetic data")
        
        documents = []
        for i in range(500):
            doc = Document(
                doc_id=f"mldr_doc_{i}",
                title=f"長文書類 {i}",
                content=f"これは長い文書{i}です。" + "詳細な内容が続きます。" * 20,
                metadata={"source": "mldr", "length": "long"},
                language="ja",
            )
            documents.append(doc)
        
        queries = []
        for i in range(50):
            relevant_docs = {f"mldr_doc_{i * 5 + j}" for j in range(2)}
            query = Query(
                query_id=f"mldr_query_{i}",
                query_text=f"長文書類に関する質問 {i}",
                relevant_docs=relevant_docs,
                metadata={"source": "mldr", "type": "long_document"},
                language="ja",
            )
            queries.append(query)
        
        info = DatasetInfo(
            name="mldr-ja",
            description="MLDR Japanese dataset (synthetic)",
            version="1.0",
            language="ja",
            num_documents=len(documents),
            num_queries=len(queries),
            source_url="https://huggingface.co/datasets/Shitao/MLDR",
            license="MIT",
            citation="Chen et al., MLDR: Multi-Language Dense Retrieval",
        )
        
        return EvaluationDataset(info=info, documents=documents, queries=queries)
    
    def _load_jagovfaqs(self) -> EvaluationDataset:
        """Load JAGovFAQs-22k dataset (placeholder implementation)."""
        logger.warning("JAGovFAQs-22k dataset loading not implemented - using synthetic data")
        
        documents = []
        for i in range(200):
            doc = Document(
                doc_id=f"faq_doc_{i}",
                title=f"よくある質問 {i}",
                content=f"質問: 行政手続き{i}について教えてください。回答: こちらが回答になります。",
                metadata={"source": "jagovfaqs", "type": "faq"},
                language="ja",
            )
            documents.append(doc)
        
        queries = []
        for i in range(30):
            relevant_docs = {f"faq_doc_{i * 3 + j}" for j in range(2)}
            query = Query(
                query_id=f"faq_query_{i}",
                query_text=f"行政手続きについて {i}",
                relevant_docs=relevant_docs,
                metadata={"source": "jagovfaqs", "type": "administrative"},
                language="ja",
            )
            queries.append(query)
        
        info = DatasetInfo(
            name="jagovfaqs-22k",
            description="Japanese Government FAQs dataset (synthetic)",
            version="1.0",
            language="ja",
            num_documents=len(documents),
            num_queries=len(queries),
            source_url="https://github.com/retrieva-jp/jagovfaqs-22k",
            license="CC-BY-4.0",
            citation="Japanese Government FAQs for Retrieval Evaluation",
        )
        
        return EvaluationDataset(info=info, documents=documents, queries=queries)
    
    def _load_jacwir(self) -> EvaluationDataset:
        """Load JACWIR dataset (placeholder implementation)."""
        logger.warning("JACWIR dataset loading not implemented - using synthetic data")
        
        documents = []
        for i in range(300):
            doc = Document(
                doc_id=f"web_doc_{i}",
                title=f"ウェブページ {i}",
                content=f"これはウェブページ{i}の内容です。カジュアルな日本語で書かれています。",
                metadata={"source": "jacwir", "type": "web"},
                language="ja",
            )
            documents.append(doc)
        
        queries = []
        for i in range(40):
            relevant_docs = {f"web_doc_{i * 3 + j}" for j in range(2)}
            query = Query(
                query_id=f"web_query_{i}",
                query_text=f"ウェブ検索クエリ {i}",
                relevant_docs=relevant_docs,
                metadata={"source": "jacwir", "type": "casual"},
                language="ja",
            )
            queries.append(query)
        
        info = DatasetInfo(
            name="jacwir",
            description="Japanese Casual Web Information Retrieval dataset (synthetic)",
            version="1.0",
            language="ja",
            num_documents=len(documents),
            num_queries=len(queries),
            source_url="https://github.com/llm-jp/JACWIR",
            license="Apache-2.0",
            citation="JACWIR: Japanese Casual Web Information Retrieval",
        )
        
        return EvaluationDataset(info=info, documents=documents, queries=queries)


# Convenience functions
def download_dataset(dataset_name: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a dataset and return the path to cached file."""
    manager = DatasetManager(cache_dir)
    dataset = manager.load_dataset(dataset_name, force_download=True)
    return manager.cache_dir / f"{dataset_name}.json"


def load_dataset(
    dataset_name: str,
    cache_dir: Optional[Path] = None,
    max_queries: Optional[int] = None,
) -> EvaluationDataset:
    """Load a dataset by name."""
    manager = DatasetManager(cache_dir)
    return manager.load_dataset(dataset_name, max_queries=max_queries)


def validate_dataset(dataset: EvaluationDataset) -> bool:
    """Validate a dataset."""
    manager = DatasetManager()
    return manager.validate_dataset(dataset)