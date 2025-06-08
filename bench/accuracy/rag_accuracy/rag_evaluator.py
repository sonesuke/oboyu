"""RAG System Evaluator for Oboyu.

This module provides the main evaluation framework for testing Oboyu as a complete
RAG (Retrieval-Augmented Generation) system, with focus on Japanese document search.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.logger import BenchmarkLogger
from oboyu.indexer import LegacyIndexer as Indexer
from oboyu.indexer.config import IndexerConfig, ModelConfig, ProcessingConfig, SearchConfig
from oboyu.indexer.storage.database_service import Database

from .metrics_calculator import MetricsCalculator


@dataclass
class RAGEvaluationConfig:
    """Configuration for RAG system evaluation."""

    db_path: str
    indexer_config: IndexerConfig
    top_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    search_modes: List[str] = field(default_factory=lambda: ["vector", "bm25", "hybrid"])
    batch_size: int = 32
    test_size: Optional[int] = None  # None means use all queries
    verbose: bool = False


@dataclass
class QueryResult:
    """Result of a single query in RAG evaluation."""

    query_id: str
    query_text: str
    retrieved_docs: List[Dict[str, Any]]
    relevant_docs: List[str]
    search_mode: str
    response_time: float
    top_k: int


@dataclass
class EvaluationResult:
    """Complete RAG evaluation results."""

    dataset_name: str
    search_mode: str
    top_k: int
    total_queries: int
    metrics: Dict[str, float]
    query_results: List[QueryResult]
    avg_response_time: float
    evaluation_time: float


class RAGEvaluator:
    """Evaluates Oboyu as a complete RAG system."""

    def __init__(self, config: RAGEvaluationConfig, logger: Optional[BenchmarkLogger] = None) -> None:
        """Initialize RAG evaluator.

        Args:
            config: RAG evaluation configuration
            logger: Optional logger for output

        """
        self.config = config
        self.logger = logger or BenchmarkLogger()
        self.metrics_calculator = MetricsCalculator()
        self._ensure_database()

    def _ensure_database(self) -> None:
        """Ensure database exists and is ready."""
        # Ensure the directory exists
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database
        self.db = Database(db_path=self.config.db_path)
        self.db.setup()

    def evaluate_dataset(
        self,
        dataset_name: str,
        queries: List[Dict[str, Any]],
        documents: Optional[List[Dict[str, Any]]] = None,
        reindex: bool = True,
    ) -> List[EvaluationResult]:
        """Evaluate RAG system on a dataset.

        Args:
            dataset_name: Name of the dataset
            queries: List of query dictionaries with 'query_id', 'text', and 'relevant_docs'
            documents: Optional list of documents to index (if reindex=True)
            reindex: Whether to reindex documents before evaluation

        Returns:
            List of evaluation results for each search mode and top_k combination

        """
        self.logger.section(f"Evaluating RAG System on {dataset_name}")

        # Reindex documents if requested
        if reindex and documents:
            self._index_documents(documents)

        # Limit queries if test_size is specified
        if self.config.test_size and len(queries) > self.config.test_size:
            queries = queries[: self.config.test_size]
            self.logger.info(f"Limited to {self.config.test_size} queries for evaluation")

        results = []

        # Evaluate each search mode and top_k combination
        for search_mode in self.config.search_modes:
            for top_k in self.config.top_k_values:
                self.logger.info(f"Evaluating {search_mode} search with top_k={top_k}")
                result = self._evaluate_configuration(dataset_name, queries, search_mode, top_k)
                results.append(result)
                self._report_result(result)

        return results

    def _index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for evaluation.

        Args:
            documents: List of documents with 'doc_id', 'title', 'content', and optional 'metadata'

        """
        self.logger.info(f"Indexing {len(documents)} documents...")
        start_time = time.time()

        # Clear existing data
        self.db.clear()

        # Create temporary files and index them
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="oboyu_rag_eval_"))
        
        # Add timeout check
        max_index_time = 300  # 5 minutes timeout (increased from 60s)

        try:
            # Write documents to temporary files
            file_paths = []
            total_chars = 0
            for i, doc in enumerate(documents):
                doc_path = temp_dir / f"{doc['doc_id']}.txt"
                content = doc.get("title", "") + "\n\n" + doc.get("content", "")
                total_chars += len(content)
                doc_path.write_text(content, encoding="utf-8")
                file_paths.append(str(doc_path))
                
                if i % 10 == 0:
                    self.logger.debug(f"Writing documents: {i+1}/{len(documents)}")
            
            self.logger.info(f"Total characters to index: {total_chars:,}")
            avg_chars = total_chars / len(documents) if documents else 0
            self.logger.info(f"Average document size: {avg_chars:.0f} characters")

            # Index using Oboyu indexer
            # Create new indexer configuration with database path
            model_config = ModelConfig()
            processing_config = ProcessingConfig(
                db_path=Path(self.config.db_path),
                chunk_size=self.config.indexer_config.chunk_size,
                chunk_overlap=self.config.indexer_config.chunk_overlap,
            )
            search_config = SearchConfig()
            
            indexer_config = IndexerConfig(
                model=model_config,
                processing=processing_config,
                search=search_config,
            )
            
            # Create indexer and index files
            indexer = Indexer(config=indexer_config)
            
            # Index with progress callback
            last_progress_time = time.time()
            def progress_callback(stage: str, current: int, total: int) -> None:
                nonlocal last_progress_time
                current_time = time.time()
                
                # Log progress every 5 seconds or every 10% completion
                if current_time - last_progress_time > 5 or (current % max(1, total // 10) == 0):
                    progress_pct = (current / total * 100) if total > 0 else 0
                    elapsed = current_time - start_time
                    self.logger.info(f"{stage} progress: {current}/{total} ({progress_pct:.1f}%) - Elapsed: {elapsed:.1f}s")
                    last_progress_time = current_time
                
                # Check timeout
                if current_time - start_time > max_index_time:
                    raise TimeoutError(f"Indexing exceeded {max_index_time}s timeout")
            
            try:
                indexer.index_directory(str(temp_dir), incremental=False, progress_callback=progress_callback)
            except Exception as e:
                self.logger.error(f"Indexing failed: {type(e).__name__}: {e}")
                raise

            index_time = time.time() - start_time
            self.logger.success(f"Indexed {len(documents)} documents in {index_time:.2f}s")

        except TimeoutError as e:
            self.logger.error(f"Indexing timeout: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Indexing error: {type(e).__name__}: {e}")
            raise
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Close the indexer to release resources
            if 'indexer' in locals():
                indexer.close()

    def _evaluate_configuration(
        self, dataset_name: str, queries: List[Dict[str, Any]], search_mode: str, top_k: int
    ) -> EvaluationResult:
        """Evaluate a specific configuration.

        Args:
            dataset_name: Name of the dataset
            queries: List of queries to evaluate
            search_mode: Search mode to use
            top_k: Number of results to retrieve

        Returns:
            Evaluation result for this configuration

        """
        query_results = []
        response_times = []
        start_time = time.time()

        # Process queries in batches
        for i in range(0, len(queries), self.config.batch_size):
            batch = queries[i : i + self.config.batch_size]
            for query in batch:
                result = self._execute_query(query, search_mode, top_k)
                query_results.append(result)
                response_times.append(result.response_time)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_retrieval_metrics(query_results, top_k)

        evaluation_time = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return EvaluationResult(
            dataset_name=dataset_name,
            search_mode=search_mode,
            top_k=top_k,
            total_queries=len(queries),
            metrics=metrics,
            query_results=query_results,
            avg_response_time=avg_response_time,
            evaluation_time=evaluation_time,
        )

    def _execute_query(self, query: Dict[str, Any], search_mode: str, top_k: int) -> QueryResult:
        """Execute a single query and measure performance.

        Args:
            query: Query dictionary with 'query_id', 'text', and 'relevant_docs'
            search_mode: Search mode to use
            top_k: Number of results to retrieve

        Returns:
            Query result with retrieved documents and timing

        """
        query_start = time.time()

        # Execute query using Oboyu's indexer
        model_config = ModelConfig()
        processing_config = ProcessingConfig(
            db_path=Path(self.config.db_path),
            chunk_size=self.config.indexer_config.chunk_size,
            chunk_overlap=self.config.indexer_config.chunk_overlap,
        )
        search_config = SearchConfig()
        
        indexer_config = IndexerConfig(
            model=model_config,
            processing=processing_config,
            search=search_config,
        )
        
        indexer = Indexer(config=indexer_config)
        
        # Perform search (currently only vector search is supported in indexer)
        # TODO: Add BM25 and hybrid search when available
        try:
            results = indexer.search(query["text"], limit=top_k)
        finally:
            # Close indexer to free resources
            indexer.close()

        response_time = time.time() - query_start

        # Convert results to standard format
        retrieved_docs = []
        for result in results:
            doc = {
                "doc_id": Path(result.path).stem if result.path else "",
                "title": result.title,
                "content": result.content,
                "score": result.score,
                "uri": result.path,
            }
            retrieved_docs.append(doc)

        return QueryResult(
            query_id=query["query_id"],
            query_text=query["text"],
            retrieved_docs=retrieved_docs,
            relevant_docs=query.get("relevant_docs", []),
            search_mode=search_mode,
            response_time=response_time,
            top_k=top_k,
        )

    def _report_result(self, result: EvaluationResult) -> None:
        """Report evaluation result.

        Args:
            result: Evaluation result to report

        """
        self.logger.info(f"\nResults for {result.search_mode} search (top_k={result.top_k}):")
        self.logger.info(f"  Total queries: {result.total_queries}")
        self.logger.info(f"  Avg response time: {result.avg_response_time:.3f}s")
        self.logger.info("  Metrics:")
        for metric, value in result.metrics.items():
            self.logger.info(f"    {metric}: {value:.4f}")

    def save_results(self, results: List[EvaluationResult], output_path: str) -> None:
        """Save evaluation results to file.

        Args:
            results: List of evaluation results
            output_path: Path to save results

        """
        output_data = []
        for result in results:
            # Convert to serializable format
            result_dict = {
                "dataset_name": result.dataset_name,
                "search_mode": result.search_mode,
                "top_k": result.top_k,
                "total_queries": result.total_queries,
                "metrics": result.metrics,
                "avg_response_time": result.avg_response_time,
                "evaluation_time": result.evaluation_time,
                # Optionally include detailed query results
                "query_results": [
                    {
                        "query_id": qr.query_id,
                        "query_text": qr.query_text,
                        "num_retrieved": len(qr.retrieved_docs),
                        "num_relevant": len(qr.relevant_docs),
                        "response_time": qr.response_time,
                    }
                    for qr in result.query_results
                ]
                if self.config.verbose
                else [],
            }
            output_data.append(result_dict)

        Path(output_path).write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        self.logger.success(f"Results saved to {output_path}")

