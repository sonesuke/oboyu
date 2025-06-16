"""Tests for benchmark metrics modules."""

import pytest
from typing import Dict, List, Set

# Import metrics modules
import sys
from pathlib import Path

# Add bench directory to path
bench_path = Path(__file__).parent.parent.parent / "bench"
sys.path.insert(0, str(bench_path))

from bench.metrics.retrieval_metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_mrr,
    calculate_hit_rate,
    calculate_f1_at_k,
    calculate_all_retrieval_metrics,
)

from bench.metrics.reranking_metrics import (
    calculate_ranking_improvement,
    calculate_map,
    calculate_position_improvement,
    calculate_all_reranking_metrics,
)

from bench.metrics.system_metrics import (
    calculate_end_to_end_accuracy,
    calculate_japanese_effectiveness,
    calculate_all_system_metrics,
)


class TestRetrievalMetrics:
    """Test retrieval metrics calculations."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant_docs = {"doc1", "doc3", "doc5"}
        
        # Test precision@3
        precision = calculate_precision_at_k(retrieved_docs, relevant_docs, 3)
        assert precision == 2/3  # 2 relevant out of 3 retrieved
        
        # Test precision@5
        precision = calculate_precision_at_k(retrieved_docs, relevant_docs, 5)
        assert precision == 3/5  # 3 relevant out of 5 retrieved
        
        # Test edge cases
        assert calculate_precision_at_k([], relevant_docs, 1) == 0.0
        assert calculate_precision_at_k(retrieved_docs, set(), 3) == 0.0
        assert calculate_precision_at_k(retrieved_docs, relevant_docs, 0) == 0.0
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant_docs = {"doc1", "doc3", "doc5"}
        
        # Test recall@3
        recall = calculate_recall_at_k(retrieved_docs, relevant_docs, 3)
        assert recall == 2/3  # 2 out of 3 relevant docs retrieved
        
        # Test recall@5
        recall = calculate_recall_at_k(retrieved_docs, relevant_docs, 5)
        assert recall == 1.0  # All 3 relevant docs retrieved
        
        # Test edge cases
        assert calculate_recall_at_k([], {"doc1"}, 1) == 0.0
        assert calculate_recall_at_k(retrieved_docs, set(), 3) == 0.0
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant_docs = {"doc1", "doc3", "doc5"}
        
        # Test NDCG@3
        ndcg = calculate_ndcg_at_k(retrieved_docs, relevant_docs, 3)
        assert 0.0 <= ndcg <= 1.0
        
        # Test perfect ranking (all relevant docs first)
        perfect_ranking = ["doc1", "doc3", "doc5", "doc2", "doc4"]
        ndcg_perfect = calculate_ndcg_at_k(perfect_ranking, relevant_docs, 3)
        assert ndcg_perfect == 1.0
        
        # Test worst ranking (no relevant docs)
        worst_ranking = ["doc2", "doc4", "doc6"]
        ndcg_worst = calculate_ndcg_at_k(worst_ranking, relevant_docs, 3)
        assert ndcg_worst == 0.0
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank calculation."""
        # Test multiple queries
        retrieved_docs_list = [
            ["doc1", "doc2", "doc3"],  # First relevant at position 1
            ["doc2", "doc1", "doc3"],  # First relevant at position 2
            ["doc2", "doc3", "doc1"],  # First relevant at position 3
        ]
        relevant_docs_list = [
            {"doc1"},
            {"doc1"},
            {"doc1"},
        ]
        
        mrr = calculate_mrr(retrieved_docs_list, relevant_docs_list)
        expected_mrr = (1.0 + 0.5 + 1/3) / 3
        assert abs(mrr - expected_mrr) < 1e-6
        
        # Test edge cases
        assert calculate_mrr([], []) == 0.0
        
        # Test no relevant documents found
        no_relevant = [["doc2", "doc3"], ["doc4", "doc5"]]
        relevant_sets = [{"doc1"}, {"doc1"}]
        assert calculate_mrr(no_relevant, relevant_sets) == 0.0
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        retrieved_docs_list = [
            ["doc1", "doc2"],  # Has relevant doc
            ["doc2", "doc3"],  # No relevant doc
            ["doc1", "doc4"],  # Has relevant doc
        ]
        relevant_docs_list = [
            {"doc1"},
            {"doc1"},
            {"doc1"},
        ]
        
        hit_rate = calculate_hit_rate(retrieved_docs_list, relevant_docs_list)
        assert hit_rate == 2/3  # 2 out of 3 queries have hits
        
        # Test with k parameter
        hit_rate_k1 = calculate_hit_rate(retrieved_docs_list, relevant_docs_list, k=1)
        assert hit_rate_k1 == 2/3  # Both hits are at position 1
    
    def test_f1_at_k(self):
        """Test F1@k calculation."""
        retrieved_docs = ["doc1", "doc2", "doc3"]
        relevant_docs = {"doc1", "doc3", "doc5"}
        
        f1 = calculate_f1_at_k(retrieved_docs, relevant_docs, 3)
        
        # Calculate expected F1
        precision = 2/3  # 2 relevant out of 3 retrieved
        recall = 2/3     # 2 out of 3 relevant retrieved
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert abs(f1 - expected_f1) < 1e-6
    
    def test_all_retrieval_metrics(self):
        """Test comprehensive retrieval metrics calculation."""
        retrieved_docs_list = [
            ["doc1", "doc2", "doc3"],
            ["doc2", "doc1", "doc3"],
        ]
        relevant_docs_list = [
            {"doc1", "doc3"},
            {"doc1"},
        ]
        
        metrics = calculate_all_retrieval_metrics(
            retrieved_docs_list,
            relevant_docs_list,
            k_values=[1, 3],
        )
        
        # Check that all metrics are calculated
        assert isinstance(metrics.precision_at_k, dict)
        assert isinstance(metrics.recall_at_k, dict)
        assert isinstance(metrics.ndcg_at_k, dict)
        assert isinstance(metrics.f1_at_k, dict)
        assert isinstance(metrics.mrr, float)
        assert isinstance(metrics.hit_rate, float)
        
        # Check k values
        assert 1 in metrics.precision_at_k
        assert 3 in metrics.precision_at_k


class TestRerankingMetrics:
    """Test reranking metrics calculations."""
    
    def test_ranking_improvement(self):
        """Test ranking improvement calculation."""
        original = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        reranked = ["doc3", "doc1", "doc2", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}
        
        improvement = calculate_ranking_improvement(original, reranked, relevant, k=5)
        
        # doc1: position 1 -> 2 (worse)
        # doc3: position 3 -> 1 (better)
        # Overall should show some improvement since doc3 moved up more than doc1 moved down
        assert isinstance(improvement, float)
    
    def test_map_calculation(self):
        """Test Mean Average Precision calculation."""
        retrieved_docs_list = [
            ["doc1", "doc2", "doc3"],
            ["doc2", "doc1", "doc3"],
        ]
        relevant_docs_list = [
            {"doc1", "doc3"},
            {"doc1"},
        ]
        
        map_score = calculate_map(retrieved_docs_list, relevant_docs_list)
        assert 0.0 <= map_score <= 1.0
    
    def test_position_improvement(self):
        """Test position improvement calculation."""
        original_rankings = [["doc1", "doc2", "doc3"]]
        reranked_rankings = [["doc3", "doc1", "doc2"]]
        relevant_docs_list = [{"doc1", "doc3"}]
        
        improvement = calculate_position_improvement(
            original_rankings, reranked_rankings, relevant_docs_list
        )
        
        # doc1: position 1 -> 2 (change: -1)
        # doc3: position 3 -> 1 (change: +2)
        # Average: (-1 + 2) / 2 = 0.5
        assert improvement == 0.5


class TestSystemMetrics:
    """Test system-level metrics calculations."""
    
    def test_end_to_end_accuracy(self):
        """Test end-to-end accuracy calculation."""
        queries = ["query1", "query2", "query3"]
        expected = ["answer1", "answer2", "answer3"]
        responses = ["answer1", "different", "answer3"]
        
        accuracy = calculate_end_to_end_accuracy(queries, expected, responses)
        assert accuracy == 2/3  # 2 out of 3 correct answers
    
    def test_japanese_effectiveness(self):
        """Test Japanese effectiveness calculation."""
        japanese_queries = ["日本語クエリ1", "日本語クエリ2"]
        japanese_documents = ["日本語文書1", "日本語文書2"]
        retrieval_results = [["doc1", "doc2"], ["doc1"]]
        relevant_docs = [{"doc1"}, {"doc1"}]
        
        effectiveness = calculate_japanese_effectiveness(
            japanese_queries, japanese_documents, retrieval_results, relevant_docs
        )
        
        assert 0.0 <= effectiveness <= 1.0
    
    def test_all_system_metrics(self):
        """Test comprehensive system metrics calculation."""
        queries = ["query1", "query2"]
        
        metrics = calculate_all_system_metrics(queries)
        
        # Check that metrics object is created
        assert hasattr(metrics, 'end_to_end_accuracy')
        assert hasattr(metrics, 'japanese_effectiveness')
        assert hasattr(metrics, 'multi_hop_capability')
        
        # All metrics should be floats between 0 and 1
        assert 0.0 <= metrics.end_to_end_accuracy <= 1.0
        assert 0.0 <= metrics.japanese_effectiveness <= 1.0
        assert 0.0 <= metrics.multi_hop_capability <= 1.0


class TestMetricsIntegration:
    """Test integration between different metrics modules."""
    
    def test_metrics_consistency(self):
        """Test that metrics are consistent across modules."""
        # Create test data
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant_docs = {"doc1", "doc3", "doc5"}
        
        # Calculate precision and recall
        precision = calculate_precision_at_k(retrieved_docs, relevant_docs, 5)
        recall = calculate_recall_at_k(retrieved_docs, relevant_docs, 5)
        
        # Calculate F1 manually and compare with function
        manual_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        function_f1 = calculate_f1_at_k(retrieved_docs, relevant_docs, 5)
        
        assert abs(manual_f1 - function_f1) < 1e-6
    
    def test_metrics_edge_cases(self):
        """Test edge cases across all metrics."""
        # Empty inputs
        empty_results = calculate_all_retrieval_metrics([], [])
        assert all(v == 0.0 for v in empty_results.precision_at_k.values())
        assert empty_results.mrr == 0.0
        assert empty_results.hit_rate == 0.0
        
        # Single query with no relevant docs
        no_relevant = calculate_all_retrieval_metrics(
            [["doc1", "doc2"]], [set()]
        )
        assert all(v == 0.0 for v in no_relevant.precision_at_k.values())


if __name__ == "__main__":
    pytest.main([__file__])