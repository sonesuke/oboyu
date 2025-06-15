"""System-level metrics calculation for RAG system evaluation.

This module implements high-level system metrics including end-to-end accuracy,
Japanese language effectiveness, and multi-hop reasoning capability.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union, Optional
import re


@dataclass  
class SystemMetrics:
    """Container for system-level evaluation metrics."""
    
    end_to_end_accuracy: float
    japanese_effectiveness: float
    multi_hop_capability: float
    semantic_coherence: float
    response_quality: float
    language_consistency: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "end_to_end_accuracy": self.end_to_end_accuracy,
            "japanese_effectiveness": self.japanese_effectiveness,
            "multi_hop_capability": self.multi_hop_capability,
            "semantic_coherence": self.semantic_coherence,
            "response_quality": self.response_quality,
            "language_consistency": self.language_consistency,
        }


def calculate_end_to_end_accuracy(
    queries: List[str],
    expected_answers: List[str],
    system_responses: List[str],
    similarity_threshold: float = 0.7,
) -> float:
    """Calculate end-to-end accuracy of the complete RAG pipeline.
    
    Measures how often the system provides correct answers to queries.
    
    Args:
        queries: List of input queries
        expected_answers: List of expected correct answers
        system_responses: List of actual system responses
        similarity_threshold: Threshold for considering answers correct
        
    Returns:
        End-to-end accuracy score (0.0 to 1.0)
    """
    if len(queries) != len(expected_answers) != len(system_responses):
        raise ValueError("All input lists must have the same length")
    
    if len(queries) == 0:
        return 0.0
    
    correct_answers = 0
    
    for expected, response in zip(expected_answers, system_responses):
        if _is_answer_correct(expected, response, similarity_threshold):
            correct_answers += 1
    
    return correct_answers / len(queries)


def calculate_japanese_effectiveness(
    japanese_queries: List[str],
    japanese_documents: List[str],
    retrieval_results: List[List[str]],
    relevant_docs: List[Set[str]],
    japanese_responses: List[str] = None,
) -> float:
    """Calculate effectiveness for Japanese language processing.
    
    Evaluates how well the system handles Japanese text processing including
    morphological analysis, encoding detection, and retrieval quality.
    
    Args:
        japanese_queries: List of Japanese queries
        japanese_documents: List of Japanese documents in the corpus
        retrieval_results: List of retrieved document IDs for each query
        relevant_docs: List of relevant document sets for each query
        japanese_responses: Optional list of Japanese responses for quality evaluation
        
    Returns:
        Japanese effectiveness score (0.0 to 1.0)
    """
    if len(japanese_queries) != len(retrieval_results) != len(relevant_docs):
        raise ValueError("Query and result lists must have the same length")
    
    if len(japanese_queries) == 0:
        return 0.0
    
    scores = []
    
    # 1. Character encoding detection effectiveness
    encoding_score = _evaluate_encoding_handling(japanese_documents)
    
    # 2. Morphological analysis effectiveness
    morphology_score = _evaluate_morphological_analysis(japanese_queries, retrieval_results)
    
    # 3. Japanese-specific retrieval quality  
    retrieval_score = _evaluate_japanese_retrieval_quality(
        japanese_queries, retrieval_results, relevant_docs
    )
    
    # 4. Response quality in Japanese (if available)
    response_score = 1.0  # Default if no responses provided
    if japanese_responses:
        response_score = _evaluate_japanese_response_quality(japanese_queries, japanese_responses)
    
    # Weighted combination of scores
    japanese_effectiveness = (
        0.2 * encoding_score +
        0.3 * morphology_score +
        0.3 * retrieval_score +
        0.2 * response_score
    )
    
    return japanese_effectiveness


def calculate_multi_hop_capability(
    multi_hop_queries: List[str],
    document_graph: Dict[str, List[str]],
    retrieval_results: List[List[str]],
    expected_hops: List[int],
    actual_hops: List[int] = None,
) -> float:
    """Calculate multi-hop reasoning capability.
    
    Evaluates the system's ability to find information that requires
    reasoning across multiple documents.
    
    Args:
        multi_hop_queries: List of queries requiring multi-hop reasoning
        document_graph: Graph of document relationships (doc_id -> connected_docs)
        retrieval_results: List of retrieved document IDs for each query
        expected_hops: List of expected number of hops for each query
        actual_hops: Optional list of actual hops detected in results
        
    Returns:
        Multi-hop capability score (0.0 to 1.0)
    """
    if len(multi_hop_queries) != len(retrieval_results) != len(expected_hops):
        raise ValueError("All input lists must have the same length")
    
    if len(multi_hop_queries) == 0:
        return 0.0
    
    scores = []
    
    for i, (query, results, expected) in enumerate(
        zip(multi_hop_queries, retrieval_results, expected_hops)
    ):
        # Calculate how well the system covered the expected hop pattern
        if actual_hops and i < len(actual_hops):
            hop_score = min(actual_hops[i] / expected, 1.0) if expected > 0 else 1.0
        else:
            # Estimate hops from document connectivity
            hop_score = _estimate_hop_coverage(results, document_graph, expected)
        
        scores.append(hop_score)
    
    return sum(scores) / len(scores)


def _is_answer_correct(
    expected: str,
    response: str,
    threshold: float,
) -> bool:
    """Check if a response is correct based on expected answer."""
    # Simple keyword overlap for now - could be enhanced with semantic similarity
    expected_words = set(expected.lower().split())
    response_words = set(response.lower().split())
    
    if not expected_words:
        return len(response_words) == 0
    
    overlap = len(expected_words.intersection(response_words))
    similarity = overlap / len(expected_words)
    
    return similarity >= threshold


def _evaluate_encoding_handling(documents: List[str]) -> float:
    """Evaluate how well the system handles different encodings."""
    if not documents:
        return 1.0
    
    # Check for encoding issues (mojibake patterns)
    encoding_issues = 0
    
    for doc in documents:
        # Look for common mojibake patterns
        if re.search(r'[ï¿½ï¿ï¿]', doc):  # UTF-8 mojibake
            encoding_issues += 1
        elif re.search(r'[ÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', doc):
            # Latin-1/CP1252 mojibake
            encoding_issues += 1
    
    return max(0.0, 1.0 - (encoding_issues / len(documents)))


def _evaluate_morphological_analysis(
    queries: List[str],
    results: List[List[str]],
) -> float:
    """Evaluate morphological analysis effectiveness for Japanese."""
    if not queries:
        return 1.0
    
    # Simple heuristic: check if results seem appropriate for Japanese queries
    # In a real implementation, this would use actual morphological analysis
    
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
    
    scores = []
    for query, result_list in zip(queries, results):
        if japanese_pattern.search(query):
            # For Japanese queries, expect some results (indicator of good processing)
            score = min(len(result_list) / 10.0, 1.0)  # Normalize to 0-1
        else:
            score = 1.0  # Non-Japanese queries score full points
        scores.append(score)
    
    return sum(scores) / len(scores)


def _evaluate_japanese_retrieval_quality(
    queries: List[str],
    results: List[List[str]], 
    relevant_docs: List[Set[str]],
) -> float:
    """Evaluate retrieval quality specifically for Japanese content."""
    if not queries:
        return 1.0
    
    from .retrieval_metrics import calculate_precision_at_k
    
    # Calculate precision@5 as a simple quality metric
    precisions = []
    for result_list, relevant in zip(results, relevant_docs):
        precision = calculate_precision_at_k(result_list, relevant, 5)
        precisions.append(precision)
    
    return sum(precisions) / len(precisions) if precisions else 0.0


def _evaluate_japanese_response_quality(
    queries: List[str],
    responses: List[str],
) -> float:
    """Evaluate quality of Japanese responses."""
    if not queries or not responses:
        return 1.0
    
    # Simple heuristics for Japanese response quality
    scores = []
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
    
    for query, response in zip(queries, responses):
        score = 1.0
        
        # Check if Japanese query gets Japanese response
        if japanese_pattern.search(query):
            if japanese_pattern.search(response):
                score = 1.0  # Good: Japanese response for Japanese query
            else:
                score = 0.5  # Partial: Non-Japanese response for Japanese query
        
        # Check response length (too short might indicate poor quality)
        if len(response.strip()) < 10:
            score *= 0.5
        
        scores.append(score)
    
    return sum(scores) / len(scores)


def _estimate_hop_coverage(
    results: List[str],
    document_graph: Dict[str, List[str]],
    expected_hops: int,
) -> float:
    """Estimate how many hops are covered in the retrieval results."""
    if not results or expected_hops <= 0:
        return 1.0
    
    # Start with first document and see how many hops we can make
    covered_docs = set(results)
    hops_found = 0
    
    current_docs = set(results)
    for hop in range(expected_hops):
        next_docs = set()
        for doc_id in current_docs:
            if doc_id in document_graph:
                connected = document_graph[doc_id]
                for connected_doc in connected:
                    if connected_doc in covered_docs:
                        next_docs.add(connected_doc)
        
        if next_docs:
            hops_found += 1
            current_docs = next_docs
        else:
            break
    
    return min(hops_found / expected_hops, 1.0)


def calculate_semantic_coherence(
    queries: List[str],
    retrieved_docs: List[List[str]],
    document_contents: Dict[str, str],
) -> float:
    """Calculate semantic coherence of retrieved documents.
    
    Measures how well retrieved documents relate to each other semantically.
    
    Args:
        queries: List of input queries
        retrieved_docs: List of retrieved document IDs for each query
        document_contents: Mapping from document ID to content
        
    Returns:
        Semantic coherence score (0.0 to 1.0)
    """
    if not queries or not retrieved_docs:
        return 1.0
    
    coherence_scores = []
    
    for query, docs in zip(queries, retrieved_docs):
        if len(docs) < 2:
            coherence_scores.append(1.0)  # Single doc is perfectly coherent
            continue
        
        # Simple keyword overlap between documents as coherence measure
        doc_word_sets = []
        for doc_id in docs:
            if doc_id in document_contents:
                content = document_contents[doc_id].lower()
                words = set(content.split())
                doc_word_sets.append(words)
        
        if len(doc_word_sets) < 2:
            coherence_scores.append(0.5)
            continue
        
        # Calculate pairwise similarity and average
        similarities = []
        for i in range(len(doc_word_sets)):
            for j in range(i + 1, len(doc_word_sets)):
                set1, set2 = doc_word_sets[i], doc_word_sets[j]
                if set1 and set2:
                    overlap = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = overlap / union if union > 0 else 0.0
                    similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        coherence_scores.append(avg_similarity)
    
    return sum(coherence_scores) / len(coherence_scores)


def calculate_response_quality(
    queries: List[str],
    responses: List[str],
    reference_answers: List[str] = None,
) -> float:
    """Calculate overall response quality.
    
    Args:
        queries: List of input queries
        responses: List of system responses
        reference_answers: Optional reference answers for comparison
        
    Returns:
        Response quality score (0.0 to 1.0)
    """
    if not queries or not responses:
        return 0.0
    
    if len(queries) != len(responses):
        raise ValueError("Queries and responses must have the same length")
    
    quality_scores = []
    
    for i, (query, response) in enumerate(zip(queries, responses)):
        score = 0.0
        
        # Basic quality checks
        if len(response.strip()) > 0:
            score += 0.3  # Non-empty response
        
        if len(response.strip()) >= 20:
            score += 0.2  # Reasonable length
        
        # Query relevance (simple keyword overlap)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        if query_words and response_words:
            overlap = len(query_words.intersection(response_words))
            relevance = min(overlap / len(query_words), 1.0)
            score += 0.3 * relevance
        
        # Reference comparison if available
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
            if _is_answer_correct(reference, response, threshold=0.5):
                score += 0.2
        else:
            score += 0.2  # Give benefit of doubt if no reference
        
        quality_scores.append(min(score, 1.0))
    
    return sum(quality_scores) / len(quality_scores)


def calculate_language_consistency(
    queries: List[str],
    responses: List[str],
) -> float:
    """Calculate language consistency between queries and responses.
    
    Args:
        queries: List of input queries
        responses: List of system responses
        
    Returns:
        Language consistency score (0.0 to 1.0)
    """
    if not queries or not responses:
        return 1.0
    
    if len(queries) != len(responses):
        raise ValueError("Queries and responses must have the same length")
    
    consistency_scores = []
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
    
    for query, response in zip(queries, responses):
        query_is_japanese = bool(japanese_pattern.search(query))
        response_is_japanese = bool(japanese_pattern.search(response))
        
        # Score based on language matching
        if query_is_japanese == response_is_japanese:
            consistency_scores.append(1.0)  # Perfect match
        elif query_is_japanese and not response_is_japanese:
            consistency_scores.append(0.3)  # Japanese query, non-Japanese response
        else:
            consistency_scores.append(0.7)  # Non-Japanese query, Japanese response (less problematic)
    
    return sum(consistency_scores) / len(consistency_scores)


def calculate_all_system_metrics(
    queries: List[str],
    expected_answers: List[str] = None,
    system_responses: List[str] = None,
    japanese_documents: List[str] = None,
    retrieval_results: List[List[str]] = None,
    relevant_docs: List[Set[str]] = None,
    multi_hop_queries: List[str] = None,
    document_graph: Dict[str, List[str]] = None,
    expected_hops: List[int] = None,
    document_contents: Dict[str, str] = None,
) -> SystemMetrics:
    """Calculate all system-level metrics.
    
    Args:
        queries: List of input queries
        expected_answers: Expected correct answers
        system_responses: Actual system responses
        japanese_documents: Japanese documents in corpus
        retrieval_results: Retrieved document IDs for each query
        relevant_docs: Relevant document sets for each query
        multi_hop_queries: Queries requiring multi-hop reasoning
        document_graph: Document relationship graph
        expected_hops: Expected number of hops for multi-hop queries
        document_contents: Document ID to content mapping
        
    Returns:
        SystemMetrics object with all calculated metrics
    """
    # Calculate individual metrics with available data
    end_to_end_accuracy = 0.0
    if expected_answers and system_responses:
        end_to_end_accuracy = calculate_end_to_end_accuracy(
            queries, expected_answers, system_responses
        )
    
    japanese_effectiveness = 0.0
    if japanese_documents and retrieval_results and relevant_docs:
        japanese_effectiveness = calculate_japanese_effectiveness(
            queries, japanese_documents, retrieval_results, relevant_docs, system_responses
        )
    
    multi_hop_capability = 0.0
    if multi_hop_queries and document_graph and expected_hops and retrieval_results:
        multi_hop_capability = calculate_multi_hop_capability(
            multi_hop_queries, document_graph, retrieval_results, expected_hops
        )
    
    semantic_coherence = 0.0
    if retrieval_results and document_contents:
        semantic_coherence = calculate_semantic_coherence(
            queries, retrieval_results, document_contents
        )
    
    response_quality = 0.0
    if system_responses:
        response_quality = calculate_response_quality(
            queries, system_responses, expected_answers
        )
    
    language_consistency = 0.0
    if system_responses:
        language_consistency = calculate_language_consistency(queries, system_responses)
    
    return SystemMetrics(
        end_to_end_accuracy=end_to_end_accuracy,
        japanese_effectiveness=japanese_effectiveness,
        multi_hop_capability=multi_hop_capability,
        semantic_coherence=semantic_coherence,
        response_quality=response_quality,
        language_consistency=language_consistency,
    )