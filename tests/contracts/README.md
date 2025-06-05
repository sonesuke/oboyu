# Contract-Based Testing for Repository Interfaces

This directory contains contract-based testing infrastructure for the oboyu search repository interfaces, as requested in issue #170.

## Overview

Contract-based testing ensures that different implementations of the same interface behave consistently and satisfy the same contracts. This is especially important for the SearchRepository interface, which may have multiple implementations (in-memory for testing, DuckDB for production, future optimized versions).

## Components

### 1. SearchRepositoryContract (`search_repository_contract.py`)

The core contract class that defines behavioral requirements for all SearchRepository implementations:

- **Storage Contracts**: Basic chunk and embedding storage/retrieval
- **Search Contracts**: Vector similarity and BM25 search functionality  
- **Deletion Contracts**: Single chunk and document-based deletion
- **Statistics Contracts**: Accurate counting of chunks and embeddings
- **Error Handling Contracts**: Consistent behavior for edge cases
- **Performance Contracts**: Concurrency safety and result ordering

### 2. InMemorySearchRepository (`in_memory_search_repository.py`)

A complete in-memory implementation of the SearchRepository interface for fast testing:

- **Fast execution**: No database I/O, all data stored in dictionaries
- **Full BM25 implementation**: Including inverted index and term frequency calculation
- **Vector similarity search**: Using cosine similarity
- **Contract compliance**: Passes all contract tests
- **Test utilities**: Additional methods for debugging and validation

### 3. Test Infrastructure (`test_search_repository_contracts.py`)

Test runners that apply contracts to different implementations:

- **Parametrized tests**: Automatically test multiple implementations
- **InMemorySearchRepository tests**: Fast contract validation
- **DuckDB integration tests**: Marked as slow tests for real database validation
- **Manual validation**: `test_manual_validation.py` for debugging

## Usage

### Running Contract Tests

```bash
# Run fast tests with in-memory implementation
uv run pytest tests/contracts/test_manual_validation.py -v

# Run all contract tests (including slow DuckDB tests)
uv run pytest tests/contracts/ -v

# Run only fast tests (exclude DuckDB integration)
uv run pytest -m "not slow" tests/contracts/ -v
```

### Validating a New Implementation

To validate a new SearchRepository implementation against all contracts:

```python
from tests.contracts.search_repository_contract import run_all_contracts

# Create your implementation
my_repository = MySearchRepository()

# Run all contracts
await run_all_contracts(my_repository)
```

### Adding New Contracts

To add a new behavioral contract:

1. Add a new test method to `SearchRepositoryContract` class
2. Follow the naming convention: `async def test_contract_name`
3. Use appropriate fixtures (`sample_chunk`, `sample_chunks`, etc.)
4. Document the expected behavior clearly

```python
async def test_my_new_contract(self, repository: SearchRepository, sample_chunk: Chunk):
    """Contract: Description of the behavioral requirement."""
    # Test implementation
    pass
```

## Benefits

1. **Behavioral Consistency**: All implementations satisfy the same contracts
2. **Fast Testing**: In-memory implementation enables rapid test execution
3. **Refactoring Safety**: Contracts catch regressions when modifying implementations
4. **Future-Proofing**: New implementations are automatically validated
5. **Better Debugging**: Contract failures pinpoint specific behavioral issues

## Implementation Details

### BM25 Implementation

The in-memory implementation includes a full BM25 search algorithm:

- **Inverted Index**: Maps terms to documents containing them
- **Term Frequency**: Tracks frequency of terms in each document  
- **IDF Calculation**: Handles single-document edge cases properly
- **Score Normalization**: Uses `Score.create()` to handle negative scores

### Vector Search

Uses cosine similarity for vector search:

- **Dimension Validation**: Ensures query and stored vectors have matching dimensions
- **Similarity Threshold**: Respects minimum similarity requirements
- **Top-K Limiting**: Returns only the requested number of results

### Memory Management

The in-memory implementation provides utilities for test cleanup:

- **`clear()`**: Resets all internal state
- **`get_indexed_terms()`**: Returns all indexed terms for debugging
- **`get_term_frequency()`**: Returns term frequency for specific chunks

## Future Enhancements

Potential improvements to the contract testing framework:

1. **Performance Contracts**: Add timing-based contracts for performance validation
2. **Stress Testing**: Large dataset contracts for scalability validation  
3. **Concurrency Contracts**: More sophisticated concurrent access patterns
4. **Data Integrity Contracts**: Ensure data consistency under various conditions
5. **Migration Contracts**: Validate data migration between implementations

## Related Files

- `src/oboyu/ports/repositories/search_repository.py`: The main SearchRepository interface
- `src/oboyu/adapters/database/duckdb_search_repository.py`: Production DuckDB implementation
- `tests/indexer/storage/`: Lower-level repository tests
- `CLAUDE.md`: Project build and testing guidelines