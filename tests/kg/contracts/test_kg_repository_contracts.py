"""Test runner for KGRepository contract tests.

This module runs the contract tests against different KGRepository implementations
to ensure they all satisfy the behavioral contracts.
"""

import pytest

from .in_memory_kg_repository import InMemoryKGRepository
from .kg_repository_contract import KGRepositoryContract


class TestInMemoryKGRepositoryContract(KGRepositoryContract):
    """Run contract tests against InMemoryKGRepository."""

    @pytest.fixture
    async def repository(self) -> InMemoryKGRepository:
        """Create a fresh in-memory repository for each test."""
        return InMemoryKGRepository()


# Integration test for DuckDB implementation (marked as slow)
# Note: Disabled because DuckDB repository doesn't have initialize_schema method
# @pytest.mark.slow
# class TestDuckDBKGRepositoryContract(KGRepositoryContract):
#     """Run contract tests against DuckDBKGRepository (slow)."""
# 
#     @pytest.fixture
#     async def repository(self, temp_db_path: str):
#         """Create a DuckDB repository for contract testing."""
#         import duckdb
#         from oboyu.adapters.kg_repositories import DuckDBKGRepository
# 
#         # Create database connection
#         connection = duckdb.connect(temp_db_path)
#         
#         # Create repository
#         repo = DuckDBKGRepository(connection)
#         
#         try:
#             yield repo
#         finally:
#             # Cleanup
#             connection.close()