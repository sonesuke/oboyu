"""Tests for extraction strategies."""

from unittest.mock import AsyncMock, Mock

import pytest

from oboyu.application.enrichment.extraction_strategies import (
    EntityExtractionStrategy,
    GraphRelationsStrategy,
    SearchContentStrategy,
)


class TestSearchContentStrategy:
    """Test cases for SearchContentStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graphrag_service = Mock()
        self.strategy = SearchContentStrategy(
            graphrag_service=self.mock_graphrag_service,
            max_results=5,
            confidence_threshold=0.5,
        )

    @pytest.mark.asyncio
    async def test_extract_value_first_result(self):
        """Test extracting value using first_result method."""
        # Mock search results
        search_results = [
            {
                "relevance_score": 0.8,
                "content": "トヨタ自動車は日本の大手自動車メーカーです。1937年に設立され、世界最大級の自動車メーカーの一つです。"
            }
        ]
        
        self.mock_graphrag_service.semantic_search_with_graph_context = AsyncMock(
            return_value=search_results
        )
        
        col_config = {
            "extraction_method": "first_result"
        }
        
        result = await self.strategy.extract_value("トヨタ自動車 概要", col_config, None)
        
        assert result == "トヨタ自動車は日本の大手自動車メーカーです。1937年に設立され、世界最大級の自動車メーカーの一つです。"

    @pytest.mark.asyncio
    async def test_extract_value_first_sentence(self):
        """Test extracting value using first_sentence method."""
        search_results = [
            {
                "relevance_score": 0.8,
                "content": "トヨタ自動車は日本の大手自動車メーカーです。1937年に設立されました。世界最大級の企業です。"
            }
        ]
        
        self.mock_graphrag_service.semantic_search_with_graph_context = AsyncMock(
            return_value=search_results
        )
        
        col_config = {
            "extraction_method": "first_sentence"
        }
        
        result = await self.strategy.extract_value("トヨタ自動車 概要", col_config, None)
        
        assert result == "トヨタ自動車は日本の大手自動車メーカーです。"

    @pytest.mark.asyncio
    async def test_extract_value_pattern_match(self):
        """Test extracting value using pattern_match method."""
        search_results = [
            {
                "relevance_score": 0.8,
                "content": "トヨタ自動車の従業員数は約366,283人です。"
            }
        ]
        
        self.mock_graphrag_service.semantic_search_with_graph_context = AsyncMock(
            return_value=search_results
        )
        
        col_config = {
            "extraction_method": "pattern_match",
            "extraction_pattern": r"(\d+(?:,\d+)*)人"
        }
        
        result = await self.strategy.extract_value("トヨタ自動車 従業員数", col_config, None)
        
        assert result == "366,283"

    @pytest.mark.asyncio
    async def test_extract_value_summarize(self):
        """Test extracting value using summarize method."""
        search_results = [
            {
                "relevance_score": 0.8,
                "content": "トヨタ自動車は日本の大手自動車メーカーです。"
            },
            {
                "relevance_score": 0.7,
                "content": "1937年に設立され、現在は世界最大級の自動車メーカーの一つです。"
            }
        ]
        
        self.mock_graphrag_service.semantic_search_with_graph_context = AsyncMock(
            return_value=search_results
        )
        
        col_config = {
            "extraction_method": "summarize"
        }
        
        result = await self.strategy.extract_value("トヨタ自動車 概要", col_config, None)
        
        # Should return first sentence of combined content
        assert "トヨタ自動車は日本の大手自動車メーカーです。" in result

    @pytest.mark.asyncio
    async def test_extract_value_no_results(self):
        """Test when no search results are found."""
        self.mock_graphrag_service.semantic_search_with_graph_context = AsyncMock(
            return_value=[]
        )
        
        col_config = {"extraction_method": "first_result"}
        
        result = await self.strategy.extract_value("unknown query", col_config, None)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_value_low_confidence(self):
        """Test filtering results by confidence threshold."""
        search_results = [
            {
                "relevance_score": 0.3,  # Below threshold
                "content": "Low confidence result"
            }
        ]
        
        self.mock_graphrag_service.semantic_search_with_graph_context = AsyncMock(
            return_value=search_results
        )
        
        col_config = {"extraction_method": "first_result"}
        
        result = await self.strategy.extract_value("query", col_config, None)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_value_without_graphrag(self):
        """Test when GraphRAG service is not available."""
        strategy = SearchContentStrategy(graphrag_service=None)
        
        result = await strategy.extract_value("query", {}, None)
        
        assert result is None


class TestEntityExtractionStrategy:
    """Test cases for EntityExtractionStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graphrag_service = Mock()
        self.strategy = EntityExtractionStrategy(
            graphrag_service=self.mock_graphrag_service,
            max_results=5,
            confidence_threshold=0.5,
        )

    @pytest.mark.asyncio
    async def test_extract_value_with_entities(self):
        """Test extracting value when entities are found."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.name = "トヨタ自動車株式会社"
        mock_entity.entity_type = "ORGANIZATION"
        
        expansion_result = {
            "expanded_entities": [
                {
                    "entity": mock_entity,
                    "relevance_score": 0.8
                }
            ]
        }
        
        self.mock_graphrag_service.expand_query_with_entities = AsyncMock(
            return_value=expansion_result
        )
        
        col_config = {}
        
        result = await self.strategy.extract_value("トヨタ", col_config, None)
        
        assert result == "トヨタ自動車株式会社"

    @pytest.mark.asyncio
    async def test_extract_value_with_entity_type_filter(self):
        """Test extracting value with entity type filtering."""
        # Mock entities of different types
        mock_entity1 = Mock()
        mock_entity1.name = "トヨタ自動車"
        mock_entity1.entity_type = "ORGANIZATION"
        
        mock_entity2 = Mock()
        mock_entity2.name = "豊田市"
        mock_entity2.entity_type = "LOCATION"
        
        expansion_result = {
            "expanded_entities": [
                {"entity": mock_entity1, "relevance_score": 0.8},
                {"entity": mock_entity2, "relevance_score": 0.7}
            ]
        }
        
        self.mock_graphrag_service.expand_query_with_entities = AsyncMock(
            return_value=expansion_result
        )
        
        col_config = {
            "entity_types": ["ORGANIZATION"]
        }
        
        result = await self.strategy.extract_value("トヨタ", col_config, None)
        
        assert result == "トヨタ自動車"

    @pytest.mark.asyncio
    async def test_extract_value_with_pattern(self):
        """Test extracting value with pattern matching."""
        mock_entity = Mock()
        mock_entity.name = "従業員数: 366,283人"
        mock_entity.entity_type = "NUMBER"
        
        expansion_result = {
            "expanded_entities": [
                {"entity": mock_entity, "relevance_score": 0.8}
            ]
        }
        
        self.mock_graphrag_service.expand_query_with_entities = AsyncMock(
            return_value=expansion_result
        )
        
        col_config = {
            "extraction_pattern": r"(\d+(?:,\d+)*)人"
        }
        
        result = await self.strategy.extract_value("従業員数", col_config, None)
        
        assert result == "366,283"

    @pytest.mark.asyncio
    async def test_extract_value_no_entities(self):
        """Test when no entities are found."""
        expansion_result = {"expanded_entities": []}
        
        self.mock_graphrag_service.expand_query_with_entities = AsyncMock(
            return_value=expansion_result
        )
        
        result = await self.strategy.extract_value("query", {}, None)
        
        assert result is None


class TestGraphRelationsStrategy:
    """Test cases for GraphRelationsStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graphrag_service = Mock()
        self.mock_kg_repository = Mock()
        self.mock_graphrag_service.kg_repository = self.mock_kg_repository
        
        self.strategy = GraphRelationsStrategy(
            graphrag_service=self.mock_graphrag_service,
            max_results=5,
            confidence_threshold=0.5,
        )

    @pytest.mark.asyncio
    async def test_extract_value_with_relations(self):
        """Test extracting value through graph relations."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = "entity_1"
        mock_entity.name = "トヨタ自動車"
        
        expansion_result = {
            "expanded_entities": [
                {"entity": mock_entity, "relevance_score": 0.8}
            ]
        }
        
        # Mock relation
        mock_relation = Mock()
        mock_relation.relation_type = "FOUNDED_IN"
        mock_relation.target_id = "entity_2"
        
        # Mock target entity
        mock_target_entity = Mock()
        mock_target_entity.name = "1937年"
        mock_target_entity.entity_type = "DATE"
        
        self.mock_graphrag_service.expand_query_with_entities = AsyncMock(
            return_value=expansion_result
        )
        self.mock_kg_repository.get_entity_relations = AsyncMock(
            return_value=[mock_relation]
        )
        self.mock_kg_repository.get_entity_by_id = AsyncMock(
            return_value=mock_target_entity
        )
        
        col_config = {
            "relation_types": ["FOUNDED_IN"],
            "target_entity_types": ["DATE"]
        }
        
        result = await self.strategy.extract_value("トヨタ 設立", col_config, None)
        
        assert result == "1937年"

    @pytest.mark.asyncio
    async def test_extract_value_no_matching_relations(self):
        """Test when no matching relations are found."""
        mock_entity = Mock()
        mock_entity.id = "entity_1"
        
        expansion_result = {
            "expanded_entities": [
                {"entity": mock_entity, "relevance_score": 0.8}
            ]
        }
        
        self.mock_graphrag_service.expand_query_with_entities = AsyncMock(
            return_value=expansion_result
        )
        self.mock_kg_repository.get_entity_relations = AsyncMock(
            return_value=[]
        )
        
        col_config = {
            "relation_types": ["FOUNDED_IN"]
        }
        
        result = await self.strategy.extract_value("query", col_config, None)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_value_no_entities(self):
        """Test when no entities are found for graph relations."""
        expansion_result = {"expanded_entities": []}
        
        self.mock_graphrag_service.expand_query_with_entities = AsyncMock(
            return_value=expansion_result
        )
        
        result = await self.strategy.extract_value("query", {}, None)
        
        assert result is None