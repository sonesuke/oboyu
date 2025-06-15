"""Tests for the EnrichmentService class."""

from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from oboyu.application.enrichment import EnrichmentService


class TestEnrichmentService:
    """Test cases for EnrichmentService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graphrag_service = Mock()
        self.mock_console = Mock()
        
        self.service = EnrichmentService(
            graphrag_service=self.mock_graphrag_service,
            console=self.mock_console,
            max_results=5,
            confidence_threshold=0.5,
        )
        
        self.sample_df = pd.DataFrame({
            "company_name": ["トヨタ自動車", "ソフトバンク"],
            "industry": ["自動車", "通信"]
        })
        
        self.sample_schema = {
            "input_schema": {
                "columns": {
                    "company_name": {"type": "string"},
                    "industry": {"type": "string"}
                }
            },
            "enrichment_schema": {
                "columns": {
                    "description": {
                        "type": "string",
                        "source_strategy": "search_content",
                        "query_template": "{company_name} 概要",
                        "extraction_method": "summarize"
                    },
                    "employees": {
                        "type": "integer",
                        "source_strategy": "search_content",
                        "query_template": "{company_name} 従業員",
                        "extraction_method": "pattern_match",
                        "extraction_pattern": r"(\d+)人"
                    }
                }
            }
        }

    def test_initialization(self):
        """Test service initialization."""
        assert self.service.graphrag_service == self.mock_graphrag_service
        assert self.service.console == self.mock_console
        assert self.service.max_results == 5
        assert self.service.confidence_threshold == 0.5
        assert len(self.service.strategies) == 3

    @pytest.mark.asyncio
    async def test_enrich_dataframe(self):
        """Test enriching a dataframe."""
        # Mock strategy behavior and progress bar
        with patch.object(self.service, '_process_batch', new_callable=AsyncMock) as mock_process, \
             patch('oboyu.application.enrichment.enrichment_service.Progress') as mock_progress:
            
            # Mock the progress context manager
            mock_progress_instance = Mock()
            mock_progress_instance.__enter__ = Mock(return_value=mock_progress_instance)
            mock_progress_instance.__exit__ = Mock(return_value=None)
            mock_progress_instance.add_task = Mock(return_value="task_id")
            mock_progress_instance.update = Mock()
            mock_progress.return_value = mock_progress_instance
            
            result = await self.service.enrich_dataframe(
                df=self.sample_df,
                schema=self.sample_schema,
                batch_size=2
            )
            
            # Check that new columns were added
            assert "description" in result.columns
            assert "employees" in result.columns
            
            # Check that process_batch was called for each batch
            assert mock_process.call_count == 1  # 2 rows with batch_size=2

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test processing a batch of rows."""
        batch_df = self.sample_df.copy()
        
        # Mock column enrichment
        with patch.object(self.service, '_enrich_column', new_callable=AsyncMock) as mock_enrich:
            await self.service._process_batch(batch_df, self.sample_schema, 0)
            
            # Should be called once for each enrichment column
            assert mock_enrich.call_count == 2

    @pytest.mark.asyncio
    async def test_enrich_column_search_content(self):
        """Test enriching a column using search_content strategy."""
        batch_df = self.sample_df.copy()
        col_config = self.sample_schema["enrichment_schema"]["columns"]["description"]
        
        # Mock strategy
        mock_strategy = AsyncMock()
        mock_strategy.extract_value.return_value = "自動車メーカー"
        self.service.strategies["search_content"] = mock_strategy
        
        await self.service._enrich_column(batch_df, "description", col_config, 0)
        
        # Check that strategy was called for each row
        assert mock_strategy.extract_value.call_count == 2

    def test_format_query_template(self):
        """Test query template formatting."""
        row = pd.Series({"company_name": "トヨタ自動車", "industry": "自動車"})
        template = "{company_name} 概要 {industry}"
        
        result = self.service._format_query_template(template, row)
        
        assert result == "トヨタ自動車 概要 自動車"

    def test_format_query_template_missing_column(self):
        """Test query template formatting with missing column."""
        row = pd.Series({"company_name": "トヨタ自動車"})
        template = "{company_name} {missing_column}"
        
        result = self.service._format_query_template(template, row)
        
        # Missing columns should be replaced with empty string
        assert result == "トヨタ自動車 {missing_column}"

    @pytest.mark.asyncio
    async def test_get_enrichment_summary(self):
        """Test getting enrichment summary."""
        # Create enriched dataframe
        enriched_df = self.sample_df.copy()
        enriched_df["description"] = ["自動車メーカー", None]
        enriched_df["employees"] = [100000, 50000]
        
        summary = await self.service.get_enrichment_summary(enriched_df, self.sample_schema)
        
        assert summary["total_rows"] == 2
        assert summary["enriched_columns"]["description"]["filled_rows"] == 1
        assert summary["enriched_columns"]["description"]["completion_rate"] == 0.5
        assert summary["enriched_columns"]["employees"]["filled_rows"] == 2
        assert summary["enriched_columns"]["employees"]["completion_rate"] == 1.0
        assert summary["overall_completion"] == 0.75  # (1 + 2) / (2 * 2)

    @pytest.mark.asyncio
    async def test_enrich_column_with_unknown_strategy(self):
        """Test enriching column with unknown strategy."""
        batch_df = self.sample_df.copy()
        # Add the test column to the dataframe first
        batch_df["test_col"] = None
        
        col_config = {
            "source_strategy": "unknown_strategy",
            "query_template": "{company_name}"
        }
        
        # Should not raise exception, just log warning
        await self.service._enrich_column(batch_df, "test_col", col_config, 0)
        
        # Column should remain None
        assert batch_df["test_col"].isna().all()

    @pytest.mark.asyncio
    async def test_enrich_column_with_exception(self):
        """Test enriching column when strategy raises exception."""
        batch_df = self.sample_df.copy()
        col_config = self.sample_schema["enrichment_schema"]["columns"]["description"]
        
        # Mock strategy to raise exception
        mock_strategy = AsyncMock()
        mock_strategy.extract_value.side_effect = Exception("Test error")
        self.service.strategies["search_content"] = mock_strategy
        
        # Should not raise exception, just set to None
        await self.service._enrich_column(batch_df, "description", col_config, 0)
        
        # All values should be None due to exception
        assert batch_df["description"].isna().all()


class TestEnrichmentServiceWithoutGraphRAG:
    """Test cases for EnrichmentService without GraphRAG."""

    def test_initialization_without_graphrag(self):
        """Test service initialization without GraphRAG service."""
        service = EnrichmentService(
            graphrag_service=None,
            console=None,
            max_results=3,
            confidence_threshold=0.7,
        )
        
        assert service.graphrag_service is None
        assert service.max_results == 3
        assert service.confidence_threshold == 0.7
        # Console should have default
        assert service.console is not None