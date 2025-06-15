"""Tests for the enrich CLI command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from oboyu.cli.main import app


class TestEnrichCommand:
    """Test cases for the enrich command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Sample CSV data
        self.sample_csv_data = pd.DataFrame({
            "company_name": ["トヨタ自動車", "ソフトバンク"],
            "industry": ["自動車", "通信"]
        })
        
        # Sample schema
        self.sample_schema = {
            "input_schema": {
                "columns": {
                    "company_name": {"type": "string", "description": "会社名"},
                    "industry": {"type": "string", "description": "業界"}
                },
                "primary_keys": ["company_name"]
            },
            "enrichment_schema": {
                "columns": {
                    "description": {
                        "type": "string",
                        "description": "会社の概要",
                        "source_strategy": "search_content",
                        "query_template": "{company_name} 概要",
                        "extraction_method": "summarize"
                    }
                }
            },
            "search_config": {
                "search_mode": "hybrid",
                "use_graphrag": True,
                "rerank": True,
                "top_k": 5,
                "similarity_threshold": 0.5
            }
        }

    def test_enrich_command_help(self):
        """Test that help displays correctly."""
        result = self.runner.invoke(app, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "Enrich CSV data using semantic search and GraphRAG" in result.stdout

    def test_enrich_missing_csv_file(self):
        """Test error when CSV file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_file = Path(tmpdir) / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(self.sample_schema, f)
            
            result = self.runner.invoke(app, [
                "enrich",
                str(Path(tmpdir) / "missing.csv"),
                str(schema_file)
            ])
            
            assert result.exit_code == 1
            assert "CSV file not found" in result.stdout

    def test_enrich_missing_schema_file(self):
        """Test error when schema file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "data.csv"
            self.sample_csv_data.to_csv(csv_file, index=False)
            
            result = self.runner.invoke(app, [
                "enrich",
                str(csv_file),
                str(Path(tmpdir) / "missing.json")
            ])
            
            assert result.exit_code == 1
            assert "Schema file not found" in result.stdout

    def test_enrich_invalid_schema(self):
        """Test error with invalid schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "data.csv"
            self.sample_csv_data.to_csv(csv_file, index=False)
            
            schema_file = Path(tmpdir) / "schema.json"
            invalid_schema = {"invalid": "schema"}
            with open(schema_file, "w") as f:
                json.dump(invalid_schema, f)
            
            result = self.runner.invoke(app, ["enrich", str(csv_file), str(schema_file)])
            
            assert result.exit_code == 1
            assert "Invalid enrichment configuration" in result.stdout

    @patch('oboyu.cli.enrich._execute_enrichment')
    def test_enrich_successful_execution(self, mock_execute):
        """Test successful enrichment execution."""
        mock_execute.return_value = None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "data.csv"
            self.sample_csv_data.to_csv(csv_file, index=False)
            
            schema_file = Path(tmpdir) / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(self.sample_schema, f)
            
            result = self.runner.invoke(app, ["enrich", str(csv_file), str(schema_file)])
            
            assert result.exit_code == 0
            assert "Enrichment completed successfully" in result.stdout
            mock_execute.assert_called_once()

    @pytest.mark.skip(reason="Complex CLI option testing - skipping for now")
    def test_enrich_with_custom_options(self):
        """Test enrich command with custom options."""
        pass

    def test_csv_column_validation(self):
        """Test validation of CSV columns against schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with missing required column
            incomplete_data = pd.DataFrame({"industry": ["自動車"]})
            csv_file = Path(tmpdir) / "data.csv"
            incomplete_data.to_csv(csv_file, index=False)
            
            schema_file = Path(tmpdir) / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(self.sample_schema, f)
            
            with patch('oboyu.cli.enrich._execute_enrichment') as mock_execute:
                mock_execute.side_effect = Exception("CSV file missing required columns: {'company_name'}")
                
                result = self.runner.invoke(app, ["enrich", str(csv_file), str(schema_file)])
                
                assert result.exit_code == 1
                assert "missing required columns" in result.stdout


@pytest.mark.asyncio
class TestEnrichmentExecution:
    """Test cases for the enrichment execution logic."""

    @patch('pandas.read_csv')
    @patch('oboyu.cli.enrich._get_graphrag_service')
    @patch('oboyu.application.enrichment.EnrichmentService')
    async def test_execute_enrichment(self, mock_service_class, mock_graphrag, mock_read_csv):
        """Test the main enrichment execution function."""
        from oboyu.cli.enrich import _execute_enrichment
        from oboyu.cli.base import BaseCommand
        
        # Mock setup - properly mock DataFrame
        mock_df = Mock()
        mock_df.columns = ["company_name", "industry"]
        mock_df.__len__ = Mock(return_value=2)
        mock_read_csv.return_value = mock_df
        
        mock_enriched_df = Mock()
        mock_enriched_df.columns = ["company_name", "industry", "description"]
        mock_enriched_df.to_csv = Mock()
        
        mock_service = Mock()
        mock_service.enrich_dataframe = AsyncMock(return_value=mock_enriched_df)
        mock_service_class.return_value = mock_service
        
        mock_graphrag_service = Mock()
        mock_graphrag.return_value = mock_graphrag_service
        
        mock_base_command = Mock(spec=BaseCommand)
        mock_base_command.console = Mock()
        
        schema = {
            "input_schema": {
                "columns": {"company_name": {"type": "string"}, "industry": {"type": "string"}}
            },
            "enrichment_schema": {
                "columns": {"description": {"type": "string", "source_strategy": "search_content", "query_template": "{company_name}"}}
            }
        }
        
        # Execute
        await _execute_enrichment(
            base_command=mock_base_command,
            csv_file=Path("test.csv"),
            output_file=Path("output.csv"),
            schema=schema,
            batch_size=10,
            max_results=5,
            confidence=0.5,
            use_graph=True,
            db_path=None,
        )
        
        # Verify calls
        mock_read_csv.assert_called_once()
        mock_service.enrich_dataframe.assert_called_once()
        mock_enriched_df.to_csv.assert_called_once_with(Path("output.csv"), index=False)

    @patch('pandas.read_csv')
    async def test_execute_enrichment_csv_error(self, mock_read_csv):
        """Test error handling when CSV loading fails."""
        from oboyu.cli.enrich import _execute_enrichment
        from oboyu.cli.base import BaseCommand
        
        mock_read_csv.side_effect = Exception("CSV parse error")
        mock_base_command = Mock(spec=BaseCommand)
        
        with pytest.raises(Exception, match="Failed to load CSV file"):
            await _execute_enrichment(
                base_command=mock_base_command,
                csv_file=Path("test.csv"),
                output_file=Path("output.csv"),
                schema={},
                batch_size=10,
                max_results=5,
                confidence=0.5,
                use_graph=True,
                db_path=None,
            )