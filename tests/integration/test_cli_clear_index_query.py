"""Test the actual CLI workflow: clear -> index -> query."""

import subprocess
import tempfile
from pathlib import Path


def test_cli_clear_index_query_workflow():
    """Test the CLI clear-index-query workflow as described in issue #29."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        db_path = tmpdir_path / "test.db"
        docs_dir = tmpdir_path / "docs"
        docs_dir.mkdir()
        
        # Create test documents
        (docs_dir / "test1.txt").write_text("Python is a programming language used for web development and data science.")
        (docs_dir / "test2.txt").write_text("Machine learning involves training models on data to make predictions.")
        
        # Step 1: Initial index
        result = subprocess.run([
            "uv", "run", "python", "-m", "oboyu.cli.main", "index", 
            "--db-path", str(db_path), str(docs_dir)
        ], capture_output=True, text=True)
        print(f"Index command output: {result.stdout}")
        print(f"Index command stderr: {result.stderr}")
        assert result.returncode == 0, f"Index failed: {result.stderr}"
        
        # Step 2: Verify initial query works
        result = subprocess.run([
            "uv", "run", "python", "-m", "oboyu.cli.main", "query",
            "--db-path", str(db_path), "Python programming"
        ], capture_output=True, text=True)
        print(f"Initial query output: {result.stdout}")
        print(f"Initial query stderr: {result.stderr}")
        assert result.returncode == 0, f"Initial query failed: {result.stderr}"
        # Should find at least one result
        assert "Results for:" in result.stdout or "Retrieved" in result.stdout
        
        # Step 3: Clear operation
        result = subprocess.run([
            "uv", "run", "python", "-m", "oboyu.cli.main", "clear", "--force",
            "--db-path", str(db_path)
        ], capture_output=True, text=True)
        print(f"Clear command output: {result.stdout}")
        print(f"Clear command stderr: {result.stderr}")
        assert result.returncode == 0, f"Clear failed: {result.stderr}"
        
        # Step 4: Re-index documents
        result = subprocess.run([
            "uv", "run", "python", "-m", "oboyu.cli.main", "index",
            "--db-path", str(db_path), str(docs_dir)
        ], capture_output=True, text=True)
        print(f"Re-index command output: {result.stdout}")
        print(f"Re-index command stderr: {result.stderr}")
        assert result.returncode == 0, f"Re-index failed: {result.stderr}"
        
        # Step 5: Query after re-index (this is where the bug would manifest)
        result = subprocess.run([
            "uv", "run", "python", "-m", "oboyu.cli.main", "query",
            "--db-path", str(db_path), "Python programming"
        ], capture_output=True, text=True)
        print(f"Final query output: {result.stdout}")
        print(f"Final query stderr: {result.stderr}")
        
        # This is the critical test - the query should work after clear+reindex
        assert result.returncode == 0, f"Query after clear+reindex failed: {result.stderr}"
        # Should find results again
        assert "Results for:" in result.stdout or "Retrieved" in result.stdout


def test_cli_multiple_clear_cycles():
    """Test multiple clear-index cycles via CLI."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        db_path = tmpdir_path / "test.db"
        docs_dir = tmpdir_path / "docs"
        docs_dir.mkdir()
        
        # Create test document
        (docs_dir / "test.txt").write_text("Artificial intelligence and machine learning are growing fields.")
        
        # Perform 3 cycles of clear->index->query
        for cycle in range(3):
            print(f"\n=== Cycle {cycle + 1} ===")
            
            # Clear (except for first cycle)
            if cycle > 0:
                result = subprocess.run([
                    "uv", "run", "python", "-m", "oboyu.cli.main", "clear", "--force",
                    "--db-path", str(db_path)
                ], capture_output=True, text=True)
                assert result.returncode == 0, f"Clear failed in cycle {cycle + 1}: {result.stderr}"
            
            # Index
            result = subprocess.run([
                "uv", "run", "python", "-m", "oboyu.cli.main", "index",
                "--db-path", str(db_path), str(docs_dir)
            ], capture_output=True, text=True)
            assert result.returncode == 0, f"Index failed in cycle {cycle + 1}: {result.stderr}"
            
            # Query
            result = subprocess.run([
                "uv", "run", "python", "-m", "oboyu.cli.main", "query",
                "--db-path", str(db_path), "artificial intelligence"
            ], capture_output=True, text=True)
            
            print(f"Query result for cycle {cycle + 1}: {result.returncode}")
            print(f"Query output: {result.stdout}")
            if result.stderr:
                print(f"Query stderr: {result.stderr}")
            
            assert result.returncode == 0, f"Query failed in cycle {cycle + 1}: {result.stderr}"
            assert "Results for:" in result.stdout or "Retrieved" in result.stdout