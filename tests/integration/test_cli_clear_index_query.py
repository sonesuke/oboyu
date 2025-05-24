"""Test the actual CLI workflow: clear -> index -> query."""
# ruff: noqa: S101, S603

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# CLI integration tests work with the package installed in editable mode


def test_cli_clear_index_query_workflow() -> None:
    """Test the CLI clear-index-query workflow as described in issue #29."""
    # Set up environment to ensure the package can be imported
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent.parent
    
    # Add src directory to PYTHONPATH to ensure imports work in subprocess
    src_path = str(project_root / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    
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
            sys.executable, "-m", "oboyu.cli", "index",
            "--db-path", str(db_path), str(docs_dir)
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        print(f"Index command output: {result.stdout}")
        print(f"Index command stderr: {result.stderr}")
        assert result.returncode == 0, f"Index failed: {result.stderr}"
        
        # Step 2: Verify initial query works
        result = subprocess.run([
            sys.executable, "-m", "oboyu.cli", "query",
            "--db-path", str(db_path), "Python programming"
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        print(f"Initial query output: {result.stdout}")
        print(f"Initial query stderr: {result.stderr}")
        assert result.returncode == 0, f"Initial query failed: {result.stderr}"
        # Should find at least one result
        assert "Results for:" in result.stdout or "Retrieved" in result.stdout
        
        # Step 3: Clear operation
        result = subprocess.run([
            sys.executable, "-m", "oboyu.cli", "clear", "--force",
            "--db-path", str(db_path)
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        print(f"Clear command output: {result.stdout}")
        print(f"Clear command stderr: {result.stderr}")
        assert result.returncode == 0, f"Clear failed: {result.stderr}"
        
        # Step 4: Re-index documents
        result = subprocess.run([
            sys.executable, "-m", "oboyu.cli", "index",
            "--db-path", str(db_path), str(docs_dir)
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        print(f"Re-index command output: {result.stdout}")
        print(f"Re-index command stderr: {result.stderr}")
        assert result.returncode == 0, f"Re-index failed: {result.stderr}"
        
        # Step 5: Query after re-index (this is where the bug would manifest)
        result = subprocess.run([
            sys.executable, "-m", "oboyu.cli", "query",
            "--db-path", str(db_path), "Python programming"
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        print(f"Final query output: {result.stdout}")
        print(f"Final query stderr: {result.stderr}")
        
        # This is the critical test - the query should work after clear+reindex
        assert result.returncode == 0, f"Query after clear+reindex failed: {result.stderr}"
        # Should find results again
        assert "Results for:" in result.stdout or "Retrieved" in result.stdout


