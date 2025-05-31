"""Integration tests for interactive query mode."""
# ruff: noqa: S101, S603

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pexpect
import pytest


@pytest.mark.slow
def test_interactive_query_mode():
    """Test the interactive query mode with real indexer."""
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
        (docs_dir / "python.txt").write_text(
            "Python is a high-level programming language known for its simplicity and readability. "
            "It supports multiple programming paradigms including object-oriented and functional programming."
        )
        (docs_dir / "machine_learning.txt").write_text(
            "Machine learning is a subset of artificial intelligence that enables systems to learn "
            "and improve from experience without being explicitly programmed."
        )
        (docs_dir / "data_science.txt").write_text(
            "Data science combines domain expertise, programming skills, and knowledge of mathematics "
            "and statistics to extract meaningful insights from data."
        )
        
        # Index the documents
        result = subprocess.run([
            sys.executable, "-m", "oboyu.cli", "index",
            "--db-path", str(db_path), str(docs_dir)
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        assert result.returncode == 0, f"Index failed: {result.stderr}"
        
        # Start interactive query session
        cmd = [
            sys.executable, "-m", "oboyu.cli", "query",
            "--db-path", str(db_path),
            "--interactive",
            "--mode", "hybrid",
            "--top-k", "2"
        ]
        
        # Use pexpect to interact with the session
        child = pexpect.spawn(
            " ".join(cmd),
            env=env,
            cwd=str(project_root),
            encoding="utf-8",
            timeout=30
        )
        
        try:
            # Wait for the ready prompt
            child.expect("Ready for search!")
            child.expect(">")
            
            # Test a search query
            child.sendline("Python programming")
            child.expect("Found .* results")
            child.expect("Python is a high-level programming language")
            child.expect(">")
            
            # Test mode change
            child.sendline("mode vector")
            child.expect("Search mode changed to: vector")
            child.expect(">")
            
            # Test another search in vector mode
            child.sendline("machine learning")
            child.expect("Found .* results")
            child.expect("Machine learning")
            child.expect(">")
            
            # Test topk change
            child.sendline("topk 1")
            child.expect("Top-K changed to: 1")
            child.expect(">")
            
            # Test settings command
            child.sendline("settings")
            child.expect("Current Settings:")
            child.expect("Mode: vector")
            child.expect("Top-K: 1")
            child.expect(">")
            
            # Test stats command
            child.sendline("stats")
            child.expect("Index Statistics:")
            child.expect("Total documents:")
            child.expect(">")
            
            # Test help command
            child.sendline("help")
            child.expect("Available Commands:")
            child.expect(">")
            
            # Exit the session
            child.sendline("exit")
            child.expect("Goodbye!")
            child.expect(pexpect.EOF)
            
        except Exception:
            print(f"Child output before failure:\n{child.before}")
            raise
        finally:
            child.close()


@pytest.mark.slow
def test_interactive_query_invalid_commands():
    """Test handling of invalid commands in interactive mode."""
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent.parent
    
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
        
        # Create a minimal document
        (docs_dir / "test.txt").write_text("Test document content")
        
        # Index the document
        result = subprocess.run([
            sys.executable, "-m", "oboyu.cli", "index",
            "--db-path", str(db_path), str(docs_dir)
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        assert result.returncode == 0
        
        # Start interactive session
        cmd = [
            sys.executable, "-m", "oboyu.cli", "query",
            "--db-path", str(db_path),
            "--interactive"
        ]
        
        child = pexpect.spawn(
            " ".join(cmd),
            env=env,
            cwd=str(project_root),
            encoding="utf-8",
            timeout=30
        )
        
        try:
            # Wait for ready
            child.expect("Ready for search!")
            child.expect(">")
            
            # Test invalid mode
            child.sendline("mode invalid")
            child.expect("Invalid mode: invalid")
            child.expect(">")
            
            # Test invalid topk
            child.sendline("topk abc")
            child.expect("Invalid number: abc")
            child.expect(">")
            
            # Test invalid weights
            child.sendline("weights 1.5 0.5")
            child.expect("Weights must be between 0 and 1")
            child.expect(">")
            
            # Test invalid rerank option
            child.sendline("rerank maybe")
            child.expect("Invalid option: maybe")
            child.expect(">")
            
            # Test unknown command
            child.sendline("unknown command")
            child.expect("Unknown command: unknown command")
            child.expect(">")
            
            # Exit
            child.sendline("q")
            child.expect("Goodbye!")
            child.expect(pexpect.EOF)
            
        except Exception:
            print(f"Child output before failure:\n{child.before}")
            raise
        finally:
            child.close()


@pytest.mark.slow  
def test_interactive_query_with_reranker():
    """Test interactive mode with reranker enabled."""
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent.parent
    
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
        
        # Create test document
        (docs_dir / "test.txt").write_text("Python programming and machine learning")
        
        # Index the document
        result = subprocess.run([
            sys.executable, "-m", "oboyu.cli", "index",
            "--db-path", str(db_path), str(docs_dir)
        ], capture_output=True, text=True, env=env, cwd=str(project_root))
        assert result.returncode == 0
        
        # Start interactive session with reranker
        cmd = [
            sys.executable, "-m", "oboyu.cli", "query",
            "--db-path", str(db_path),
            "--interactive",
            "--rerank"
        ]
        
        child = pexpect.spawn(
            " ".join(cmd),
            env=env,
            cwd=str(project_root),
            encoding="utf-8",
            timeout=60  # Longer timeout for reranker initialization
        )
        
        try:
            # Wait for initialization (reranker takes time)
            child.expect("Ready for search!", timeout=45)
            child.expect(">")
            
            # Verify reranker is enabled in initial status
            child.sendline("settings")
            child.expect("Reranker: .*enabled")
            child.expect(">")
            
            # Do a search
            child.sendline("Python")
            child.expect("Found .* results")
            child.expect(">")
            
            # Toggle reranker off
            child.sendline("rerank off")
            child.expect("Reranker .*disabled")
            child.expect(">")
            
            # Exit
            child.sendline("exit")
            child.expect("Goodbye!")
            child.expect(pexpect.EOF)
            
        except Exception:
            print(f"Child output before failure:\n{child.before}")
            raise
        finally:
            child.close()