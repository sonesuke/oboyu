"""
Installation validation tests to ensure oboyu is properly installed and functional.
These tests should pass in any environment where oboyu is installed.
"""

import subprocess
import sys
import importlib
import pytest
from pathlib import Path


class TestInstallationValidation:
    """Test suite to validate oboyu installation."""

    def test_package_import(self):
        """Test that oboyu package can be imported."""
        try:
            import oboyu
            assert hasattr(oboyu, '__version__')
            print(f"Successfully imported oboyu version: {oboyu.__version__}")
        except ImportError as e:
            pytest.fail(f"Failed to import oboyu: {e}")

    def test_submodules_import(self):
        """Test that all major submodules can be imported."""
        submodules = [
            'oboyu.cli',
            'oboyu.crawler',
            'oboyu.indexer',
            'oboyu.retriever',
            'oboyu.config',
            'oboyu.common',
        ]
        
        for module_name in submodules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
                print(f"✓ Successfully imported {module_name}")
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_cli_entry_point(self):
        """Test that the CLI entry point is available and functional."""
        # Test using subprocess to simulate real CLI usage
        result = subprocess.run(
            [sys.executable, '-m', 'oboyu', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert 'usage:' in result.stdout.lower() or 'Usage:' in result.stdout
        print("✓ CLI entry point works correctly")

    def test_cli_commands_available(self):
        """Test that main CLI commands are available."""
        commands = ['index', 'query', 'manage', 'health']
        
        for command in commands:
            result = subprocess.run(
                [sys.executable, '-m', 'oboyu', command, '--help'],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"Command '{command}' failed: {result.stderr}"
            assert 'usage:' in result.stdout.lower() or 'Usage:' in result.stdout
            print(f"✓ Command '{command}' is available")

    def test_console_script_installation(self):
        """Test that the console script is properly installed."""
        # Try to run oboyu directly (should work if installed properly)
        result = subprocess.run(
            ['oboyu', 'version'],
            capture_output=True,
            text=True,
            shell=True  # Use shell to find the command in PATH
        )
        
        # Check if command was found (returncode 0) or at least recognized
        if result.returncode == 0:
            assert 'oboyu' in result.stdout.lower() or result.stdout.strip()
            print("✓ Console script 'oboyu' is properly installed")
        else:
            # On some systems, the script might not be in PATH yet
            print("⚠ Console script might not be in PATH, checking alternative methods...")
            
            # Try using python -m instead
            result = subprocess.run(
                [sys.executable, '-m', 'oboyu', 'version'],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, "Neither console script nor module execution works"

    def test_dependencies_installed(self):
        """Test that critical dependencies are installed."""
        critical_deps = [
            'duckdb',
            'torch',
            'transformers',
            'click',
            'pydantic',
            'rich',
            'tqdm',
            'huggingface_hub',
        ]
        
        for dep in critical_deps:
            try:
                importlib.import_module(dep.replace('-', '_'))
                print(f"✓ Dependency '{dep}' is installed")
            except ImportError:
                pytest.fail(f"Critical dependency '{dep}' is not installed")

    def test_data_files_accessible(self):
        """Test that package data files are accessible."""
        import oboyu
        
        # Check if package has proper structure
        package_dir = Path(oboyu.__file__).parent
        assert package_dir.exists(), "Package directory not found"
        
        # Check for py.typed marker
        py_typed = package_dir / 'py.typed'
        assert py_typed.exists(), "py.typed marker not found"
        
        print("✓ Package data files are accessible")

    def test_basic_functionality(self):
        """Test basic functionality without requiring external resources."""
        from oboyu.config import ConfigManager
        from oboyu.common.types import SearchMode
        
        # Test configuration manager creation
        config_manager = ConfigManager()
        assert config_manager is not None
        print("✓ ConfigManager object created successfully")
        
        # Test enum access
        assert SearchMode.HYBRID
        assert SearchMode.VECTOR
        assert SearchMode.BM25
        print("✓ Enums are accessible")

    @pytest.mark.parametrize("command", [
        ["version"],
        ["--help"],
        ["health", "--help"],
        ["index", "--help"],
        ["query", "--help"],
    ])
    def test_cli_command_execution(self, command):
        """Test that various CLI commands execute without errors."""
        full_command = [sys.executable, '-m', 'oboyu'] + command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=10  # Prevent hanging
        )
        
        assert result.returncode == 0, f"Command failed: {' '.join(command)}\nError: {result.stderr}"
        print(f"✓ Command '{' '.join(command)}' executed successfully")

    def test_no_import_side_effects(self):
        """Test that importing oboyu doesn't have unwanted side effects."""
        # Create a clean subprocess to test import
        code = """
import sys
import os

# Capture initial state
initial_cwd = os.getcwd()
initial_env_count = len(os.environ)

# Import oboyu
import oboyu

# Check for side effects
assert os.getcwd() == initial_cwd, "Working directory changed"
assert abs(len(os.environ) - initial_env_count) < 5, "Too many environment variables added"

print("No significant side effects detected")
"""
        
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Import test failed: {result.stderr}"
        assert "No significant side effects detected" in result.stdout
        print("✓ Package import has no unwanted side effects")


if __name__ == "__main__":
    # Allow running this file directly for quick validation
    pytest.main([__file__, "-v"])