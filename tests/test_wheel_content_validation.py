"""Test to validate wheel package contents and installation."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.mark.slow
class TestWheelContentValidation:
    """Test suite to validate wheel package contents."""

    def test_build_package_available(self):
        """Test that build package is available before running wheel tests."""
        try:
            import build  # noqa: F401
        except ImportError:
            pytest.fail("build package not available. Install with: pip install build")

    def test_wheel_contains_source_files(self, tmp_path):
        """Test that built wheel contains all necessary source files."""
        # Build wheel in temporary directory
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        # Build the wheel
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(build_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,  # Project root
        )

        if result.returncode != 0:
            print(f"Build stdout: {result.stdout}")
            print(f"Build stderr: {result.stderr}")
            pytest.fail(f"Failed to build wheel: {result.stderr}")

        # Find the built wheel
        wheel_files = list(build_dir.glob("*.whl"))
        assert len(wheel_files) == 1, f"Expected 1 wheel file, found {len(wheel_files)}"
        wheel_file = wheel_files[0]

        # Extract and check wheel contents
        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(wheel_file, "r") as zf:
            zf.extractall(extract_dir)

        # Check for essential source files
        oboyu_dir = extract_dir / "oboyu"
        assert oboyu_dir.exists(), "oboyu package directory not found in wheel"

        # Check for key modules
        expected_modules = [
            "__init__.py",
            "__main__.py",
            "py.typed",
        ]

        for module in expected_modules:
            module_path = oboyu_dir / module
            assert module_path.exists(), f"Missing {module} in wheel"

        # Check for subpackages
        expected_subpackages = [
            "cli",
            "crawler",
            "indexer",
            "retriever",
            "config",
            "common",
        ]

        for subpackage in expected_subpackages:
            subpackage_dir = oboyu_dir / subpackage
            assert subpackage_dir.exists(), f"Missing subpackage {subpackage} in wheel"
            assert (subpackage_dir / "__init__.py").exists(), f"Missing __init__.py in {subpackage}"

    @pytest.mark.skip(reason="Skipped in CI due to disk space constraints")
    def test_wheel_installation_in_isolated_env(self, tmp_path):
        """Test wheel installation in a completely isolated environment."""
        # Skip this test in CI environments where disk space may be limited
        if "CI" in os.environ or "GITHUB_ACTIONS" in os.environ:
            pytest.skip("Skipping isolated env test in CI due to disk space constraints")

        # Build wheel
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(build_dir)], capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

        if result.returncode != 0:
            print(f"Build stdout: {result.stdout}")
            print(f"Build stderr: {result.stderr}")
            pytest.fail(f"Failed to build wheel: {result.stderr}")

        wheel_file = list(build_dir.glob("*.whl"))[0]

        # Create isolated virtual environment
        venv_dir = tmp_path / "test_venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

        # Use the venv's python
        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
            venv_pip = venv_dir / "Scripts" / "pip.exe"
        else:
            venv_python = venv_dir / "bin" / "python"
            venv_pip = venv_dir / "bin" / "pip"

        # Install wheel in isolated environment
        install_result = subprocess.run([str(venv_pip), "install", str(wheel_file)], capture_output=True, text=True)

        assert install_result.returncode == 0, f"Failed to install wheel: {install_result.stderr}"

        # Test import in isolated environment
        import_test = subprocess.run([str(venv_python), "-c", "import oboyu; print(oboyu.__version__)"], capture_output=True, text=True)

        assert import_test.returncode == 0, f"Failed to import oboyu: {import_test.stderr}"
        assert import_test.stdout.strip(), "Version string is empty"

        # Test submodule imports
        submodules_test = subprocess.run(
            [
                str(venv_python),
                "-c",
                """
import oboyu.cli
import oboyu.crawler
import oboyu.indexer
import oboyu.retriever
import oboyu.config
import oboyu.common
print('All submodules imported successfully')
""",
            ],
            capture_output=True,
            text=True,
        )

        assert submodules_test.returncode == 0, f"Failed to import submodules: {submodules_test.stderr}"
        assert "All submodules imported successfully" in submodules_test.stdout

        # Test CLI entry point
        cli_test = subprocess.run([str(venv_python), "-m", "oboyu", "--help"], capture_output=True, text=True)

        assert cli_test.returncode == 0, f"CLI failed: {cli_test.stderr}"
        assert "usage:" in cli_test.stdout.lower() or "Usage:" in cli_test.stdout

    def test_wheel_metadata(self, tmp_path):
        """Test that wheel contains proper metadata."""
        # Build wheel
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(build_dir)], capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

        if result.returncode != 0:
            print(f"Build stdout: {result.stdout}")
            print(f"Build stderr: {result.stderr}")
            pytest.fail(f"Failed to build wheel: {result.stderr}")

        wheel_file = list(build_dir.glob("*.whl"))[0]

        # Extract and check metadata
        with zipfile.ZipFile(wheel_file, "r") as zf:
            # Find metadata directory
            metadata_dirs = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
            assert len(metadata_dirs) > 0, "No metadata found in wheel"

            metadata_content = zf.read(metadata_dirs[0]).decode("utf-8")

            # Check essential metadata
            assert "Name: oboyu" in metadata_content
            assert "Requires-Python:" in metadata_content
            assert "License:" in metadata_content

            # Check dependencies are listed
            assert "duckdb" in metadata_content
            assert "torch" in metadata_content
            assert "transformers" in metadata_content
