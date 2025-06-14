#!/usr/bin/env python3
"""Validation script for oboyu installation testing.

This script validates that oboyu installations work correctly across different
installation methods (pip source, pip wheel, pip editable, UV).

Referenced in:
- tests/installation/docker/Dockerfile.test (line 137-142)
- tests/installation/docker/docker-compose.test.yml (line 89)
"""

import os
import subprocess
import sys
from typing import Dict, List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # noqa: S603
        success = result.returncode == 0
        output = result.stdout + result.stderr
        print(f"{'✅' if success else '❌'} {description}")
        if not success:
            print(f"   Command: {' '.join(cmd)}")
            print(f"   Output: {output}")
        return success, output
    except subprocess.TimeoutExpired:
        print(f"❌ {description} (timeout)")
        return False, "Command timed out"
    except Exception as e:
        print(f"❌ {description} (error: {e})")
        return False, str(e)


def validate_python_import(python_path: str, description: str) -> bool:
    """Validate that oboyu can be imported in the given Python environment."""
    cmd = [python_path, "-c", "import oboyu; print('Import successful')"]
    success, _ = run_command(cmd, f"{description} - Python import")
    return success


def validate_cli_help(python_path: str, description: str) -> bool:
    """Validate that oboyu CLI help works in the given Python environment."""
    cmd = [python_path, "-m", "oboyu", "--help"]
    success, _ = run_command(cmd, f"{description} - CLI help")
    return success


def validate_version_info(python_path: str, description: str) -> bool:
    """Validate that oboyu version info can be retrieved."""
    cmd = [python_path, "-c", "import oboyu; print(f'oboyu version: {oboyu.__version__}')"]
    success, output = run_command(cmd, f"{description} - Version info")
    if success and "oboyu version:" in output:
        print(f"   {output.strip()}")
    return success


def validate_basic_functionality(python_path: str, description: str) -> bool:
    """Validate basic oboyu functionality."""
    # Test basic import and initialization
    test_code = """
import oboyu
from oboyu.cli.main import main
from oboyu.indexer import indexer
from oboyu.retriever import retriever
print('Basic functionality test passed')
"""
    cmd = [python_path, "-c", test_code]
    success, _ = run_command(cmd, f"{description} - Basic functionality")
    return success


def validate_installation(venv_path: str, description: str) -> Dict[str, bool]:
    """Validate a specific installation method."""
    print(f"\n🧪 Testing {description}")
    print("=" * 50)

    # Determine Python executable path
    if venv_path.endswith(".venv"):
        # UV virtual environment
        python_path = os.path.join(venv_path, "bin", "python")
    else:
        # Regular virtual environment
        python_path = os.path.join(venv_path, "bin", "python")

    if not os.path.exists(python_path):
        print(f"❌ Python executable not found: {python_path}")
        return {"overall": False}

    results = {}
    results["import"] = validate_python_import(python_path, description)
    results["cli_help"] = validate_cli_help(python_path, description)
    results["version"] = validate_version_info(python_path, description)
    results["functionality"] = validate_basic_functionality(python_path, description)

    results["overall"] = all(results.values())

    if results["overall"]:
        print(f"✅ {description} validation PASSED")
    else:
        print(f"❌ {description} validation FAILED")

    return results


def main() -> None:
    """Validate oboyu installations across different methods."""
    print("🚀 Starting oboyu installation validation")
    print("=" * 60)

    # Define installation environments to test
    installations = [
        ("/home/testuser/venv-source", "pip source installation"),
        ("/home/testuser/venv-wheel", "pip wheel installation"),
        ("/home/testuser/venv-editable", "pip editable installation"),
        ("/home/testuser/.venv", "UV installation"),
    ]

    all_results = {}
    overall_success = True

    for venv_path, description in installations:
        if os.path.exists(venv_path):
            results = validate_installation(venv_path, description)
            all_results[description] = results
            if not results["overall"]:
                overall_success = False
        else:
            print(f"\n⏭️  Skipping {description} - environment not found: {venv_path}")
            all_results[description] = {"overall": False, "skipped": True}
            overall_success = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    for description, results in all_results.items():
        if results.get("skipped"):
            print(f"⏭️  {description}: SKIPPED")
        elif results["overall"]:
            print(f"✅ {description}: PASSED")
        else:
            print(f"❌ {description}: FAILED")

    if overall_success:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        sys.exit(0)
    else:
        print("\n💥 SOME VALIDATIONS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
