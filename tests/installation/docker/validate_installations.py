#!/usr/bin/env python3
"""Validate all oboyu installations across different methods."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], env_name: str) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✅ {env_name}: {' '.join(cmd)} - Success")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {env_name}: {' '.join(cmd)} - Failed")
        print(f"   Error: {e.stderr}")
        return False


def validate_installation(venv_path: Path, env_name: str) -> bool:
    """Validate a single installation."""
    print(f"\n🔍 Validating {env_name} installation...")
    
    python_path = venv_path / "bin" / "python"
    oboyu_path = venv_path / "bin" / "oboyu"
    
    # Check if paths exist
    if not python_path.exists():
        print(f"❌ {env_name}: Python not found at {python_path}")
        return False
    
    if not oboyu_path.exists():
        print(f"❌ {env_name}: oboyu CLI not found at {oboyu_path}")
        return False
    
    # Test Python import
    if not run_command(
        [str(python_path), "-c", "import oboyu; print(f'Version: {oboyu.__version__}')"],
        env_name
    ):
        return False
    
    # Test CLI help
    if not run_command([str(oboyu_path), "--help"], env_name):
        return False
    
    # Test CLI version command
    if not run_command([str(oboyu_path), "version"], env_name):
        return False
    
    return True


def main():
    """Validate all installations."""
    print("🚀 Starting oboyu installation validation...")
    
    # Define test environments
    environments = [
        (Path("/home/testuser/venv-source"), "pip-source"),
        (Path("/home/testuser/venv-wheel"), "pip-wheel"),
        (Path("/home/testuser/venv-editable"), "pip-editable"),
        (Path("/home/testuser/oboyu/.venv"), "uv"),
    ]
    
    results = {}
    
    # Validate each environment
    for venv_path, env_name in environments:
        if venv_path.exists():
            results[env_name] = validate_installation(venv_path, env_name)
        else:
            print(f"⚠️  {env_name}: Virtual environment not found at {venv_path}")
            results[env_name] = False
    
    # Summary
    print("\n📊 Installation Validation Summary:")
    print("=" * 50)
    
    all_passed = True
    for env_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{env_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 All installation tests PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Some installation tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()