#!/usr/bin/env python3
"""Installation validation script for oboyu Docker testing."""
import subprocess  # noqa: S404
import sys


def test_installation(venv_path: str, name: str) -> bool:
    """Test that oboyu works in the given virtual environment."""
    print(f"\n{'='*60}")
    print(f"Testing {name} installation...")
    print('='*60)
    
    activate_cmd = f". {venv_path}/bin/activate"
    
    # Test import
    result = subprocess.run(  # noqa: S602
        f"{activate_cmd} && python -c \"import oboyu; print(oboyu.__version__)\"",
        shell=True,  # noqa: S602
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ {name}: Import failed!")
        print(result.stderr)
        return False
    
    print(f"✅ {name}: Import successful - Version: {result.stdout.strip()}")
    
    # Test CLI
    result = subprocess.run(  # noqa: S602
        f"{activate_cmd} && oboyu --help",
        shell=True,  # noqa: S602
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ {name}: CLI failed!")
        print(result.stderr)
        return False
    
    print(f"✅ {name}: CLI works!")
    return True

def main() -> int:
    """Run all installation validation tests."""
    tests = [
        ("/home/testuser/venv-source", "pip from source"),
        ("/home/testuser/venv-wheel", "pip from wheel"),
        ("/home/testuser/venv-editable", "pip editable installation"),
        ("/home/testuser/.venv", "UV installation"),
    ]
    
    results = []
    for venv_path, name in tests:
        results.append(test_installation(venv_path, name))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for (_, name), success in zip(tests, results):
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
