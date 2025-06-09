# Installation Testing Guide

This guide describes the comprehensive testing environment for validating oboyu package installation across different platforms and scenarios.

## Overview

The installation testing framework ensures that oboyu can be reliably installed in various environments, catching dependency conflicts and platform-specific issues before they affect users.

## Testing Components

### 1. Docker-based Testing

Docker provides isolated environments for testing installation in clean conditions.

#### Files:
- `Dockerfile.test` - Multi-stage Dockerfile with different installation scenarios
- `docker-compose.test.yml` - Orchestrates multiple test environments

#### Running Docker Tests:
```bash
# Run specific test
docker compose -f docker-compose.test.yml up test-pip-source

# Build and test specific stage
docker build -f Dockerfile.test --target test-pip-wheel -t oboyu-test:wheel .
docker run --rm oboyu-test:wheel

# Run all tests
docker compose -f docker-compose.test.yml up
```

#### Test Scenarios:
1. **pip from source** - Tests `pip install .`
2. **pip from wheel** - Tests building and installing wheel
3. **UV installation** - Tests installation using UV package manager
4. **With conflicts** - Tests installation alongside common packages
5. **Test PyPI** - Simulates installation from Test PyPI

### 2. GitHub Actions Workflow

Automated testing on every push, pull request, and release.

#### File:
- `.github/workflows/test-installation.yml`

#### Test Matrix:
- **Operating Systems**: Ubuntu, macOS (Windows via WSL only)
- **Python Versions**: 3.11, 3.12, 3.13
- **Installation Methods**: source, wheel, editable
- **Package Managers**: pip, UV

#### Triggers:
- Push to main/develop branches
- Pull requests
- **Release and prerelease workflows** (critical for deployment safety)
- Manual dispatch
- Daily scheduled runs

### 3. Installation Validation Tests

Pytest-based tests that validate installation.

#### File:
- `tests/test_installation_validation.py`

#### Running:
```bash
# Run installation validation tests
pytest tests/test_installation_validation.py -v

# Run specific test
pytest tests/test_installation_validation.py::TestInstallationValidation::test_package_import
```

## Testing Procedures

### For Contributors

Before submitting a PR:

1. **Run Docker tests** (if you have Docker):
   ```bash
   docker compose -f docker-compose.test.yml up test-pip-source
   ```

2. **Check GitHub Actions** - The installation tests will run automatically on your PR

### For Maintainers

Release process (automated):

1. **Tagging triggers automatic testing** - Installation tests run before any release
2. **Release workflow dependency** - Package is only published if all tests pass
3. **Manual testing** (optional):
   ```bash
   # Test wheel building locally
   python -m build
   twine check dist/*
   
   # Test in clean environment
   docker compose -f docker-compose.test.yml up test-runner
   ```

The installation tests are automatically integrated into the release workflows:
- `release.yml` - For stable releases (v1.0.0)
- `prerelease.yml` - For alpha/beta/rc releases (v1.0.0a1)

## Troubleshooting Installation Issues

### Common Problems

1. **Import Error**
   - Check Python version (3.11+)
   - Verify all dependencies installed
   - Run: `pip check`

2. **CLI Not Found**
   - Ensure scripts directory is in PATH
   - Try: `python -m oboyu` instead

3. **Dependency Conflicts**
   - Create fresh virtual environment
   - Install in isolated environment
   - Check conflicting packages with `pip check`

### Debug Commands

```bash
# Check installation
pip show oboyu

# List all dependencies
pip list

# Check for conflicts
pip check

# Validate installation
python -c "import oboyu; print(oboyu.__version__)"

# Test CLI
oboyu --help
# or
python -m oboyu --help
```

## Adding New Tests

### Docker Test Stage

Add to `Dockerfile.test`:
```dockerfile
FROM base as test-new-scenario
# Your test scenario
```


### GitHub Actions

Add to matrix in `.github/workflows/test-installation.yml`:
```yaml
matrix:
  include:
    - os: ubuntu-latest
      python-version: '3.13'
      test-scenario: 'new-test'
```

## Best Practices

1. **Always test in clean environments** - Avoid pollution from existing packages
2. **Test multiple Python versions** - Ensure compatibility
3. **Include dependency conflict tests** - Common packages might conflict
4. **Validate both import and CLI** - Both should work after installation
5. **Document failures** - Help users troubleshoot

## Environment Variables

- `TEST_NAME` - Identifies test in logs
- `PIP_INDEX_URL` - Override PyPI index
- `PIP_EXTRA_INDEX_URL` - Additional package index

## Success Criteria

Installation is considered successful when:

- [ ] Package imports without errors
- [ ] All submodules are accessible  
- [ ] CLI commands execute properly
- [ ] No dependency conflicts reported
- [ ] Basic functionality works

## Maintenance

- Run tests weekly via GitHub Actions schedule
- Update test matrix when adding Python versions
- Add new conflict scenarios as discovered
- Keep Docker images updated