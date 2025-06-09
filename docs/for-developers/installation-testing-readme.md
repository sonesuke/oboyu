# Installation Testing Environment

This directory contains comprehensive testing infrastructure to ensure reliable installation of the oboyu package across different platforms and environments.

## Quick Start

### Test with Docker
```bash
# Run specific test scenario
docker compose -f docker-compose.test.yml up test-pip-source

# Run all tests
docker compose -f docker-compose.test.yml up
```

## Components


### Docker Infrastructure
- `Dockerfile.test` - Multi-stage testing environments
- `docker-compose.test.yml` - Orchestrates test scenarios

### GitHub Actions
- `.github/workflows/test-installation.yml` - Automated CI/CD testing

### Test Suite
- `tests/test_installation_validation.py` - Pytest-based validation

### Documentation
- `docs/installation-testing.md` - Detailed testing procedures
- `docs/installation-troubleshooting.md` - Troubleshooting guide

## Test Scenarios

1. **Clean pip install from source**
2. **pip install from built wheel**
3. **pip editable install**
4. **UV package manager install**
5. **Installation with common packages**
6. **Multi-platform testing** (Linux, macOS)
7. **Multiple Python versions** (3.11, 3.12, 3.13)

## For Contributors

Before submitting PRs:
```bash
# 1. (Optional) Run Docker tests locally
docker compose -f docker-compose.test.yml up test-pip-source

# 2. GitHub Actions will automatically test your PR
```

## For Users

If you encounter installation issues:
```bash
# 1. Test import
python -c "import oboyu; print(oboyu.__version__)"

# 2. Check troubleshooting guide
cat docs/installation-troubleshooting.md

# 3. Try clean installation
python -m venv fresh-env
source fresh-env/bin/activate
pip install --upgrade pip
pip install .
```

## Success Metrics

Installation is successful when:
- ✅ Package imports without errors
- ✅ CLI commands work properly
- ✅ All dependencies resolve correctly
- ✅ No conflicts with common packages
- ✅ Works across different platforms