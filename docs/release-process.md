# Release Process

This document describes the automated release process for the oboyu package using GitHub Actions workflows.

## Overview

The project uses three GitHub Actions workflows to automate package publishing:

1. **Test PyPI Publishing** (`test-pypi.yml`) - For development testing
2. **Prerelease Publishing** (`prerelease.yml`) - For alpha, beta, and release candidate versions
3. **Stable Release Publishing** (`release.yml`) - For production releases

## Workflows

### Test PyPI Publishing

**Triggered by:** Pushes to `main` or `develop` branches

**Purpose:** Allows testing of the package installation process without affecting the production PyPI repository.

**Environment:** `test-pypi` (requires environment configuration in repository settings)

**URL:** https://test.pypi.org/p/oboyu

### Prerelease Publishing

**Triggered by:** Tags matching prerelease patterns:
- Alpha: `v0.2.0a1`, `v0.2.0a2`, etc.
- Beta: `v0.2.0b1`, `v0.2.0b2`, etc.
- Release Candidate: `v0.2.0rc1`, `v0.2.0rc2`, etc.

**Purpose:** Publishes prerelease versions to PyPI for early testing by users.

**Environment:** `pypi-prerelease` (requires environment configuration in repository settings)

**Installation:**
```bash
# Install latest prerelease
pip install --pre oboyu

# Install specific prerelease
pip install oboyu==0.2.0a1
```

### Stable Release Publishing

**Triggered by:** Tags matching stable release pattern: `v0.2.0`, `v1.0.0`, etc.

**Purpose:** Publishes stable releases to PyPI and creates GitHub releases.

**Environment:** `pypi-release` (requires environment configuration in repository settings)

**Installation:**
```bash
pip install oboyu
```

## Release Process

### Prerequisites

1. **GitHub Environment Configuration**: Set up the following environments in repository settings:
   - `test-pypi`
   - `pypi-prerelease` 
   - `pypi-release`

2. **PyPI Trusted Publishing**: Configure trusted publishing for each environment to avoid managing API tokens.

### Making a Release

#### 1. Update Version

Update the version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # or "0.2.0a1" for prerelease
```

#### 2. Commit and Push

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0"
git push origin main
```

#### 3. Create and Push Tag

```bash
# For stable release
git tag v0.2.0
git push origin v0.2.0

# For prerelease
git tag v0.2.0a1
git push origin v0.2.0a1
```

#### 4. Monitor Workflow

The appropriate workflow will automatically:
1. Run tests (lint, type check, pytest)
2. Verify version matches tag
3. Build the package
4. Publish to PyPI
5. Create GitHub release (for stable releases)

## Version Scheme

The project follows [PEP 440](https://peps.python.org/pep-0440/) versioning:

- **Development**: `0.2.0.dev1` (not automated)
- **Alpha**: `0.2.0a1`
- **Beta**: `0.2.0b1`
- **Release Candidate**: `0.2.0rc1`
- **Stable**: `0.2.0`

## Security

- **Trusted Publishing**: Uses OpenID Connect (OIDC) for secure authentication with PyPI
- **Environment Protection**: Production releases require environment approval
- **No API Tokens**: Avoids storing sensitive API tokens in repository secrets

## Testing

Before releasing:

1. **Local Testing**:
   ```bash
   uv run ruff check --fix
   uv run mypy
   uv run pytest --cov=src
   ```

2. **Test PyPI**: Test installation from Test PyPI before tagging:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ oboyu
   ```

## Troubleshooting

### Version Mismatch Error

If the workflow fails with a version mismatch error:
1. Ensure `pyproject.toml` version matches the git tag
2. Update the version and create a new tag

### Publish Failure

If publishing fails:
1. Check environment configuration
2. Verify trusted publishing setup on PyPI
3. Ensure package name is available on PyPI

### Test Failures

If tests fail during the workflow:
1. Run tests locally to reproduce the issue
2. Fix the failing tests
3. Push the fix and re-tag if necessary