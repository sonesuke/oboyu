# Release Process

This document describes the automated release process for the oboyu package using GitHub Actions workflows with changelog generation and GitHub Packages support.

## Overview

The project uses three GitHub Actions workflows to automate package publishing:

1. **Test PyPI Publishing** (`test-pypi.yml`) - For development testing
2. **Prerelease Publishing** (`prerelease.yml`) - For alpha, beta, and release candidate versions with GitHub releases
3. **Stable Release Publishing** (`release.yml`) - For production releases with automatic changelog generation

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

**Purpose:** 
- Publishes prerelease versions to PyPI for early testing
- Creates GitHub prerelease with changelog
- Publishes to GitHub Packages

**Environment:** `pypi-prerelease` (requires environment configuration in repository settings)

**Features:**
- Automatic prerelease type detection (Alpha/Beta/RC)
- Changelog generation from conventional commits
- GitHub release creation with appropriate prerelease flag

**Installation:**
```bash
# Install latest prerelease
pip install --pre oboyu

# Install specific prerelease
pip install oboyu==0.2.0a1
```

### Stable Release Publishing

**Triggered by:** Tags matching stable release pattern: `v0.2.0`, `v1.0.0`, etc.

**Purpose:** 
- Publishes stable releases to PyPI
- Creates GitHub releases with auto-generated changelog
- Updates CHANGELOG.md file
- Publishes to GitHub Packages

**Environment:** `pypi-release` (requires environment configuration in repository settings)

**Features:**
- Automatic changelog generation from conventional commits
- Categorized changelog (Features, Bug Fixes, Documentation, etc.)
- CHANGELOG.md auto-update and commit
- Release assets upload (wheels and source distributions)

**Installation:**
```bash
pip install oboyu

# Or from GitHub Packages
pip install oboyu --index-url https://ghcr.io/sonesuke/oboyu
```

## Release Process

### Prerequisites

1. **GitHub Environment Configuration**: Set up the following environments in repository settings:
   - `test-pypi`
   - `pypi-prerelease` 
   - `pypi-release`

2. **PyPI Trusted Publishing**: Configure trusted publishing for each environment to avoid managing API tokens.

3. **Conventional Commits**: Ensure all commits follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for proper changelog generation:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `chore:` for maintenance tasks
   - `refactor:` for code refactoring
   - `test:` for test additions/changes
   - `style:` for code style changes

### Making a Release

#### 1. Create and Push Tag

The package version is automatically determined from Git tags using `hatch-vcs`. No manual version updates in `pyproject.toml` are required.

```bash
# For stable release
git tag v0.2.0
git push origin v0.2.0

# For prerelease
git tag v0.2.0a1
git push origin v0.2.0a1
```

#### 2. Monitor Workflow

The appropriate workflow will automatically:
1. Run tests (lint, type check, pytest)
2. Build the package with version from Git tag
3. Publish to PyPI
4. Publish to GitHub Packages
5. Generate changelog from conventional commits
6. Create GitHub release with changelog
7. Update CHANGELOG.md (for stable releases)
8. Upload release assets (wheels and source distributions)

## Version Scheme

### Automatic Versioning

The project uses `hatch-vcs` to automatically determine versions from Git tags, following [PEP 440](https://peps.python.org/pep-0440/) versioning:

- **Alpha**: `0.2.0a1` (from tag `v0.2.0a1`)
- **Beta**: `0.2.0b1` (from tag `v0.2.0b1`)
- **Release Candidate**: `0.2.0rc1` (from tag `v0.2.0rc1`)
- **Stable**: `0.2.0` (from tag `v0.2.0`)
- **Development**: `0.2.0.dev1+g1234567` (automatically generated for untagged commits)

### Tag Format

Tags must follow the format `v{version}` where `{version}` matches PEP 440:
- `v0.2.0` → `0.2.0`
- `v0.2.0a1` → `0.2.0a1`
- `v0.2.0b1` → `0.2.0b1`
- `v0.2.0rc1` → `0.2.0rc1`

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

## Changelog Generation

### Automatic Changelog

The release workflows automatically generate changelogs based on:
- Pull request titles following conventional commit format
- Commit messages between releases
- GitHub labels on pull requests

### Changelog Categories

Changes are organized into categories:
- 🚀 **Features** - New features and enhancements
- 🐛 **Bug Fixes** - Bug fixes
- 📚 **Documentation** - Documentation updates
- 🔧 **Maintenance** - Chores and refactoring
- 🎨 **Style** - Code style changes
- 🚨 **Tests** - Test additions and changes

### CHANGELOG.md Updates

For stable releases, the CHANGELOG.md file is automatically updated with:
- Version number and release date
- Categorized list of changes
- Links to pull requests and commits

## GitHub Packages

### Publishing

Both stable and prerelease versions are published to GitHub Packages alongside PyPI. This provides:
- Alternative package registry
- Better integration with GitHub ecosystem
- Package visibility in repository

### Installation from GitHub Packages

```bash
# Install latest stable
pip install oboyu --index-url https://ghcr.io/sonesuke/oboyu

# Install prerelease
pip install --pre oboyu --index-url https://ghcr.io/sonesuke/oboyu
```

## Troubleshooting

### Tag Format Error

If the workflow fails due to tag format issues:
1. Ensure tags follow the `v{version}` format (e.g., `v0.2.0`, `v0.2.0a1`)
2. Verify the tag triggers the correct workflow pattern
3. Delete incorrect tags with `git tag -d <tagname>` and `git push origin :refs/tags/<tagname>`

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

### Changelog Generation Issues

If changelog is empty or incorrect:
1. Ensure pull requests have proper labels
2. Verify commit messages follow conventional format
3. Check that PRs are merged to the main branch

### GitHub Packages Publishing

If GitHub Packages publishing fails:
1. Verify repository has packages write permissions
2. Check GitHub token has correct scopes
3. Ensure package naming follows GitHub requirements