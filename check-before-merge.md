# Pre-Merge Checklist

## Installation Testing Infrastructure

This PR introduces comprehensive pip installation testing to ensure package reliability across different environments.

### ✅ **What's Included**

1. **Docker-based Testing Environment**
   - Multi-stage Dockerfile for isolated testing
   - Docker Compose orchestration for different scenarios
   - Tests: pip from source, wheel, UV installation, conflict detection

2. **GitHub Actions Integration**
   - Automated testing on main/develop branch pushes
   - Full testing on releases (blocks release if tests fail)
   - Daily scheduled runs for dependency monitoring
   - Manual dispatch available

3. **Installation Validation Suite**
   - pytest-based tests for installation validation
   - Import tests, CLI functionality tests, dependency conflict detection

4. **Documentation**
   - Installation testing guide
   - Troubleshooting documentation
   - Updated prerequisites in README and installation docs

### 🧪 **Test Coverage**

- **Platforms**: Ubuntu (with macOS/Windows WSL planned)
- **Python Versions**: 3.13 (with 3.11/3.12 planned for full runs)
- **Installation Methods**: pip source, pip wheel, editable, UV
- **Conflict Testing**: Common packages (numpy, pandas, requests, etc.)

### 🔒 **Release Safety**

- Installation tests are **mandatory** before any release
- Tests must pass for both stable releases and prereleases
- Prevents publishing broken packages to PyPI

### 📋 **Files Added/Modified**

- `.github/workflows/test-installation.yml` - New installation testing workflow
- `.github/workflows/release.yml` - Updated to include installation tests
- `.github/workflows/prerelease.yml` - Updated to include installation tests
- `Dockerfile.test` - Multi-stage Docker testing environment
- `docker-compose.test.yml` - Test orchestration
- `tests/test_installation_validation.py` - Installation validation tests
- `docs/installation-testing*.md` - Testing documentation
- `README.md`, `docs/quickstart.md`, `website/docs/getting-started/installation.md` - Updated prerequisites

### 🎯 **Benefits**

1. **Early Detection**: Catch installation issues before users encounter them
2. **Platform Confidence**: Ensure compatibility across different environments
3. **Dependency Safety**: Detect conflicts with common packages
4. **Release Quality**: Automated testing prevents broken releases

### ⚠️ **Things to Verify After Merge**

1. GitHub Actions workflows execute correctly
2. Installation tests pass on main branch
3. Release workflows properly integrate installation testing
4. Documentation is accessible and accurate

### 🚀 **Next Steps (Future Enhancements)**

1. Expand to full Python version matrix (3.11, 3.12, 3.13)
2. Add macOS testing to CI/CD
3. Include Windows WSL testing
4. Add performance benchmarks for installation time
5. Test PyPI upload/download cycle

---

**Ready for merge when:**
- [ ] All GitHub Actions tests pass
- [ ] Documentation has been reviewed
- [ ] Installation testing workflow validates correctly
- [ ] No breaking changes to existing functionality