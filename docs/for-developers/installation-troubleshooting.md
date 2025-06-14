# Installation Troubleshooting Guide

This guide helps resolve common installation issues with the oboyu package.

## Quick Diagnostics

Test if oboyu is properly installed:
```bash
python -c "import oboyu; print(f'Version: {oboyu.__version__}')"
oboyu --help
```

## Common Installation Issues

### 1. Package Not Found

**Error**: `ERROR: Could not find a version that satisfies the requirement oboyu`

**Solutions**:
- If installing from source:
  ```bash
  git clone https://github.com/yourusername/oboyu.git
  cd oboyu
  pip install .
  ```
- If installing from PyPI (when available):
  ```bash
  pip install --upgrade pip
  pip install oboyu
  ```

### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'oboyu'`

**Solutions**:
- Verify installation:
  ```bash
  pip show oboyu
  ```
- Reinstall in current environment:
  ```bash
  pip uninstall oboyu
  pip install .
  ```
- Check Python path:
  ```python
  import sys
  print(sys.path)
  ```

### 3. CLI Command Not Found

**Error**: `command not found: oboyu`

**Solutions**:
- Use Python module syntax:
  ```bash
  python -m oboyu --help
  ```
- Check scripts installation:
  ```bash
  pip show -f oboyu | grep scripts
  ```
- Add scripts to PATH:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"  # Linux/macOS
  # or
  set PATH=%USERPROFILE%\AppData\Roaming\Python\Scripts;%PATH%  # Windows
  ```

### 4. Dependency Conflicts

**Error**: `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`

**Solutions**:
- Create fresh virtual environment:
  ```bash
  python -m venv fresh-env
  source fresh-env/bin/activate  # Linux/macOS
  # or
  fresh-env\Scripts\activate  # Windows
  pip install --upgrade pip
  pip install .
  ```
- Use UV for better dependency resolution:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv
  uv pip install .
  ```

### 5. PyTorch Installation Issues

**Error**: `torch` installation fails or wrong version installed

**Solutions**:
- Install PyTorch separately first:
  ```bash
  # CPU only
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  
  # CUDA 11.8
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  
  # Then install oboyu
  pip install .
  ```

### 6. DuckDB Build Errors

**Error**: Build errors related to DuckDB

**Solutions**:
- Install pre-built wheel:
  ```bash
  pip install --only-binary :all: duckdb
  ```
- On macOS with M1/M2:
  ```bash
  arch -x86_64 pip install duckdb  # If ARM build fails
  ```

### 7. Transformers Cache Issues

**Error**: `OSError: Can't load tokenizer`

**Solutions**:
- Clear Hugging Face cache:
  ```bash
  rm -rf ~/.cache/huggingface/
  ```
- Set cache directory:
  ```bash
  export HF_HOME=/path/to/cache
  export TRANSFORMERS_CACHE=/path/to/cache
  ```

## Platform-Specific Issues

### Windows

1. **Long path issues**:
   ```powershell
   # Enable long paths (requires admin)
   Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
   ```

2. **Visual C++ errors**:
   - Install Visual Studio Build Tools
   - Or use pre-built wheels

### macOS

1. **SSL certificate errors**:
   ```bash
   pip install --upgrade certifi
   ```

2. **Architecture issues (M1/M2)**:
   ```bash
   # Force x86_64 if needed
   arch -x86_64 pip install oboyu
   ```

### Linux

1. **Missing system libraries**:
   ```bash
   # Ubuntu/Debian - Complete build environment
   sudo apt-get update && sudo apt-get install -y \
       git \
       curl \
       build-essential \
       cmake \
       pkg-config \
       libssl-dev \
       python3-dev \
       libfreetype6-dev \
       libfontconfig1-dev \
       libjpeg-dev \
       libpng-dev \
       zlib1g-dev
   
   # CentOS/RHEL - Complete build environment
   sudo yum install -y \
       git \
       curl \
       gcc-c++ \
       cmake \
       pkg-config \
       openssl-devel \
       python3-devel \
       freetype-devel \
       fontconfig-devel \
       libjpeg-devel \
       libpng-devel \
       zlib-devel
   
   # macOS - Ensure Xcode tools and dependencies
   xcode-select --install
   brew install cmake pkg-config
   ```

## Environment Setup

### Recommended Installation

```bash
# 1. Create virtual environment
python -m venv oboyu-env

# 2. Activate environment
# Linux/macOS:
source oboyu-env/bin/activate
# Windows:
oboyu-env\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install oboyu
pip install .  # From source
# or
pip install oboyu  # From PyPI

# 5. Validate installation
python scripts/validate-installation.py
```

### Using UV (Recommended)

```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create project
uv venv
source .venv/bin/activate

# 3. Install
uv pip install .

# 4. Validate
python scripts/validate-installation.py
```

## Debugging Steps

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.11+
   ```

2. **Check pip version**:
   ```bash
   pip --version
   ```

3. **List installed packages**:
   ```bash
   pip list
   ```

4. **Check for conflicts**:
   ```bash
   pip check
   ```

5. **Verbose installation**:
   ```bash
   pip install -v .
   ```

6. **Clean installation**:
   ```bash
   pip uninstall -y oboyu
   pip cache purge
   pip install --no-cache-dir .
   ```

## Getting Help

If issues persist:

1. **Run diagnostic script**:
   ```bash
   python scripts/validate-installation.py > diagnostic.txt
   ```

2. **Collect system info**:
   ```bash
   python -m pip debug > pip-debug.txt
   ```

3. **Check GitHub Issues**: Look for similar problems
4. **Create Issue**: Include diagnostic output and steps to reproduce

## Testing Installation

After resolving issues, verify:

```bash
# Quick test
oboyu --version
oboyu health

# Full validation
python scripts/validate-installation.py

# Run test suite
pytest tests/test_installation_validation.py
```