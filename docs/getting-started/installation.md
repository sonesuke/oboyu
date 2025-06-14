---
id: installation
title: Installation
sidebar_position: 1
---

# Installation

This guide will help you install Oboyu on your system. Oboyu is a command-line tool for semantic search across your local documents, with enhanced support for Japanese text.

## Prerequisites

- Python 3.13 or higher (3.11+ supported, 3.13 recommended)
- pip (Python package manager, latest version recommended)
- Operating System: Linux, macOS, or Windows with WSL

### System Dependencies (for building from source)

Most users installing via `pip install oboyu` won't need these dependencies as we provide pre-built wheels. However, if you're building from source or encounter compilation issues, install the following:

> **Why these dependencies?** Oboyu depends on several Python packages with native extensions:
> - **fasttext**: Fast text classification and representation learning (C++)
> - **sentencepiece**: Text tokenization (C++)
> - **cryptography**: Cryptographic operations (Rust/C)
> - **pymupdf**: PDF processing (C++)
> - **Image processing libraries**: For handling images in PDFs

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update && sudo apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libfreetype6-dev \
    libfontconfig1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libssl-dev \
    python3-dev
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install -y \
    git \
    curl \
    gcc-c++ \
    cmake \
    pkg-config \
    freetype-devel \
    fontconfig-devel \
    libjpeg-devel \
    libpng-devel \
    zlib-devel \
    openssl-devel \
    python3-devel
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install additional dependencies via Homebrew
brew install cmake pkg-config
```

## Quick Install

Install Oboyu using pip:

```bash
pip install oboyu
```

## Verify Installation

After installation, verify that Oboyu is working correctly:

```bash
oboyu --version
```

You should see the version number displayed.

## System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended for larger document collections)
- **Storage**: 500MB for Oboyu installation + space for your document index
- **OS**: macOS 10.15+, Linux (Ubuntu 20.04+, CentOS 8+), or Windows with WSL2

### Recommended Specifications
- **RAM**: 8GB or more
- **Storage**: SSD with at least 10GB free space
- **CPU**: Multi-core processor for faster indexing

## Installation Methods

### Using pip (Recommended)

```bash
pip install oboyu
```

### Using pipx (Isolated Environment)

For a cleaner installation that won't interfere with other Python packages:

```bash
pipx install oboyu
```

### Development Installation

If you want to contribute or use the latest features:

```bash
git clone https://github.com/sonesuke/oboyu.git
cd oboyu
pip install -e .
```

## Post-Installation Setup

### 1. Create a Configuration Directory

Oboyu stores its configuration and indices in your home directory:

```bash
mkdir -p ~/.oboyu
```

### 2. Language Models

Oboyu includes built-in Japanese language support with pre-configured models:
- Japanese tokenizer models (automatically loaded)
- Semantic embedding models optimized for Japanese text

No additional model downloads are required.

### 3. Environment Variables (Optional)

Oboyu supports the following environment variable:

```bash
# Set custom database path
export OBOYU_DB_PATH=/path/to/your/database
```

## Troubleshooting Installation

### Common Issues

#### Python Version Error
If you see "Python 3.11+ required":
```bash
python --version  # Check your Python version
# Consider using pyenv or conda to manage Python versions
```

#### Permission Denied
If you encounter permission errors:
```bash
pip install --user oboyu  # Install for current user only
```

#### Missing Dependencies
All required dependencies are included in the base installation. If you encounter issues:
```bash
pip install --upgrade oboyu  # Update to latest version
```

#### Build Errors (Native Dependencies)
If you encounter build errors for native dependencies (sentencepiece, fasttext, cryptography, etc.):

```bash
# Linux (Ubuntu/Debian) - Install complete build environment
sudo apt-get update && sudo apt-get install -y \
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

# Linux (CentOS/RHEL)
sudo yum install -y gcc-c++ cmake pkg-config openssl-devel python3-devel \
    freetype-devel fontconfig-devel libjpeg-devel libpng-devel zlib-devel

# macOS
xcode-select --install
brew install cmake pkg-config

# Then retry installation
pip install oboyu
```

If you still encounter issues, try upgrading pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
pip install oboyu
```

### Platform-Specific Notes

#### macOS
- If using Homebrew Python, ensure it's in your PATH
- May need to install Xcode Command Line Tools: `xcode-select --install`

#### Windows (WSL2)
- Install WSL2 following [Microsoft's guide](https://docs.microsoft.com/en-us/windows/wsl/install)
- Use Ubuntu or another Linux distribution within WSL2
- Follow the Linux installation instructions

#### Linux
- May need to install python3-dev package: `sudo apt-get install python3-dev`
- Some distributions require pip3 instead of pip

## Next Steps

Now that you have Oboyu installed, proceed to:
- [Creating Your First Index](first-index.md) - Learn how to index your documents
- [Executing Your First Search](first-search.md) - Start searching your documents

## Updating Oboyu

To update to the latest version:

```bash
pip install --upgrade oboyu
```

## Uninstalling

If you need to uninstall Oboyu:

```bash
pip uninstall oboyu
```

To remove all data and configuration:
```bash
rm -rf ~/.oboyu  # Warning: This will delete all your indices
```