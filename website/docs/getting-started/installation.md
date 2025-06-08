---
id: installation
title: Installation
sidebar_position: 1
---

# Installation

This guide will help you install Oboyu on your system. Oboyu is a command-line tool for semantic search across your local documents, with enhanced support for Japanese text.

## Prerequisites

- Python 3.13 or higher
- pip (Python package manager)
- Git (for development installation)

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
- **OS**: macOS 10.15+ or Linux (Ubuntu 20.04+, CentOS 8+)
  - **Note**: Windows is not currently supported

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
If you see "Python 3.13+ required":
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

### Platform-Specific Notes

#### macOS
- If using Homebrew Python, ensure it's in your PATH
- May need to install Xcode Command Line Tools: `xcode-select --install`

#### Windows
- **Windows is not currently supported**
- Consider using WSL (Windows Subsystem for Linux) as an alternative

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