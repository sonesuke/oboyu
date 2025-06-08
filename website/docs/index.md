---
id: intro
title: Oboyu Documentation
sidebar_position: 1
slug: /
---

# Search Your Documents Like a Pro

Oboyu is a powerful command-line tool that transforms how you find information in your local documents. Whether you're managing personal notes, technical documentation, or research papers, Oboyu helps you discover the right content instantly.

## Why Choose Oboyu?

### 🚀 **Instant Search Across All Your Documents**
Stop wasting time hunting through folders. Index once, search forever.

```bash
# Index your documents
oboyu index ~/Documents

# Find what you need instantly  
oboyu query "project deadline"
```

### 🧠 **Smart Semantic Understanding**
Oboyu doesn't just match keywords—it understands meaning and context.

```bash
# These all find relevant documents:
oboyu query "budget planning"           # Finds "financial projections"
oboyu query "team meeting notes"       # Finds "standup discussions"
oboyu query "how to configure SSL"     # Finds setup tutorials
```

### 🇯🇵 **Excellent Japanese Support**
Native Japanese text processing with proper tokenization and semantic understanding.

```bash
# Search Japanese content naturally
oboyu query "機械学習の基礎"
oboyu query "明日の会議について"
```

### 🤖 **Claude AI Integration**
Connect with Claude Desktop for AI-powered document assistance.

## Common Use Cases

### 📋 **Meeting Notes & Project Management**
Never lose track of decisions, action items, or project status again.

- Find last week's meeting notes instantly
- Track project deadlines across documents
- Discover action items assigned to team members

[**Learn more →**](real-world-scenarios/meeting-notes)

### 💻 **Technical Documentation**
Search through code documentation, API references, and technical guides.

- Find configuration examples quickly
- Locate troubleshooting guides
- Search across multiple project docs

[**Learn more →**](real-world-scenarios/technical-docs)

### 📚 **Research & Academic Papers**
Organize and search through research papers, notes, and academic content.

- Find papers by topic or methodology
- Search through literature reviews
- Discover related research quickly

[**Learn more →**](real-world-scenarios/research-papers)

### 📝 **Personal Knowledge Management**
Transform your personal notes into a searchable knowledge base.

- Find old ideas and thoughts
- Search across journals and notes
- Connect related concepts

[**Learn more →**](real-world-scenarios/personal-notes)

## Get Started in 3 Minutes

### 1. Install Oboyu
```bash
pip install oboyu
```

### 2. Index Your Documents
```bash
oboyu index ~/Documents
```

### 3. Start Searching
```bash
oboyu query "what you're looking for"
```

[**Full installation guide →**](getting-started/installation)

## Real User Workflows

### Daily Knowledge Worker
```bash
# Morning routine: catch up on recent updates
oboyu query "status update OR meeting" --days 1

# Find documents for upcoming meeting
oboyu query "project alpha timeline"

# Quick reference lookup
oboyu query "API authentication examples"
```

### Research Workflow
```bash
# Find papers on specific methodology
oboyu query "quantitative analysis methods" --mode semantic

# Locate previous research notes
oboyu query "literature review notes" --days 30

# Cross-reference findings
oboyu query "similar to current-paper.pdf" --mode vector
```

### Development Workflow
```bash
# Find configuration examples
oboyu query "nginx SSL configuration"

# Search through documentation
oboyu query "error handling best practices" --file-type md

# Locate troubleshooting guides
oboyu query "deployment issues solutions"
```

[**Explore all workflows →**](usage-examples/basic-workflow)

## Tool Features That Make the Difference

### Multiple Search Modes
- **Keyword search**: Perfect for exact terms and technical identifiers
- **Semantic search**: Understands concepts and natural language
- **Hybrid mode**: Best of both worlds for highest quality results

### Smart Filtering
```bash
# Filter by file type
oboyu query "configuration" --file-type yaml,json

# Filter by date
oboyu query "meeting notes" --days 7

# Filter by location
oboyu query "API docs" --path "**/api/**"
```

### Performance Optimized
- Lightning-fast searches across millions of documents
- Incremental indexing for quick updates
- Memory-efficient processing

### Integration Ready
- **Claude Desktop**: AI-powered document assistance
- **Command line**: Scriptable and automation-friendly
- **Multiple formats**: Markdown, text, code, and more

## Documentation Structure

This documentation is organized around practical usage:

- **[Getting Started](getting-started/installation)**: Install and run your first search
- **[Usage Examples](usage-examples/basic-workflow)**: Learn essential daily workflows  
- **[Real-world Scenarios](real-world-scenarios/technical-docs)**: Domain-specific examples
- **[Configuration](configuration-optimization/configuration)**: Optimize for your needs
- **[Integration](integration/mcp-integration)**: Connect with other tools
- **[Reference](reference-troubleshooting/cli-reference)**: Complete command reference

## Community & Support

- **[GitHub Repository](https://github.com/sonesuke/oboyu)**: Source code and releases
- **[Issue Tracker](https://github.com/sonesuke/oboyu/issues)**: Bug reports and feature requests
- **[Discussions](https://github.com/sonesuke/oboyu/discussions)**: Community Q&A

## Ready to Transform Your Document Search?

[**Install Oboyu →**](getting-started/installation) or [**Try the Quick Start →**](getting-started/first-index)

---

*Oboyu is open source software licensed under the [MIT License](https://github.com/sonesuke/oboyu/blob/main/LICENSE.md)*