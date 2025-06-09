---
id: research-papers
title: Academic Paper Search Examples
sidebar_position: 3
---

# Academic Paper Search Examples

Navigate through research papers, academic articles, dissertations, and scholarly content efficiently using Oboyu's powerful search capabilities.

## Scenario: PhD Researcher

Managing hundreds of research papers across multiple topics, tracking citations, and organizing literature reviews.

### Setting Up Research Index

```bash
# Index research papers
oboyu index ~/Research/Papers --db-path ~/indexes/research.db \
    --include "*.pdf" \
    --include "*.tex" \
    --include "*.bib"

# Add thesis drafts and notes
oboyu index ~/Research/Thesis --db-path ~/indexes/thesis.db --update
oboyu index ~/Research/Notes --db-path ~/indexes/research-notes.db --update
```

### Literature Review Workflow

#### Finding Relevant Papers
```bash
# Search by topic
oboyu query --query "machine learning optimization" --db-path ~/indexes/research.db

# Find recent papers
oboyu query --query "neural networks 2023 2024" --mode hybrid

# Search by methodology
oboyu query --query "methodology quantitative analysis" --mode vector
```

#### Citation Management
```bash
# Find papers by author
oboyu query --query "author: Smith et al"

# Find papers citing specific work
oboyu query --query "Smith2023 cited reference" --mode vector

# Find papers with specific citations
oboyu query --query "\(Smith, 2023\)|\[Smith2023\]"
```

## Scenario: Research Group Leader

Coordinating multiple research projects, tracking student progress, and managing collaborative papers.

### Project Management
```bash
# Find papers by project
oboyu query --query "Project-ID: ML-OPT-2024" --db-path ~/indexes/research.db

# Track student contributions
oboyu query --query "author: PhD-Student-Name"

# Find collaborative papers
oboyu query --query "author: University-A University-B" --mode vector
```

### Grant Writing Support
```bash
# Find impact statements
oboyu query --query "impact significance contribution" --mode vector

# Gather preliminary results
oboyu query --query "preliminary results findings"

# Find methodology descriptions
oboyu query --query "methodology approach technique" --db-path ~/indexes/research.db
```

## Scenario: Academic Librarian

Helping researchers find relevant papers and managing institutional repository.

### Advanced Search Assistance
```bash
# Multi-field search
oboyu query --query "title: machine learning AND author: Chen AND year: 2023"

# Subject area search
oboyu query --query "subject: computer science AI" --mode vector

# Abstract search
oboyu query --query "abstract: novel approach deep learning" --mode vector
```

### Repository Management
```bash
# Find papers missing metadata
oboyu query --query "*.pdf" --not "author: title:"

# Find duplicate papers
# (Note: duplicates detection not available in current implementation)
oboyu query --query "*" --db-path ~/indexes/research.db

# Find papers needing OCR
# (Note: min-words filter not available in current implementation)
oboyu query --query "*.pdf"
```

## Advanced Academic Search Patterns

### Finding Specific Sections
```bash
# Find introductions
oboyu query --query "1. Introduction|1 Introduction"

# Find methodologies
oboyu query --query "Methodology|Methods|Experimental Setup" --mode vector

# Find results sections
oboyu query --query "Results|Findings|Experimental Results" 

# Find conclusions
oboyu query --query "Conclusion|Summary|Future Work" --mode vector
```

### Statistical Analysis
```bash
# Find papers with specific statistics
oboyu query --query "p < 0.05|significant at"

# Find sample sizes
oboyu query --query "n = \d+|sample size|participants"

# Find effect sizes
oboyu query --query "effect size|Cohen's d|eta squared" 
```

## Real-World Example: Systematic Literature Review

Conducting a systematic review on "AI in Healthcare":

```bash
# 1. Initial broad search
oboyu query --query "artificial intelligence healthcare medical" --db-path ~/indexes/research.db

# 2. Refine by methodology
oboyu query --query "AI healthcare clinical trial" --mode vector

# 3. Find review papers
oboyu query --query "systematic review AI healthcare" --mode hybrid

# 4. Search by specific applications
oboyu query --query "diagnosis prediction treatment AI"

# 5. Find datasets used
oboyu query --query "dataset: medical imaging ECG" --mode vector

# 6. Extract evaluation metrics
oboyu query --query "accuracy precision recall F1 AUC"

# 7. Create bibliography
oboyu query --query "references bibliography"
```

## Research Paper Patterns

### Conference vs Journal
```bash
# Find conference papers
oboyu query --query "conference proceedings ICML NeurIPS CVPR" 

# Find journal articles  
oboyu query --query "journal IEEE ACM Nature Science" 

# Find preprints
oboyu query --query "arXiv bioRxiv preprint" --mode vector
```

### Research Trends
```bash
# Track emerging topics
oboyu query --query "novel approach new method" --mode vector

# Find research gaps
oboyu query --query "future work limitation gap" --mode vector

# Identify hot topics
oboyu query --query "state-of-the-art SOTA benchmark"
```

## Citation Network Analysis

### Building Citation Maps
```bash
# Find most cited papers
# (Note: citation analysis not available in current implementation)
oboyu query --query "transformer attention BERT" --mode vector --top-k 20

# Find citation clusters
oboyu query --query "transformer attention BERT" --mode vector

# Track citation evolution
# (Note: timeline analysis not available in current implementation)
oboyu query --query "cited by subsequent work"
```

### Author Collaboration
```bash
# Find frequent collaborators
# (Note: collaboration statistics not available in current implementation)
oboyu query --query "author:"

# Find international collaborations
oboyu query --query "University Country-A Country-B" --mode vector

# Find interdisciplinary work
oboyu query --query "computer science biology interdisciplinary" --mode vector
```

## Japanese Academic Search

For Japanese research papers:

```bash
# Search Japanese papers
oboyu query --query "機械学習 深層学習" --db-path ~/indexes/research.db

# Find papers with English abstracts
oboyu query --query "Abstract 概要"

# Mixed language papers
oboyu query --query "深層学習 deep learning" --mode hybrid

# Japanese author names
oboyu query --query "著者: 山田 田中" --db-path ~/indexes/research.db
```

## Research Note Organization

### Effective Tagging
```markdown
---
title: "Paper Review: Smith et al. 2023"
tags: [machine-learning, optimization, reviewed]
project: ML-OPT-2024
date: 2024-01-15
---

## Key Findings
- Finding 1: ...
- Finding 2: ...

## Relevance to My Research
...

## Questions/Critiques
...
```

### Literature Database
```bash
# Create paper database
# (Note: extract-metadata option not available in current implementation)
oboyu index ~/Research/Papers --db-path ~/indexes/paper-db.db

# Search by metadata
oboyu query --query "year: 2023 AND venue: ICML" --db-path ~/indexes/paper-db.db

# Export for reference manager
# (Note: bibtex export not available in current implementation)
oboyu query --query "*" --db-path ~/indexes/paper-db.db
```

## Integration with Research Tools

### LaTeX Integration
```bash
# Find citations in LaTeX
oboyu query --query "\\cite{"

# Find undefined references
oboyu query --query "undefined reference"

# Find figure references
oboyu query --query "\\ref{fig:"
```

### Reference Management
```bash
# Export to BibTeX
# (Note: bibtex export not available in current implementation)
oboyu query --query "machine learning"

# Find missing citations
# (Note: not-in filter not available in current implementation)
oboyu query --query "\\cite{.*}"

# Update citation keys
# (Note: update-key functionality not available in current implementation)
oboyu query --query "author: Smith year: 2023"
```

## Research Productivity Tips

### Daily Research Routine
```bash
# Morning: Check new papers
oboyu query --query "*" --db-path ~/indexes/research.db

# Focus session: Related work
# (Note: related-to functionality not available in current implementation)
oboyu query --query "related work" --mode vector

# Evening: Update notes
oboyu query --query "TODO: read" --db-path ~/indexes/research-notes.db
```

### Paper Reading Workflow
```bash
# First pass: Abstract and conclusion
# (Note: file-specific search not available in current implementation)
oboyu query --query "abstract:|conclusion:"

# Second pass: Figures and results  
oboyu query --query "figure|table|results"

# Deep dive: Methodology
oboyu query --query "method|algorithm|implementation"
```

## Quality Control

### Finding High-Quality Papers
```bash
# Well-cited papers
oboyu query --query "cited by many highly cited" --mode vector

# Peer-reviewed only
oboyu query --query "peer-reviewed journal conference" --not "preprint arxiv"

# Reproducible research
oboyu query --query "code available github reproducible" --mode vector
```

## Next Steps

- Explore [Personal Notes Search](personal-notes.md) for research note management
- Learn about [Configuration Optimization](../configuration-optimization/configuration.md) for large paper collections
- Set up [MCP Integration](../integration/mcp-integration.md) for AI-assisted research