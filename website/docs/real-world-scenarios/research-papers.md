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
oboyu index ~/Research/Papers --name research \
    --include "*.pdf" \
    --include "*.tex" \
    --include "*.bib"

# Add thesis drafts and notes
oboyu index ~/Research/Thesis --name thesis --update
oboyu index ~/Research/Notes --name research-notes --update
```

### Literature Review Workflow

#### Finding Relevant Papers
```bash
# Search by topic
oboyu query "machine learning optimization" --index research

# Find recent papers
oboyu query "neural networks 2023 2024" --mode hybrid

# Search by methodology
oboyu query "methodology quantitative analysis" --mode semantic
```

#### Citation Management
```bash
# Find papers by author
oboyu query "author: Smith et al" --file-type pdf

# Find papers citing specific work
oboyu query "Smith2023 cited reference" --mode semantic

# Find papers with specific citations
oboyu query "\(Smith, 2023\)|\[Smith2023\]" --regex
```

## Scenario: Research Group Leader

Coordinating multiple research projects, tracking student progress, and managing collaborative papers.

### Project Management
```bash
# Find papers by project
oboyu query "Project-ID: ML-OPT-2024" --index research

# Track student contributions
oboyu query "author: PhD-Student-Name" --file-type pdf,tex

# Find collaborative papers
oboyu query "author: University-A University-B" --mode semantic
```

### Grant Writing Support
```bash
# Find impact statements
oboyu query "impact significance contribution" --mode semantic

# Gather preliminary results
oboyu query "preliminary results findings" --file-type pdf

# Find methodology descriptions
oboyu query "methodology approach technique" --index research
```

## Scenario: Academic Librarian

Helping researchers find relevant papers and managing institutional repository.

### Advanced Search Assistance
```bash
# Multi-field search
oboyu query "title: machine learning AND author: Chen AND year: 2023"

# Subject area search
oboyu query "subject: computer science AI" --mode semantic

# Abstract search
oboyu query "abstract: novel approach deep learning" --mode semantic
```

### Repository Management
```bash
# Find papers missing metadata
oboyu query "*.pdf" --not "author: title:" --regex

# Find duplicate papers
oboyu query --duplicates --index research

# Find papers needing OCR
oboyu query "*.pdf" --min-words 100 --file-type pdf
```

## Advanced Academic Search Patterns

### Finding Specific Sections
```bash
# Find introductions
oboyu query "1. Introduction|1 Introduction" --file-type pdf

# Find methodologies
oboyu query "Methodology|Methods|Experimental Setup" --mode semantic

# Find results sections
oboyu query "Results|Findings|Experimental Results" 

# Find conclusions
oboyu query "Conclusion|Summary|Future Work" --mode semantic
```

### Statistical Analysis
```bash
# Find papers with specific statistics
oboyu query "p &lt; 0.05|significant at" --regex

# Find sample sizes
oboyu query "n = \d+|sample size|participants" --regex

# Find effect sizes
oboyu query "effect size|Cohen's d|eta squared" 
```

## Real-World Example: Systematic Literature Review

Conducting a systematic review on "AI in Healthcare":

```bash
# 1. Initial broad search
oboyu query "artificial intelligence healthcare medical" --index research --export initial-search.txt

# 2. Refine by methodology
oboyu query "AI healthcare clinical trial" --mode semantic

# 3. Find review papers
oboyu query "systematic review AI healthcare" --mode hybrid

# 4. Search by specific applications
oboyu query "diagnosis prediction treatment AI" --file-type pdf

# 5. Find datasets used
oboyu query "dataset: medical imaging ECG" --mode semantic

# 6. Extract evaluation metrics
oboyu query "accuracy precision recall F1 AUC" --context 200

# 7. Create bibliography
oboyu query "references bibliography" --path "**/*healthcare*.pdf" --export bibliography.txt
```

## Research Paper Patterns

### Conference vs Journal
```bash
# Find conference papers
oboyu query "conference proceedings ICML NeurIPS CVPR" 

# Find journal articles  
oboyu query "journal IEEE ACM Nature Science" 

# Find preprints
oboyu query "arXiv bioRxiv preprint" --mode semantic
```

### Research Trends
```bash
# Track emerging topics
oboyu query "novel approach new method" --days 365 --mode semantic

# Find research gaps
oboyu query "future work limitation gap" --mode semantic

# Identify hot topics
oboyu query "state-of-the-art SOTA benchmark" --days 180
```

## Citation Network Analysis

### Building Citation Maps
```bash
# Find most cited papers
oboyu query --citation-count --sort citations --limit 20

# Find citation clusters
oboyu query "transformer attention BERT" --mode semantic --related

# Track citation evolution
oboyu query "cited by subsequent work" --timeline
```

### Author Collaboration
```bash
# Find frequent collaborators
oboyu query "author:" --stats collaborations

# Find international collaborations
oboyu query "University Country-A Country-B" --mode semantic

# Find interdisciplinary work
oboyu query "computer science biology interdisciplinary" --mode semantic
```

## Japanese Academic Search

For Japanese research papers:

```bash
# Search Japanese papers
oboyu query "機械学習 深層学習" --index research

# Find papers with English abstracts
oboyu query "Abstract 概要" --file-type pdf

# Mixed language papers
oboyu query "深層学習 deep learning" --mode hybrid

# Japanese author names
oboyu query "著者: 山田 田中" --index research
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
oboyu index ~/Research/Papers --extract-metadata --name paper-db

# Search by metadata
oboyu query "year: 2023 AND venue: ICML" --index paper-db

# Export for reference manager
oboyu query "*" --index paper-db --format bibtex --export references.bib
```

## Integration with Research Tools

### LaTeX Integration
```bash
# Find citations in LaTeX
oboyu query "\\cite{" --file-type tex

# Find undefined references
oboyu query "undefined reference" --file-type log

# Find figure references
oboyu query "\\ref{fig:" --file-type tex
```

### Reference Management
```bash
# Export to BibTeX
oboyu query "machine learning" --format bibtex

# Find missing citations
oboyu query "\\cite{.*}" --not-in references.bib

# Update citation keys
oboyu query "author: Smith year: 2023" --update-key Smith2023
```

## Research Productivity Tips

### Daily Research Routine
```bash
# Morning: Check new papers
oboyu query "*" --days 1 --index research

# Focus session: Related work
oboyu query "related to current-paper.pdf" --mode semantic

# Evening: Update notes
oboyu query "TODO: read" --index research-notes
```

### Paper Reading Workflow
```bash
# First pass: Abstract and conclusion
oboyu query "abstract:|conclusion:" --file paper.pdf

# Second pass: Figures and results  
oboyu query "figure|table|results" --file paper.pdf

# Deep dive: Methodology
oboyu query "method|algorithm|implementation" --file paper.pdf
```

## Quality Control

### Finding High-Quality Papers
```bash
# Well-cited papers
oboyu query "cited by many highly cited" --mode semantic

# Peer-reviewed only
oboyu query "peer-reviewed journal conference" --not "preprint arxiv"

# Reproducible research
oboyu query "code available github reproducible" --mode semantic
```

## Next Steps

- Explore [Personal Notes Search](personal-notes.md) for research note management
- Learn about [Configuration Optimization](../configuration-optimization/configuration.md) for large paper collections
- Set up [MCP Integration](../integration/mcp-integration.md) for AI-assisted research