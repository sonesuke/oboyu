# CSV Data Enrichment Use Case

## Overview

Oboyu's CSV enrichment feature allows you to automatically populate CSV columns with relevant information from your indexed knowledge base using semantic search and GraphRAG. This powerful capability transforms basic tabular data into rich, informative datasets by leveraging AI-powered information extraction.

The `oboyu enrich` command processes CSV files according to configurable schemas, supporting multiple extraction strategies to find and extract relevant information from your knowledge base.

## Key Features

- **Multiple Extraction Strategies**: Search content, extract entities, or follow graph relationships
- **Flexible Configuration**: JSON schema-based configuration with customizable query templates
- **Batch Processing**: Efficient processing with configurable batch sizes
- **GraphRAG Integration**: Enhanced semantic search with graph context
- **Progress Tracking**: Real-time progress visualization with detailed statistics
- **Error Recovery**: Robust error handling with partial completion support

## Sample Scenario: Enriching Company Database

Let's walk through a practical example of enriching a company database with business information, industry classifications, and key metrics.

### Initial Data

Suppose you have a simple CSV file `companies.csv` containing basic company information:

```csv
company_name,industry
株式会社ソフトバンク,通信
トヨタ自動車株式会社,自動車
楽天グループ株式会社,EC・インターネット
```

### Goal

Enrich this data with additional information:
- Company descriptions and business overviews
- Employee counts
- Founding years
- Market capitalization data

## Step-by-Step Guide

### Step 1: Prepare Your Knowledge Base

First, ensure your knowledge base contains relevant information about these companies:

```bash
# Index company-related documents
oboyu index documents/company-reports/
oboyu index documents/financial-data/
oboyu index documents/news-articles/
```

### Step 2: Create Enrichment Schema

Create a configuration file `enrichment-schema.json` that defines how to enrich your data:

```json
{
  "input_schema": {
    "columns": {
      "company_name": {
        "type": "string",
        "description": "会社名",
        "required": true
      },
      "industry": {
        "type": "string",
        "description": "業界",
        "required": false
      }
    },
    "primary_keys": ["company_name"]
  },
  "enrichment_schema": {
    "columns": {
      "description": {
        "type": "string",
        "description": "会社の概要・事業内容",
        "source_strategy": "search_content",
        "query_template": "{company_name} 概要 事業内容 ビジネスモデル",
        "extraction_method": "summarize"
      },
      "employees": {
        "type": "integer",
        "description": "従業員数",
        "source_strategy": "search_content",
        "query_template": "{company_name} 従業員数 社員数",
        "extraction_method": "pattern_match",
        "extraction_pattern": "\\d+(?:人|名|万人)"
      },
      "founded_year": {
        "type": "integer",
        "description": "設立年",
        "source_strategy": "graph_relations",
        "query_template": "{company_name} 設立 創業",
        "relation_types": ["FOUNDED_IN", "ESTABLISHED_IN"],
        "target_entity_types": ["DATE", "YEAR"]
      },
      "market_cap": {
        "type": "string",
        "description": "時価総額",
        "source_strategy": "search_content",
        "query_template": "{company_name} 時価総額 企業価値",
        "extraction_method": "pattern_match",
        "extraction_pattern": "\\d+(?:兆|億|万)円"
      }
    }
  },
  "search_config": {
    "search_mode": "hybrid",
    "use_graphrag": true,
    "rerank": true,
    "top_k": 5,
    "similarity_threshold": 0.5
  }
}
```

### Step 3: Run Enrichment Command

Execute the enrichment process:

```bash
# Basic enrichment
oboyu enrich companies.csv enrichment-schema.json

# With custom options
oboyu enrich companies.csv enrichment-schema.json \
  --output enriched-companies.csv \
  --batch-size 5 \
  --confidence 0.7 \
  --max-results 3
```

### Step 4: Analyze Results

The enriched output `companies_enriched.csv` will contain:

```csv
company_name,industry,description,employees,founded_year,market_cap
株式会社ソフトバンク,通信,"通信事業を中核とし、インターネット関連事業、AI・IoT事業を展開する総合テクノロジーグループ",80000,1981,7兆円
トヨタ自動車株式会社,自動車,"世界最大級の自動車メーカーで、ハイブリッド技術のパイオニア。グローバルに自動車製造・販売を展開",370000,1937,35兆円
楽天グループ株式会社,EC・インターネット,"eコマース、フィンテック、モバイル通信など70以上のサービスを提供するインターネット・サービス企業",28000,1997,1兆円
```

## Extraction Strategies Deep Dive

### 1. Search Content Strategy (`search_content`)

Performs semantic search against your knowledge base and extracts relevant text content.

**Extraction Methods:**

- **`first_result`** (default): Returns first 200 characters of the top search result
- **`first_sentence`**: Extracts the first complete sentence with proper punctuation
- **`summarize`**: Combines information from top 3 results into a concise summary
- **`pattern_match`**: Uses regex patterns to extract specific data formats

**Example Configuration:**
```json
{
  "company_revenue": {
    "type": "string",
    "source_strategy": "search_content",
    "query_template": "{company_name} 売上 収益 年間売上高",
    "extraction_method": "pattern_match",
    "extraction_pattern": "\\d+(?:兆|億|万)?円"
  }
}
```

### 2. Entity Extraction Strategy (`entity_extraction`)

Extracts specific entities from the knowledge graph based on entity types and similarity.

**Configuration Options:**
- `entity_types`: Filter by specific entity types (e.g., `["PERSON", "ORGANIZATION"]`)
- `similarity_threshold`: Minimum similarity score for entity matching
- `max_entities`: Maximum number of entities to return

**Example Configuration:**
```json
{
  "ceo_name": {
    "type": "string",
    "source_strategy": "entity_extraction",
    "query_template": "{company_name} CEO 代表取締役 社長",
    "entity_types": ["PERSON"],
    "similarity_threshold": 0.8
  }
}
```

### 3. Graph Relations Strategy (`graph_relations`)

Follows knowledge graph relationships to discover connected information.

**Configuration Options:**
- `relation_types`: Types of relationships to follow (e.g., `["FOUNDED_IN", "LOCATED_IN"]`)
- `target_entity_types`: Types of target entities to find
- `max_hops`: Maximum relationship traversal depth

**Example Configuration:**
```json
{
  "headquarters": {
    "type": "string",
    "source_strategy": "graph_relations",
    "query_template": "{company_name} 本社 所在地",
    "relation_types": ["LOCATED_IN", "HEADQUARTERED_IN"],
    "target_entity_types": ["LOCATION", "CITY"]
  }
}
```

## Advanced Configuration Options

### Search Configuration

Fine-tune search behavior for optimal results:

```json
{
  "search_config": {
    "search_mode": "hybrid",          // vector, bm25, or hybrid
    "use_graphrag": true,             // Enable GraphRAG enhancement
    "rerank": true,                   // Enable result reranking
    "top_k": 5,                       // Number of search results
    "similarity_threshold": 0.5,      // Minimum similarity score
    "max_tokens": 4000               // Maximum tokens per search
  }
}
```

### Template Variables

Use dynamic query construction with template variables:

```json
{
  "query_template": "{company_name} {industry} 業界 市場シェア",
  "context_template": "業界: {industry}, 地域: 日本"
}
```

### Batch Processing Options

Optimize performance with batch processing:

```bash
# Small datasets (faster feedback)
oboyu enrich data.csv schema.json --batch-size 5

# Large datasets (better throughput)
oboyu enrich data.csv schema.json --batch-size 20

# Memory-constrained environments
oboyu enrich data.csv schema.json --batch-size 2
```

## Performance and Best Practices

### Query Template Design

1. **Be Specific**: Include relevant context and keywords
   ```json
   // Good
   "query_template": "{company_name} 従業員数 正社員 社員数"
   
   // Too generic
   "query_template": "{company_name} 情報"
   ```

2. **Use Multiple Keywords**: Include synonyms and variations
   ```json
   "query_template": "{company_name} 設立 創業 創立 会社設立"
   ```

3. **Include Industry Context**: Add industry-specific terms when available
   ```json
   "query_template": "{company_name} {industry} 事業内容 ビジネスモデル"
   ```

### Strategy Selection Guidelines

- **Use `search_content`** for: Descriptions, summaries, general information
- **Use `entity_extraction`** for: Names, specific entities, structured data
- **Use `graph_relations`** for: Related facts, connections, hierarchical data

### Error Handling and Validation

The enrichment process includes robust error handling:

- **Partial Completion**: Continue processing even if some cells fail
- **Validation**: Schema validation before processing begins
- **Logging**: Detailed error logs for troubleshooting
- **Recovery**: Resume processing from interruption points

### Confidence Thresholds

Adjust confidence levels based on your accuracy requirements:

```bash
# High precision (fewer results, higher accuracy)
oboyu enrich data.csv schema.json --confidence 0.8

# Balanced approach
oboyu enrich data.csv schema.json --confidence 0.5

# High recall (more results, potentially lower accuracy)
oboyu enrich data.csv schema.json --confidence 0.3
```

## Common Use Cases

### 1. Customer Database Enrichment
- Enrich customer lists with company information
- Add industry classifications and company sizes
- Include contact information and business details

### 2. Financial Analysis
- Populate financial metrics from reports
- Add market data and performance indicators
- Include regulatory and compliance information

### 3. Research and Analysis
- Enrich research datasets with background information
- Add contextual data from multiple sources
- Create comprehensive analytical datasets

### 4. Data Migration and Integration
- Enhance legacy data with modern information
- Integrate data from multiple sources
- Standardize and enrich imported data

## Troubleshooting

### Common Issues and Solutions

**Low Enrichment Success Rate:**
- Increase search results: `--max-results 10`
- Lower confidence threshold: `--confidence 0.3`
- Improve query templates with more keywords
- Check if knowledge base contains relevant information

**Memory or Performance Issues:**
- Reduce batch size: `--batch-size 5`
- Disable GraphRAG for simple searches: `--no-graph`
- Use vector-only search: `"search_mode": "vector"`

**Schema Validation Errors:**
- Ensure all required fields are present
- Check column name conflicts between input and enrichment schemas
- Validate JSON syntax and structure
- Verify strategy-specific configuration options

**Missing Results for Specific Strategies:**
- **Entity Extraction**: Lower similarity threshold, check entity types
- **Graph Relations**: Verify relation types exist in knowledge graph
- **Search Content**: Improve query templates, check search mode

## Next Steps

After successful CSV enrichment:

1. **Validate Results**: Review enriched data for accuracy and completeness
2. **Iterate Schema**: Refine configuration based on results
3. **Automate Workflows**: Integrate enrichment into data processing pipelines
4. **Monitor Performance**: Track enrichment success rates and adjust parameters
5. **Expand Knowledge Base**: Add more relevant documents to improve coverage

The CSV enrichment feature transforms basic tabular data into rich, informative datasets, making Oboyu a powerful tool for data analysis and knowledge extraction workflows.