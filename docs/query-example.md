# Query Examples

## Basic Query

```bash
$ oboyu query "システムの設計原則について教えてください"

Results for: "システムの設計原則について教えてください"
----------------------------------------------------------

タイトル: システム設計の基本原則
スニペット: "...当システムの設計は「シンプルさ」「モジュール性」「拡張性」の
三つの原則に基づいています。これらの原則は開発の初期段階から一貫して守られ、
各コンポーネントの実装において重視されています..."
URI: file:///projects/docs/設計/principles.md
Score: 0.91


タイトル: アーキテクチャ概要
スニペット: "...設計原則としては、「関心の分離」を徹底し、各モジュールが
単一の責任を持つようにしています。これにより、テストが容易になり、
将来の変更にも柔軟に対応できるシステムとなっています..."
URI: file:///projects/docs/アーキテクチャ/overview.md
Score: 0.85


タイトル: プロジェクト概要
スニペット: "...本システムは、設計の段階から「ユーザー中心設計」の原則を
採用しており、すべての機能はユーザーの具体的なニーズに基づいて
実装されています。特に検索機能においては、日本語処理に優れた..."
URI: file:///projects/README.md
Score: 0.79

----------------------------------------------------------
Retrieved 3 documents in 0.12 seconds
```

## Interactive Mode Example

```bash
$ oboyu query --interactive --use-reranker

🔍 Oboyu Interactive Search
📊 Mode: hybrid | Top-K: 5 | Vector: 0.7 | BM25: 0.3 | Reranker: enabled

⚡ Loading embedding model (cl-nagoya/ruri-v3-30m)...
⚡ Loading reranker model (cl-nagoya/ruri-v3-reranker-310m)...
⚡ Initializing database connection...
✅ Ready for search!

Type your search query (or 'help' for commands, 'exit' to quit):

> システムの設計原則について教えてください
🔍 Searching...
📊 Found 3 results in 0.12 seconds

• システム設計の基本原則 (Score: 0.91)
  当システムの設計は「シンプルさ」「モジュール性」「拡張性」の...
  Source: /projects/docs/設計/principles.md

• アーキテクチャ概要 (Score: 0.85)
  設計原則としては、「関心の分離」を徹底し...
  Source: /projects/docs/アーキテクチャ/overview.md

• プロジェクト概要 (Score: 0.79)
  本システムは、設計の段階から「ユーザー中心設計」の原則を...
  Source: /projects/README.md

> mode vector
✅ Search mode changed to: vector

> machine learning algorithms
🔍 Searching...
📊 Found 5 results in 0.08 seconds

• Introduction to Machine Learning (Score: 0.93)
  Machine learning algorithms can be broadly categorized into...
  Source: /docs/ml/intro.md

• Deep Learning Fundamentals (Score: 0.87)
  Deep learning algorithms are a subset of machine learning...
  Source: /docs/ml/deep_learning.md

[Additional results...]

> settings
Current settings:
- Mode: vector
- Top-K: 5
- Vector weight: 0.7
- BM25 weight: 0.3
- Reranker: enabled
- Database: ~/.oboyu/index.db

> exit
👋 Goodbye!
```

## Using Different Search Modes

```bash
# Vector search (semantic similarity)
$ oboyu query "日本語の自然言語処理" --mode vector

# BM25 search (keyword matching)
$ oboyu query "日本語の自然言語処理" --mode bm25

# Hybrid search (combines vector and BM25)
$ oboyu query "日本語の自然言語処理" --mode hybrid --vector-weight 0.8 --bm25-weight 0.2

# With reranking for improved relevance
$ oboyu query "日本語の自然言語処理" --rerank
```
