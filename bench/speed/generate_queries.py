#!/usr/bin/env python3
"""Generate query datasets for Oboyu benchmarks."""

import argparse
import secrets
from pathlib import Path
from typing import Dict, List

from rich.console import Console

from bench.config import QUERIES_DIR, QUERY_CONFIG
from bench.utils import ensure_directory, print_header, print_metric, print_section, save_json

console = Console()

# Query templates by language and type
QUERY_TEMPLATES = {
    "japanese": {
        "technical": [
            "{tech}の実装方法について教えてください",
            "{tech}における{concept}の役割は何ですか",
            "{tech}を使用した{task}の手順",
            "{tech}のベストプラクティス",
            "{tech}と{tech2}の違いを説明してください",
            "{concept}に関する技術文書",
            "{tech}のパフォーマンス最適化",
            "{tech}のセキュリティ考慮事項",
            "{tech}のトラブルシューティング方法",
            "{tech}の最新動向"
        ],
        "business": [
            "{company}の{year}年度事業計画",
            "{product}の市場分析レポート",
            "{department}部門の業績報告書",
            "{project}プロジェクトの進捗状況",
            "{topic}に関する経営戦略",
            "{industry}業界の動向分析",
            "{metric}の改善提案書",
            "{process}プロセスの効率化",
            "顧客満足度向上のための施策",
            "{region}市場への参入戦略"
        ],
        "general": [
            "{topic}について詳しく知りたい",
            "{topic}の基本的な説明",
            "{topic}に関する資料",
            "{topic}の歴史と発展",
            "{topic}の利点と欠点",
            "{topic}の具体例",
            "{topic}を始めるには",
            "{topic}の注意点",
            "{topic}の将来性",
            "{topic}関連の情報"
        ],
        "code": [
            "{language}で{algorithm}を実装する方法",
            "{framework}の{feature}機能の使い方",
            "{error}エラーの解決方法",
            "{language}のコーディング規約",
            "{pattern}パターンの実装例",
            "{library}ライブラリの使用方法",
            "{language}での{task}処理",
            "{tool}の設定ファイル",
            "{language}のデバッグ手法",
            "{framework}のベンチマーク結果"
        ]
    },
    "english": {
        "technical": [
            "How to implement {tech}",
            "What is the role of {concept} in {tech}",
            "Steps for {task} using {tech}",
            "Best practices for {tech}",
            "Difference between {tech} and {tech2}",
            "Technical documentation on {concept}",
            "Performance optimization for {tech}",
            "Security considerations for {tech}",
            "Troubleshooting {tech} issues",
            "Latest trends in {tech}"
        ],
        "business": [
            "{company} business plan for {year}",
            "Market analysis report for {product}",
            "Performance report for {department} department",
            "Progress update on {project} project",
            "Business strategy regarding {topic}",
            "{industry} industry trend analysis",
            "Improvement proposal for {metric}",
            "Efficiency optimization for {process} process",
            "Customer satisfaction improvement initiatives",
            "Market entry strategy for {region}"
        ],
        "general": [
            "I want to know more about {topic}",
            "Basic explanation of {topic}",
            "Resources about {topic}",
            "History and development of {topic}",
            "Advantages and disadvantages of {topic}",
            "Examples of {topic}",
            "How to get started with {topic}",
            "Important considerations for {topic}",
            "Future prospects of {topic}",
            "Information related to {topic}"
        ],
        "code": [
            "How to implement {algorithm} in {language}",
            "Using {feature} feature in {framework}",
            "Solving {error} error",
            "Coding standards for {language}",
            "Implementation example of {pattern} pattern",
            "How to use {library} library",
            "{task} processing in {language}",
            "Configuration file for {tool}",
            "Debugging techniques in {language}",
            "Benchmark results for {framework}"
        ]
    },
    "mixed": {
        "technical": [
            "{tech}のarchitectureについて",
            "How to integrate {tech} with {tech2}システム",
            "{concept}のimplementationガイド",
            "Performance tuningのベストプラクティス for {tech}",
            "{tech}におけるsecurity best practices",
        ],
        "general": [
            "{topic}のoverview資料",
            "Getting started with {topic}ガイド",
            "{topic}に関するFAQ document",
            "Troubleshooting guide for {topic}関連issues",
            "{topic}のroadmapとfuture plans"
        ]
    }
}

# Sample data for filling templates
TEMPLATE_DATA = {
    "tech": [
        "Docker", "Kubernetes", "機械学習", "ブロックチェーン", "React",
        "データベース", "マイクロサービス", "API", "クラウド", "AI",
        "IoT", "ビッグデータ", "DevOps", "CI/CD", "セキュリティ"
    ],
    "tech2": [
        "AWS", "Azure", "GCP", "オンプレミス", "レガシーシステム",
        "モノリス", "サーバーレス", "コンテナ", "仮想マシン"
    ],
    "concept": [
        "スケーラビリティ", "可用性", "冗長性", "レジリエンス",
        "パフォーマンス", "セキュリティ", "監視", "ログ管理",
        "自動化", "最適化"
    ],
    "task": [
        "デプロイメント", "モニタリング", "バックアップ", "リストア",
        "スケーリング", "マイグレーション", "インテグレーション",
        "テスト", "デバッグ", "最適化"
    ],
    "company": ["テック株式会社", "イノベーション社", "デジタル企業", "Tech Corp", "Innovation Inc"],
    "year": ["2024", "2025", "2023"],
    "product": ["新製品A", "サービスB", "プラットフォームC", "Product X", "Service Y"],
    "department": ["開発", "営業", "マーケティング", "人事", "財務"],
    "project": ["デジタル変革", "新システム導入", "業務改善", "DX推進", "Innovation"],
    "topic": [
        "人工知能", "持続可能性", "リモートワーク", "デジタルマーケティング",
        "ブランディング", "イノベーション", "グローバル展開", "ESG",
        "カスタマーエクスペリエンス", "アジャイル開発"
    ],
    "industry": ["IT", "製造業", "金融", "ヘルスケア", "小売"],
    "metric": ["売上", "利益率", "顧客満足度", "生産性", "品質"],
    "process": ["開発", "製造", "販売", "カスタマーサポート", "品質管理"],
    "region": ["アジア", "北米", "ヨーロッパ", "日本", "中国"],
    "language": ["Python", "Java", "JavaScript", "Go", "Rust", "TypeScript"],
    "algorithm": ["ソート", "検索", "グラフ探索", "動的計画法", "機械学習"],
    "framework": ["Django", "React", "Vue.js", "Spring", "Express"],
    "feature": ["認証", "キャッシング", "ルーティング", "ミドルウェア", "ORM"],
    "error": ["NullPointer", "Timeout", "Memory Leak", "Permission Denied", "Connection Refused"],
    "pattern": ["Singleton", "Factory", "Observer", "MVC", "Repository"],
    "library": ["NumPy", "Pandas", "TensorFlow", "jQuery", "Lodash"],
    "tool": ["Git", "Docker", "Jenkins", "Webpack", "ESLint"]
}


def generate_query(template: str) -> str:
    """Generate a query from a template by filling in placeholders."""
    query = template
    
    # Find all placeholders in the template
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    # Replace each placeholder with random data
    for placeholder in placeholders:
        if placeholder in TEMPLATE_DATA:
            value = secrets.choice(TEMPLATE_DATA[placeholder])
            query = query.replace(f"{{{placeholder}}}", value)
    
    return query


def generate_query_set(
    language: str,
    query_type: str,
    count: int
) -> List[Dict[str, str]]:
    """Generate a set of queries for a specific language and type."""
    queries = []
    templates = QUERY_TEMPLATES.get(language, {}).get(query_type, [])
    
    if not templates:
        console.print(f"[yellow]Warning: No templates for {language}/{query_type}[/yellow]")
        return queries
    
    for i in range(count):
        template = secrets.choice(templates)
        query_text = generate_query(template)
        
        queries.append({
            "id": f"{language}_{query_type}_{i:03d}",
            "text": query_text,
            "language": language,
            "type": query_type,
            "metadata": {
                "template": template,
                "generated": True
            }
        })
    
    return queries


def generate_all_queries() -> Dict[str, List[Dict[str, str]]]:
    """Generate all query datasets based on configuration."""
    all_queries = {}
    
    for language, config in QUERY_CONFIG.items():
        print_section(f"Generating {language.capitalize()} queries")
        
        language_queries = []
        
        for query_type, type_count in config["types"].items():
            queries = generate_query_set(language, query_type, type_count)
            language_queries.extend(queries)
            print_metric(f"  {query_type}", len(queries), "queries")
        
        all_queries[language] = language_queries
        print_metric("Total", len(language_queries), "queries")
    
    return all_queries


def save_queries(queries: Dict[str, List[Dict[str, str]]], output_dir: Path) -> None:
    """Save query datasets to files."""
    ensure_directory(output_dir)
    
    for language, query_list in queries.items():
        # Save as JSON
        json_file = output_dir / f"{language}_queries.json"
        save_json(query_list, json_file)
        
        # Also save as plain text for easy viewing
        txt_file = output_dir / f"{language}_queries.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"# {language.capitalize()} Queries\n")
            f.write(f"# Total: {len(query_list)} queries\n\n")
            
            for query in query_list:
                f.write(f"[{query['id']}] ({query['type']})\n")
                f.write(f"{query['text']}\n\n")
        
        console.print(f"✓ Saved {language} queries to {json_file.name} and {txt_file.name}")


def main() -> None:
    """Generate query datasets."""
    parser = argparse.ArgumentParser(description="Generate query datasets for Oboyu benchmarks")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=QUERIES_DIR,
        help="Output directory for query datasets"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["japanese", "english", "mixed", "all"],
        default=["all"],
        help="Languages to generate queries for"
    )
    
    args = parser.parse_args()
    
    print_header("Oboyu Query Generator")
    
    # Generate queries
    all_queries = generate_all_queries()
    
    # Filter by requested languages
    if "all" not in args.languages:
        all_queries = {
            lang: queries
            for lang, queries in all_queries.items()
            if lang in args.languages
        }
    
    # Save queries
    print_section("Saving query datasets")
    save_queries(all_queries, args.output_dir)
    
    # Print summary
    print_section("Summary")
    total_queries = sum(len(queries) for queries in all_queries.values())
    print_metric("Total queries generated", total_queries)
    print_metric("Output directory", str(args.output_dir))
    
    console.print("\n[green]✨ Query generation complete![/green]")


if __name__ == "__main__":
    main()
