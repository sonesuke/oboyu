#!/usr/bin/env python3
"""Generate test data for Oboyu benchmarks."""

import argparse
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

from rich.console import Console
from rich.progress import track

from bench.config import DATA_DIR, DATASET_SIZES, get_dataset_config
from bench.utils import ensure_directory, print_header, print_metric, print_section

console = Console()

# Sample content templates for different file types
CONTENT_TEMPLATES = {
    ".txt": {
        "japanese": [
            """本日は{date}です。お天気は{weather}で、気温は{temp}度です。
{topic}について説明します。

{content}

まとめ：
{summary}
""",
            """重要なお知らせ

日時：{date}
件名：{subject}

詳細：
{content}

以上、よろしくお願いいたします。
""",
        ],
        "english": [
            """Date: {date}
Weather: {weather}, Temperature: {temp}°C

Topic: {topic}

{content}

Summary:
{summary}
""",
            """Important Notice

Date: {date}
Subject: {subject}

Details:
{content}

Thank you for your attention.
""",
        ],
    },
    ".md": {
        "japanese": [
            """# {title}

## 概要
{overview}

## 詳細
{content}

## 参考資料
- 資料1: {ref1}
- 資料2: {ref2}

最終更新日: {date}
""",
            """# {project}プロジェクト

## 背景
{background}

## 目的
{purpose}

## 実装詳細
{content}

## 今後の予定
{future}
""",
        ],
        "english": [
            """# {title}

## Overview
{overview}

## Details
{content}

## References
- Reference 1: {ref1}
- Reference 2: {ref2}

Last updated: {date}
""",
            """# {project} Project

## Background
{background}

## Purpose
{purpose}

## Implementation Details
{content}

## Future Plans
{future}
""",
        ],
    },
    ".py": {
        "code": [
            '''"""Module for {purpose}."""

import random
from typing import List, Dict, Optional

class {classname}:
    """Class for {description}."""
    
    def __init__(self, name: str, value: int = 0):
        """Initialize {classname}."""
        self.name = name
        self.value = value
        self._data: Dict[str, any] = {{}}
    
    def process(self, items: List[str]) -> Optional[Dict[str, int]]:
        """Process items and return results."""
        results = {{}}
        for item in items:
            # Process each item
            key = item.lower().strip()
            results[key] = random.randint(1, 100)
        
        return results if results else None

def main():
    """Main function."""
    instance = {classname}("test", 42)
    test_items = ["item1", "item2", "item3"]
    result = instance.process(test_items)
    print(f"Results: {{result}}")

if __name__ == "__main__":
    main()
''',
        ],
    },
    ".html": {
        "web": [
            """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    <h1>{heading}</h1>
    <p>{content}</p>
    <ul>
        <li>{item1}</li>
        <li>{item2}</li>
        <li>{item3}</li>
    </ul>
    <footer>
        <p>作成日: {date}</p>
    </footer>
</body>
</html>""",
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    <h1>{heading}</h1>
    <p>{content}</p>
    <div class="info">
        <h2>Information</h2>
        <p>{info}</p>
    </div>
    <footer>
        <p>Created: {date}</p>
    </footer>
</body>
</html>""",
        ],
    },
}

# Sample data for filling templates
SAMPLE_DATA = {
    "topics": {
        "japanese": [
            "人工知能の最新動向", "量子コンピューティング", "持続可能な開発",
            "デジタルトランスフォーメーション", "ブロックチェーン技術",
            "機械学習アルゴリズム", "自然言語処理", "コンピュータビジョン",
            "ロボティクス", "IoTセンサー技術"
        ],
        "english": [
            "Artificial Intelligence Trends", "Quantum Computing",
            "Sustainable Development", "Digital Transformation",
            "Blockchain Technology", "Machine Learning Algorithms",
            "Natural Language Processing", "Computer Vision",
            "Robotics", "IoT Sensor Technology"
        ]
    },
    "weather": {
        "japanese": ["晴れ", "曇り", "雨", "雪", "快晴"],
        "english": ["Sunny", "Cloudy", "Rainy", "Snowy", "Clear"]
    },
    "subjects": {
        "japanese": [
            "新システム導入について", "会議のお知らせ", "プロジェクト進捗報告",
            "研修プログラムのご案内", "セキュリティアップデート"
        ],
        "english": [
            "New System Implementation", "Meeting Announcement",
            "Project Progress Report", "Training Program Notice",
            "Security Update"
        ]
    },
    "projects": {
        "japanese": ["次世代AI", "スマートシティ", "グリーンエネルギー"],
        "english": ["Next-Gen AI", "Smart City", "Green Energy"]
    }
}


def generate_content(min_chars: int, max_chars: int, language: str = "mixed") -> str:
    """Generate random content of specified length."""
    if language == "japanese":
        chars = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
        particles = ["は", "が", "を", "に", "で", "と", "の", "から", "まで", "より"]
        endings = ["です。", "ます。", "でした。", "ました。", "でしょう。"]
    else:
        chars = "abcdefghijklmnopqrstuvwxyz"
        particles = [" the ", " and ", " or ", " with ", " from ", " to ", " in ", " on ", " at "]
        endings = [". ", "! ", "? "]
    
    content = []
    current_length = 0
    target_length = random.randint(min_chars, max_chars)
    
    while current_length < target_length:
        # Generate a sentence
        random.randint(10, 50)
        words = []
        
        for _ in range(random.randint(5, 15)):
            word_length = random.randint(2, 8)
            word = "".join(random.choice(chars) for _ in range(word_length))
            words.append(word)
            
            if random.random() < 0.3 and len(words) > 1:
                words.append(random.choice(particles))
        
        sentence = "".join(words) + random.choice(endings)
        content.append(sentence)
        current_length += len(sentence)
        
        # Add paragraph break occasionally
        if random.random() < 0.2:
            content.append("\n\n")
    
    return "".join(content)[:target_length]


def generate_file_content(file_type: str, size_range: Tuple[int, int]) -> str:
    """Generate content for a specific file type."""
    templates = CONTENT_TEMPLATES.get(file_type, {})
    
    if file_type == ".py":
        template = random.choice(templates.get("code", [""]))
        return template.format(
            purpose=random.choice(["data processing", "web scraping", "analysis", "automation"]),
            classname=f"TestClass{random.randint(100, 999)}",
            description="testing benchmark performance"
        )
    
    elif file_type == ".html":
        template = random.choice(templates.get("web", [""]))
        lang = random.choice(["japanese", "english"])
        return template.format(
            title=random.choice(SAMPLE_DATA["topics"][lang]),
            heading=random.choice(SAMPLE_DATA["subjects"][lang]),
            content=generate_content(100, 500, lang),
            item1=random.choice(SAMPLE_DATA["topics"][lang]),
            item2=random.choice(SAMPLE_DATA["topics"][lang]),
            item3=random.choice(SAMPLE_DATA["topics"][lang]),
            info=generate_content(50, 200, lang),
            date=datetime.now().strftime("%Y-%m-%d")
        )
    
    else:  # .txt or .md
        lang = random.choice(["japanese", "english"])
        templates_list = templates.get(lang, [])
        if not templates_list:
            return generate_content(*size_range)
        
        template = random.choice(templates_list)
        base_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        return template.format(
            date=base_date.strftime("%Y年%m月%d日" if lang == "japanese" else "%Y-%m-%d"),
            weather=random.choice(SAMPLE_DATA["weather"][lang]),
            temp=random.randint(10, 35),
            topic=random.choice(SAMPLE_DATA["topics"][lang]),
            subject=random.choice(SAMPLE_DATA["subjects"][lang]),
            title=random.choice(SAMPLE_DATA["topics"][lang]),
            project=random.choice(SAMPLE_DATA["projects"][lang]),
            content=generate_content(*size_range, lang),
            summary=generate_content(50, 150, lang),
            overview=generate_content(100, 300, lang),
            background=generate_content(100, 300, lang),
            purpose=generate_content(50, 150, lang),
            future=generate_content(100, 200, lang),
            ref1=f"https://example.com/ref{random.randint(1, 100)}",
            ref2=f"https://example.com/doc{random.randint(1, 100)}"
        )


def generate_dataset(size: str, output_dir: Path, clean: bool = True) -> Dict[str, int]:
    """Generate a dataset of the specified size."""
    config = get_dataset_config(size)
    
    # Clean existing directory if requested
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    
    ensure_directory(output_dir)
    
    print_section(f"Generating {config['name']}")
    
    stats = {
        "total_files": 0,
        "total_size": 0,
        "files_by_type": {}
    }
    
    # Generate files for each type
    for file_type, count in config["files_per_type"].items():
        type_dir = output_dir / file_type.strip(".")
        ensure_directory(type_dir)
        
        for i in track(range(count), description=f"Creating {file_type} files"):
            filename = f"test_{size}_{i:05d}{file_type}"
            filepath = type_dir / filename
            
            content = generate_file_content(file_type, config["content_size_range"])
            filepath.write_text(content, encoding="utf-8")
            
            stats["total_files"] += 1
            stats["total_size"] += len(content.encode("utf-8"))
            stats["files_by_type"][file_type] = stats["files_by_type"].get(file_type, 0) + 1
    
    return stats


def main() -> None:
    """Generate test data for benchmarks."""
    parser = argparse.ArgumentParser(description="Generate test data for Oboyu benchmarks")
    parser.add_argument(
        "sizes",
        nargs="+",
        choices=list(DATASET_SIZES.keys()) + ["all"],
        help="Dataset sizes to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean existing data before generating"
    )
    
    args = parser.parse_args()
    
    print_header("Oboyu Benchmark Data Generator")
    
    # Determine which sizes to generate
    if "all" in args.sizes:
        sizes = list(DATASET_SIZES.keys())
    else:
        sizes = args.sizes
    
    # Generate each dataset
    for size in sizes:
        dataset_dir = args.output_dir / size
        stats = generate_dataset(size, dataset_dir, clean=not args.no_clean)
        
        print_metric("Total files", stats["total_files"])
        print_metric("Total size", stats["total_size"], "size")
        
        console.print("\nFiles by type:")
        for file_type, count in stats["files_by_type"].items():
            print_metric(f"  {file_type}", count, "files")
        
        console.print(f"\n✓ Dataset saved to: {dataset_dir}")
    
    console.print("\n[green]✨ Test data generation complete![/green]")


if __name__ == "__main__":
    main()
