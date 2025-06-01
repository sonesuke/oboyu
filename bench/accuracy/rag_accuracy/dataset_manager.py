"""Dataset Manager for Japanese RAG Evaluation.

This module handles loading and managing Japanese datasets for RAG system evaluation,
including JMTEB retrieval tasks and custom datasets.
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from bench.logger import BenchmarkLogger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from bench.logger import BenchmarkLogger


@dataclass
class Document:
    """Represents a document in the dataset."""

    doc_id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Query:
    """Represents a query in the dataset."""

    query_id: str
    text: str
    relevant_docs: List[str]  # List of relevant document IDs
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Dataset:
    """Represents a complete dataset for RAG evaluation."""

    name: str
    documents: List[Document]
    queries: List[Query]
    language: str = "ja"
    description: str = ""


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self) -> Dataset:
        """Load the dataset.

        Returns:
            Loaded dataset

        """
        pass


class SyntheticJapaneseDatasetLoader(DatasetLoader):
    """Loader for synthetic Japanese datasets for testing."""

    def __init__(self, num_docs: int = 100, num_queries: int = 20, seed: int = 42) -> None:
        """Initialize synthetic dataset loader.

        Args:
            num_docs: Number of documents to generate
            num_queries: Number of queries to generate
            seed: Random seed for reproducibility

        """
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.seed = seed
        random.seed(seed)

    def load(self) -> Dataset:
        """Generate synthetic Japanese dataset.

        Returns:
            Synthetic dataset for testing

        """
        documents = self._generate_documents()
        queries = self._generate_queries(documents)

        return Dataset(
            name="synthetic_japanese",
            documents=documents,
            queries=queries,
            language="ja",
            description="Synthetic Japanese dataset for RAG evaluation testing",
        )

    def _generate_documents(self) -> List[Document]:
        """Generate synthetic Japanese documents."""
        topics = [
            ("機械学習", "機械学習は人工知能の一分野で、コンピュータがデータから学習することを可能にします。"),
            ("自然言語処理", "自然言語処理は、人間の言語をコンピュータで処理する技術です。"),
            ("深層学習", "深層学習はニューラルネットワークを使用した機械学習の手法です。"),
            ("データサイエンス", "データサイエンスは、データから価値のある洞察を抽出する学問分野です。"),
            ("検索システム", "検索システムは、大量の情報から関連性の高い情報を見つけるための技術です。"),
            ("日本語処理", "日本語処理には、形態素解析、係り受け解析、意味解析などが含まれます。"),
            ("情報検索", "情報検索は、ユーザーのクエリに基づいて関連文書を取得する技術です。"),
            ("ベクトル検索", "ベクトル検索は、文書やクエリをベクトル化して類似度を計算する手法です。"),
            ("RAGシステム", "RAGシステムは、検索と生成を組み合わせたAIシステムです。"),
            ("埋め込みモデル", "埋め込みモデルは、テキストを密なベクトル表現に変換します。"),
        ]

        documents = []
        for i in range(self.num_docs):
            topic_idx = i % len(topics)
            title, base_content = topics[topic_idx]

            # Add variation to content
            variations = [
                f"これは{title}に関する文書{i+1}です。",
                f"{base_content}",
                f"文書ID: doc_{i+1:04d}",
                f"詳細な説明: {title}の応用例や実装方法について説明します。",
            ]

            content = "\n".join(variations)

            documents.append(
                Document(
                    doc_id=f"doc_{i+1:04d}",
                    title=f"{title} - 文書{i+1}",
                    content=content,
                    metadata={"topic": title, "index": i},
                )
            )

        return documents

    def _generate_queries(self, documents: List[Document]) -> List[Query]:
        """Generate queries with relevant documents."""
        query_templates = [
            "{topic}について教えてください",
            "{topic}の基本的な概念は何ですか",
            "{topic}の応用例を説明してください",
            "{topic}と{other_topic}の違いは何ですか",
            "{topic}の実装方法について",
        ]

        queries = []
        topics = list(set(doc.metadata["topic"] for doc in documents if doc.metadata))

        for i in range(self.num_queries):
            template = random.choice(query_templates)  # noqa: S311
            topic = random.choice(topics)  # noqa: S311

            if "{other_topic}" in template:
                other_topic = random.choice([t for t in topics if t != topic])  # noqa: S311
                query_text = template.format(topic=topic, other_topic=other_topic)
                # Find relevant docs for both topics
                relevant_docs = [
                    doc.doc_id
                    for doc in documents
                    if doc.metadata and doc.metadata["topic"] in [topic, other_topic]
                ]
            else:
                query_text = template.format(topic=topic)
                # Find relevant docs for the topic
                relevant_docs = [doc.doc_id for doc in documents if doc.metadata and doc.metadata["topic"] == topic]

            # Limit relevant docs
            relevant_docs = relevant_docs[:5]

            queries.append(
                Query(
                    query_id=f"query_{i+1:03d}",
                    text=query_text,
                    relevant_docs=relevant_docs,
                    metadata={"topic": topic},
                )
            )

        return queries


class JMTEBDatasetLoader(DatasetLoader):
    """Loader for JMTEB (Japanese Massive Text Embedding Benchmark) datasets."""

    SUPPORTED_DATASETS = {
        "miracl-ja": "MIRACL Japanese retrieval dataset",
        "mldr-ja": "Multilingual Long Document Retrieval Japanese dataset",
        "jagovfaqs-22k": "Japanese Government FAQs dataset",
        "jacwir": "Japanese Casual Web Information Retrieval dataset",
    }

    def __init__(self, dataset_name: str, cache_dir: Optional[Path] = None) -> None:
        """Initialize JMTEB dataset loader.

        Args:
            dataset_name: Name of the JMTEB dataset to load
            cache_dir: Directory to cache downloaded datasets

        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {list(self.SUPPORTED_DATASETS.keys())}")

        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or Path.home() / ".cache" / "oboyu" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dataset:
        """Load the JMTEB dataset.

        Returns:
            Loaded dataset

        Note:
            This is a placeholder implementation. In a real implementation,
            this would download and load actual JMTEB datasets using the
            datasets library or similar.

        """
        # Load from HuggingFace datasets
        try:
            import datasets
        except ImportError:
            raise ImportError("Please install datasets library: pip install datasets")
        
        # Create synthetic data for now (until actual datasets are available)
        # In production, this would load actual JMTEB datasets
        return self._create_jmteb_like_dataset()
    
    def _create_jmteb_like_dataset(self) -> Dataset:
        """Create a JMTEB-like dataset with Japanese content.
        
        Returns:
            Dataset mimicking the structure of JMTEB datasets

        """
        if self.dataset_name == "miracl-ja":
            return self._create_miracl_ja_like()
        elif self.dataset_name == "mldr-ja":
            return self._create_mldr_ja_like()
        elif self.dataset_name == "jagovfaqs-22k":
            return self._create_jagovfaqs_like()
        elif self.dataset_name == "jacwir":
            return self._create_jacwir_like()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _create_miracl_ja_like(self) -> Dataset:
        """Create MIRACL-ja like dataset."""
        # Simulate multilingual IR dataset with Japanese focus
        topics = [
            "人工知能", "機械学習", "深層学習", "自然言語処理", "コンピュータビジョン",
            "ロボティクス", "データサイエンス", "量子コンピューティング", "ブロックチェーン", "IoT"
        ]
        
        documents = []
        for i in range(100):  # Reduced from 150 to 100
            topic = topics[i % len(topics)]
            doc_id = f"miracl_doc_{i+1:05d}"
            title = f"{topic}に関する研究論文 {i+1}"
            content = f"""
{topic}は現代の技術革新において重要な役割を果たしています。
本論文では、{topic}の最新の研究動向と応用について詳しく説明します。
特に、日本における{topic}の発展と、国際的な研究コミュニティへの貢献について論じます。

主な内容：
1. {topic}の基本原理と理論的背景
2. 最新の研究成果と技術的進歩
3. 実世界での応用例と成功事例
4. 今後の課題と研究の方向性

この分野の研究は急速に進化しており、新しい発見が次々と報告されています。
"""
            documents.append(Document(
                doc_id=doc_id,
                title=title,
                content=content,
                metadata={"topic": topic, "language": "ja", "source": "academic"}
            ))
        
        # Create queries
        queries = []
        query_templates = [
            "{topic}の最新研究",
            "{topic}の日本での応用",
            "{topic}の基本原理",
            "{topic}と{other_topic}の関係",
            "{topic}の将来展望"
        ]
        
        for i in range(30):
            template = query_templates[i % len(query_templates)]
            topic = topics[i % len(topics)]
            
            if "{other_topic}" in template:
                other_topic = topics[(i + 1) % len(topics)]
                query_text = template.format(topic=topic, other_topic=other_topic)
                relevant_docs = [
                    doc.doc_id for doc in documents
                    if doc.metadata and (doc.metadata["topic"] == topic or doc.metadata["topic"] == other_topic)
                ][:5]
            else:
                query_text = template.format(topic=topic)
                relevant_docs = [
                    doc.doc_id for doc in documents
                    if doc.metadata and doc.metadata["topic"] == topic
                ][:5]
            
            queries.append(Query(
                query_id=f"miracl_q_{i+1:03d}",
                text=query_text,
                relevant_docs=relevant_docs,
                metadata={"topic": topic}
            ))
        
        return Dataset(
            name="miracl-ja",
            documents=documents,
            queries=queries,
            language="ja",
            description=f"MIRACL Japanese - Multilingual Information Retrieval (Synthetic, {len(documents)} docs)"
        )
    
    def _create_mldr_ja_like(self) -> Dataset:
        """Create MLDR-ja like dataset with long documents."""
        # Simulate long document retrieval
        topics = [
            "日本の歴史", "経済政策", "環境問題", "教育システム", "医療技術",
            "文化遺産", "科学技術", "国際関係", "社会保障", "エネルギー政策"
        ]
        
        documents = []
        for i in range(50):  # Reduced from 100 to 50 for faster processing
            topic = topics[i % len(topics)]
            doc_id = f"mldr_doc_{i+1:05d}"
            title = f"{topic}に関する詳細報告書"
            
            # Create longer content (simulating long documents)
            sections = []
            for j in range(3):  # Reduced from 5 to 3 sections
                sections.append(f"""
第{j+1}章：{topic}の{['概要', '現状分析', '将来展望'][j]}

{topic}について、本章では詳細な分析を行います。
日本における{topic}の重要性は年々高まっており、
様々な取り組みが進められています。

主なポイント：
- {topic}に関する最新動向
- 実施されている施策
- 今後の展望
""")
            
            content = "\n\n".join(sections)
            
            documents.append(Document(
                doc_id=doc_id,
                title=title,
                content=content,
                metadata={"topic": topic, "language": "ja", "doc_type": "report", "length": "long"}
            ))
        
        # Create queries for long documents
        queries = []
        for i in range(25):
            topic = topics[i % len(topics)]
            query_types = [
                f"{topic}の現状と課題",
                f"{topic}に関する政策提言",
                f"{topic}の国際比較",
                f"{topic}の将来予測",
                f"{topic}の経済的影響"
            ]
            
            query_text = query_types[i % len(query_types)]
            relevant_docs = [
                doc.doc_id for doc in documents
                if doc.metadata and doc.metadata["topic"] == topic
            ][:3]  # Fewer relevant docs for long document retrieval
            
            queries.append(Query(
                query_id=f"mldr_q_{i+1:03d}",
                text=query_text,
                relevant_docs=relevant_docs,
                metadata={"topic": topic, "query_type": "analytical"}
            ))
        
        return Dataset(
            name="mldr-ja",
            documents=documents,
            queries=queries,
            language="ja",
            description="Multilingual Long Document Retrieval - Japanese (Synthetic)"
        )
    
    def _create_jagovfaqs_like(self) -> Dataset:
        """Create JaGovFaqs-like dataset."""
        # Simulate government FAQ dataset
        categories = [
            "税金", "社会保険", "教育", "医療", "年金",
            "住民登録", "パスポート", "運転免許", "子育て支援", "労働"
        ]
        
        documents = []
        for i in range(100):  # Reduced from 200 to 100
            category = categories[i % len(categories)]
            doc_id = f"govfaq_doc_{i+1:05d}"
            
            faq_templates = [
                (f"{category}の手続きについて", f"{category}に関する手続きは、市役所または区役所で行うことができます。必要な書類は身分証明書と申請書です。"),
                (f"{category}の申請方法", f"{category}の申請は、オンラインまたは窓口で可能です。詳細は政府のウェブサイトをご確認ください。"),
                (f"{category}の必要書類", f"{category}の手続きには、本人確認書類、住民票、印鑑が必要です。"),
                (f"{category}の費用", f"{category}に関する費用は、ケースによって異なります。詳細は担当窓口にお問い合わせください。"),
                (f"{category}の期限", f"{category}の手続きには期限があります。早めの申請をお勧めします。")
            ]
            
            title, content = faq_templates[i % len(faq_templates)]
            
            # Expand content
            content += f"""

詳細情報：
- 受付時間：平日9:00-17:00
- 問い合わせ先：{category}担当課
- 必要な時間：約30分から1時間
- オンライン申請：可能（要マイナンバーカード）

よくある質問：
Q: {category}の手続きは代理人でも可能ですか？
A: 委任状があれば代理人による手続きも可能です。

Q: {category}の申請はいつまでに行う必要がありますか？
A: 原則として、事由が発生してから14日以内に申請してください。

注意事項：
- 書類に不備がある場合は、再度お越しいただく必要があります
- 混雑時は待ち時間が長くなる場合があります
- 詳細は各自治体のホームページをご確認ください
"""
            
            documents.append(Document(
                doc_id=doc_id,
                title=title,
                content=content,
                metadata={"category": category, "type": "faq", "source": "government"}
            ))
        
        # Create queries
        queries = []
        for i in range(40):
            category = categories[i % len(categories)]
            query_templates = [
                f"{category}の手続き方法を教えてください",
                f"{category}に必要な書類は何ですか",
                f"{category}の申請期限はいつまでですか",
                f"{category}の費用はいくらかかりますか",
                f"{category}はオンラインで申請できますか"
            ]
            
            query_text = query_templates[i % len(query_templates)]
            relevant_docs = [
                doc.doc_id for doc in documents
                if doc.metadata and doc.metadata["category"] == category
            ][:5]
            
            queries.append(Query(
                query_id=f"govfaq_q_{i+1:03d}",
                text=query_text,
                relevant_docs=relevant_docs,
                metadata={"category": category, "type": "faq"}
            ))
        
        return Dataset(
            name="jagovfaqs-22k",
            documents=documents,
            queries=queries,
            language="ja",
            description="Japanese Government FAQs Dataset (Synthetic)"
        )
    
    def _create_jacwir_like(self) -> Dataset:
        """Create JaCWIR-like dataset."""
        # Simulate casual web information retrieval
        topics = [
            "料理レシピ", "旅行", "健康", "美容", "ファッション",
            "スポーツ", "エンターテインメント", "テクノロジー", "ビジネス", "ライフスタイル"
        ]
        
        documents = []
        for i in range(80):  # Reduced from 180 to 80
            topic = topics[i % len(topics)]
            doc_id = f"jacwir_doc_{i+1:05d}"
            
            # Create casual web content
            content_templates = [
                f"今日は{topic}について書いてみたいと思います。最近{topic}にハマっていて、いろいろ調べてみました。",
                f"{topic}初心者の私が、実際に試してみた感想をシェアします。意外と簡単でした！",
                f"{topic}のプロが教える、知っておきたい5つのポイント。これを知れば、あなたも{topic}マスター！",
                f"【保存版】{topic}の基本から応用まで。この記事を読めば、{topic}の全てがわかります。",
                f"{topic}で失敗しないために。私の経験から学んだ、大切なことをお伝えします。"
            ]
            
            title = content_templates[i % len(content_templates)]
            
            content = f"""
{title}

みなさん、こんにちは！今回は{topic}についてお話しします。

{topic}って、意外と奥が深いんですよね。
私も最初は全然わからなかったんですが、
少しずつ勉強していくうちに、だんだん楽しくなってきました。

特に重要だと思ったポイントは：

1. 基本をしっかり理解すること
2. 実践を重ねること
3. 他の人の経験から学ぶこと

{topic}に興味がある方は、ぜひ参考にしてみてください。
質問があれば、コメント欄で聞いてくださいね！

最後まで読んでいただき、ありがとうございました。
次回も{topic}に関する情報をお届けする予定です。
お楽しみに！

##{topic} #日本 #初心者向け #わかりやすい
"""
            
            documents.append(Document(
                doc_id=doc_id,
                title=title,
                content=content,
                metadata={"topic": topic, "style": "casual", "platform": "blog"}
            ))
        
        # Create casual queries
        queries = []
        for i in range(35):
            topic = topics[i % len(topics)]
            query_templates = [
                f"{topic} 初心者",
                f"{topic} おすすめ",
                f"{topic} 簡単",
                f"{topic} コツ",
                f"{topic} 方法"
            ]
            
            query_text = query_templates[i % len(query_templates)]
            relevant_docs = [
                doc.doc_id for doc in documents
                if doc.metadata and doc.metadata["topic"] == topic
            ][:6]
            
            queries.append(Query(
                query_id=f"jacwir_q_{i+1:03d}",
                text=query_text,
                relevant_docs=relevant_docs,
                metadata={"topic": topic, "style": "casual"}
            ))
        
        return Dataset(
            name="jacwir",
            documents=documents,
            queries=queries,
            language="ja",
            description="Japanese Casual Web Information Retrieval (Synthetic)"
        )


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom datasets from JSON files."""

    def __init__(self, dataset_path: Path) -> None:
        """Initialize custom dataset loader.

        Args:
            dataset_path: Path to the dataset JSON file

        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    def load(self) -> Dataset:
        """Load custom dataset from JSON.

        Expected JSON format:
        {
            "name": "dataset_name",
            "language": "ja",
            "description": "Dataset description",
            "documents": [
                {
                    "doc_id": "doc1",
                    "title": "Document Title",
                    "content": "Document content",
                    "metadata": {}
                }
            ],
            "queries": [
                {
                    "query_id": "q1",
                    "text": "Query text",
                    "relevant_docs": ["doc1", "doc2"],
                    "metadata": {}
                }
            ]
        }

        Returns:
            Loaded dataset

        """
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = [
            Document(
                doc_id=doc["doc_id"],
                title=doc["title"],
                content=doc["content"],
                metadata=doc.get("metadata"),
            )
            for doc in data["documents"]
        ]

        queries = [
            Query(
                query_id=query["query_id"],
                text=query["text"],
                relevant_docs=query["relevant_docs"],
                metadata=query.get("metadata"),
            )
            for query in data["queries"]
        ]

        return Dataset(
            name=data.get("name", "custom_dataset"),
            documents=documents,
            queries=queries,
            language=data.get("language", "ja"),
            description=data.get("description", ""),
        )


class DatasetManager:
    """Manages datasets for RAG evaluation."""

    def __init__(self, logger: Optional[BenchmarkLogger] = None) -> None:
        """Initialize dataset manager.

        Args:
            logger: Optional logger for output

        """
        self.logger = logger or BenchmarkLogger()
        self._loaded_datasets: Dict[str, Dataset] = {}

    def load_dataset(self, dataset_name: str, dataset_path: Optional[Path] = None) -> Dataset:
        """Load a dataset by name or path.

        Args:
            dataset_name: Name of the dataset (jmteb, synthetic, or custom)
            dataset_path: Optional path for custom datasets

        Returns:
            Loaded dataset

        """
        # Check cache first
        cache_key = f"{dataset_name}:{dataset_path}"
        if cache_key in self._loaded_datasets:
            self.logger.info(f"Using cached dataset: {dataset_name}")
            return self._loaded_datasets[cache_key]

        self.logger.info(f"Loading dataset: {dataset_name}")

        # Load based on dataset type
        if dataset_name == "synthetic":
            loader = SyntheticJapaneseDatasetLoader()
        elif dataset_name in JMTEBDatasetLoader.SUPPORTED_DATASETS:
            loader = JMTEBDatasetLoader(dataset_name)
        elif dataset_name == "custom" and dataset_path:
            loader = CustomDatasetLoader(dataset_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset = loader.load()
        self._loaded_datasets[cache_key] = dataset

        self.logger.success(
            f"Loaded dataset '{dataset.name}' with {len(dataset.documents)} documents "
            f"and {len(dataset.queries)} queries"
        )

        return dataset

    def prepare_dataset_for_evaluation(
        self, dataset: Dataset, max_queries: Optional[int] = None, shuffle: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare dataset for RAG evaluation.

        Args:
            dataset: Dataset to prepare
            max_queries: Maximum number of queries to use
            shuffle: Whether to shuffle queries

        Returns:
            Tuple of (queries, documents) in format expected by RAGEvaluator

        """
        # Convert to evaluation format
        documents = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content": doc.content,
                "metadata": doc.metadata or {},
            }
            for doc in dataset.documents
        ]

        queries = [
            {
                "query_id": query.query_id,
                "text": query.text,
                "relevant_docs": query.relevant_docs,
                "metadata": query.metadata or {},
            }
            for query in dataset.queries
        ]

        # Shuffle if requested
        if shuffle:
            random.shuffle(queries)

        # Limit queries if requested
        if max_queries and len(queries) > max_queries:
            queries = queries[:max_queries]

        return queries, documents

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets.

        Returns:
            List of dataset names

        """
        available = ["synthetic", "custom"]
        available.extend(JMTEBDatasetLoader.SUPPORTED_DATASETS.keys())
        return available

