"""Japanese LLM implementation for KG extraction.

This module provides a concrete implementation of the KG extraction service
using Japanese language models for local CPU-based inference.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import instructor
import jaconv
from llama_cpp import Llama
from pydantic import BaseModel, Field

from oboyu.common.paths import CACHE_BASE_DIR
from oboyu.domain.models.knowledge_graph import (
    ENTITY_TYPES,
    RELATION_TYPES,
    Entity,
    KnowledgeGraphExtraction,
    Relation,
)
from oboyu.ports.services.kg_extraction_service import ExtractionError, KGExtractionService

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """Pydantic model for extracted entity."""

    name: str = Field(description="エンティティの名前")
    entity_type: str = Field(description="エンティティのタイプ（PERSON, COMPANY, ORGANIZATION, LOCATION等）")
    definition: Optional[str] = Field(description="エンティティの定義・説明", default=None)
    confidence: float = Field(description="信頼度スコア (0.0-1.0)", default=0.8, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    """Pydantic model for extracted relation."""

    source: str = Field(description="ソースエンティティ名")
    target: str = Field(description="ターゲットエンティティ名")
    relation_type: str = Field(description="リレーションタイプ（WORKS_AT, CEO_OF, LOCATED_IN等）")
    confidence: float = Field(description="信頼度スコア (0.0-1.0)", default=0.8, ge=0.0, le=1.0)


class StructuredKGExtraction(BaseModel):
    """Pydantic model for complete knowledge graph extraction."""

    entities: List[ExtractedEntity] = Field(description="抽出されたエンティティのリスト", default_factory=list)
    relations: List[ExtractedRelation] = Field(description="抽出されたリレーションのリスト", default_factory=list)


class LLMKGExtractionService(KGExtractionService):
    """Japanese LLM-based knowledge graph extraction service.

    Uses Japanese language models for local CPU inference
    with Japanese language support and business-oriented entity/relation extraction.
    Supports multiple models including TinySwallow and ELYZA variants.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        verbose: bool = False,
    ) -> None:
        """Initialize Japanese LLM KG extraction service.

        Args:
            model_path: Path to Japanese LLM GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            verbose: Enable verbose logging

        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._llm: Optional[Llama] = None
        self._model_loaded = False
        self._instructor_client = None
        self._detected_model_name: Optional[str] = None

    def _ensure_model_loaded(self) -> None:
        """Ensure the LLM model is loaded."""
        if self._llm is None:
            # Check if model_path is a HuggingFace model ID or local path
            if "/" in str(self.model_path) and not self.model_path.exists():
                # Looks like a HuggingFace model ID, try to download
                try:
                    from huggingface_hub import hf_hub_download

                    logger.info(f"Downloading Japanese LLM model from HuggingFace Hub: {self.model_path}")

                    # Set up cache directory following the same pattern as embeddings
                    kg_cache_dir = CACHE_BASE_DIR / "kg" / "models"
                    kg_cache_dir.mkdir(parents=True, exist_ok=True)

                    # Try multiple common GGUF filenames for supported models
                    possible_filenames = [
                        "tinyswallow-1.5b-instruct-q5_k_m.gguf",  # TinySwallow primary
                        "tinyswallow-1.5b-instruct-q8_0.gguf",  # TinySwallow alternative quantization
                        "Llama-3-ELYZA-JP-8B-q4_k_m.gguf",  # ELYZA model
                        "model.gguf",  # Generic model name
                        "pytorch_model.gguf",  # PyTorch export
                        f"{str(self.model_path).split('/')[-1].lower()}.gguf",
                    ]

                    model_file = None
                    for filename in possible_filenames:
                        try:
                            logger.info(f"Trying to download: {filename}")
                            model_file = hf_hub_download(repo_id=str(self.model_path), filename=filename, cache_dir=kg_cache_dir, local_files_only=False)
                            logger.info(f"Successfully downloaded: {filename}")
                            break
                        except Exception as e:
                            logger.debug(f"Failed to download {filename}: {e}")
                            continue

                    if model_file is None:
                        raise ExtractionError(
                            f"No GGUF model files found for {self.model_path}. "
                            f"Tried: {', '.join(possible_filenames)}. "
                            f"Please ensure the model repository contains a GGUF file or specify a local path."
                        )
                    self.model_path = Path(model_file)
                    logger.info(f"Model downloaded to: {self.model_path}")

                except ImportError:
                    raise ExtractionError("huggingface_hub is required to download models from HuggingFace Hub. Install with: pip install huggingface_hub")
                except Exception as e:
                    raise ExtractionError(f"Failed to download model from HuggingFace Hub: {e}")

            elif not self.model_path.exists():
                raise ExtractionError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading KG extraction model from {self.model_path}")
            try:
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=min(self.n_ctx, 2048),  # Smaller context for 1B model
                    n_threads=self.n_threads,
                    verbose=self.verbose,
                    n_gpu_layers=-1,  # Use all GPU layers for Metal acceleration
                )
                self._model_loaded = True

                # Set up instructor client for structured output
                self._instructor_client = instructor.patch(create=self._llm.create_chat_completion_openai_v1, mode=instructor.Mode.JSON_SCHEMA)

                logger.info("KG extraction model loaded successfully")
            except Exception as e:
                raise ExtractionError(f"Failed to load KG extraction model: {e}")

    def _normalize_japanese_text(self, text: str) -> str:
        """Normalize Japanese text for consistent processing."""
        # Convert full-width to half-width for ASCII characters
        text = jaconv.z2h(text, ascii=True, digit=True)
        # Normalize katakana
        text = jaconv.kata2hira(text)
        return text.strip()

    def _create_extraction_prompt(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> str:
        """Create prompt for knowledge graph extraction."""
        # Use default types if not specified
        if entity_types is None:
            entity_types = list(ENTITY_TYPES.keys())
        if relation_types is None:
            relation_types = list(RELATION_TYPES.keys())

        entity_examples = ", ".join(entity_types[:10])  # Limit for prompt size
        relation_examples = ", ".join(relation_types[:15])  # Limit for prompt size

        prompt = f"""あなたは知識グラフ抽出の専門家です。以下のテキストから構造化された知識グラフを抽出してください。

テキスト:
{text}

抽出ルール:
1. エンティティタイプ: {entity_examples}
2. リレーションタイプ: {relation_examples}
3. 明確に記載されている事実のみを抽出
4. 推測や解釈は含めない
5. 信頼度スコア(0.0-1.0)を各要素に付与

出力形式（JSON）:
{{
  "entities": [
    {{
      "name": "エンティティ名",
      "entity_type": "PERSON|COMPANY|ORGANIZATION|PRODUCT|LOCATION|EVENT|POSITION|TECHNOLOGY|CONCEPT|DATE",
      "definition": "エンティティの定義・説明",
      "confidence": 0.95
    }}
  ],
  "relations": [
    {{
      "source": "ソースエンティティ名",
      "target": "ターゲットエンティティ名",
      "relation_type": "WORKS_AT|CEO_OF|LOCATED_IN|etc",
      "confidence": 0.9
    }}
  ]
}}

JSON形式のみで回答してください:"""

        return prompt

    def _extract_structured_kg(self, text: str, chunk_id: str) -> StructuredKGExtraction:
        """Extract knowledge graph using instructor for structured output."""
        # Normalize Japanese text
        normalized_text = self._normalize_japanese_text(text)

        # Create system prompt
        system_prompt = """あなたは知識グラフ抽出の専門家です。
以下のテキストから構造化された知識グラフを抽出してください。

抽出ルール:
1. 明確に記載されている事実のみを抽出
2. 推測や解釈は含めない
3. エンティティタイプ: PERSON, COMPANY, ORGANIZATION, PRODUCT, LOCATION, EVENT, POSITION, TECHNOLOGY, CONCEPT, DATE
4. リレーションタイプ: WORKS_AT, CEO_OF, LOCATED_IN, MEMBER_OF, PARENT_COMPANY, SUBSIDIARY, PARTNER_WITH等
5. 信頼度スコア(0.0-1.0)を各要素に付与"""

        user_prompt = f"テキスト: {normalized_text}"

        # Use instructor for structured output
        extraction = self._instructor_client(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_model=StructuredKGExtraction,
            max_tokens=min(self.max_tokens, 512),
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return extraction

    def _parse_llm_response(self, response: str, chunk_id: str) -> KnowledgeGraphExtraction:
        """Parse LLM JSON response into KnowledgeGraphExtraction."""
        try:
            # Clean response - extract JSON if wrapped in other text
            response = response.strip()
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            # Find JSON object boundaries
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx]

            data = json.loads(response)

            entities = []
            relations = []

            # Parse entities
            for entity_data in data.get("entities", []):
                entity = Entity(
                    name=entity_data["name"],
                    entity_type=entity_data["entity_type"],
                    definition=entity_data.get("definition"),
                    confidence=entity_data.get("confidence", 0.8),
                    chunk_id=chunk_id,
                )
                entities.append(entity)

            # Parse relations
            entity_name_to_id = {e.name: e.id for e in entities}

            for relation_data in data.get("relations", []):
                source_name = relation_data["source"]
                target_name = relation_data["target"]

                # Find entity IDs
                source_id = entity_name_to_id.get(source_name)
                target_id = entity_name_to_id.get(target_name)

                if source_id and target_id:
                    relation = Relation(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=relation_data["relation_type"],
                        confidence=relation_data.get("confidence", 0.8),
                        chunk_id=chunk_id,
                    )
                    relations.append(relation)
                else:
                    logger.warning(f"Could not find entity IDs for relation: {source_name} -> {target_name}")

            # Calculate overall confidence
            all_confidences = [e.confidence for e in entities] + [r.confidence for r in relations]
            overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

            return KnowledgeGraphExtraction(
                chunk_id=chunk_id,
                entities=entities,
                relations=relations,
                confidence=overall_confidence,
                model_used=self._get_model_name(),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            raise ExtractionError(f"Invalid JSON response from LLM: {e}", chunk_id)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            raise ExtractionError(f"Failed to parse extraction result: {e}", chunk_id)

    async def extract_knowledge_graph(
        self,
        text: str,
        chunk_id: str,
        language: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> KnowledgeGraphExtraction:
        """Extract knowledge graph from text using Japanese LLM model."""
        start_time = time.time()

        try:
            self._ensure_model_loaded()

            logger.debug(f"Starting structured KG extraction for chunk {chunk_id}")

            # Use structured extraction with instructor
            loop = asyncio.get_event_loop()
            if self._instructor_client is None:
                raise ExtractionError("Instructor client not initialized", chunk_id)

            try:
                structured_extraction = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._extract_structured_kg(text, chunk_id)),
                    timeout=5.0 if chunk_id == "test_validation" else 30.0,  # TinySwallow 1.5B is lightweight and fast
                )
            except asyncio.TimeoutError:
                raise ExtractionError(f"Structured KG extraction timed out for chunk {chunk_id}", chunk_id)

            # Convert structured extraction to domain model
            entities = []
            relations = []

            # Convert entities
            for ent in structured_extraction.entities:
                entity = Entity(
                    name=ent.name,
                    entity_type=ent.entity_type,
                    definition=ent.definition,
                    confidence=ent.confidence,
                    chunk_id=chunk_id,
                )
                entities.append(entity)

            # Convert relations
            entity_name_to_id = {e.name: e.id for e in entities}
            for rel in structured_extraction.relations:
                source_id = entity_name_to_id.get(rel.source)
                target_id = entity_name_to_id.get(rel.target)

                if source_id and target_id:
                    relation = Relation(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=rel.relation_type,
                        confidence=rel.confidence,
                        chunk_id=chunk_id,
                    )
                    relations.append(relation)
                else:
                    logger.warning(f"Could not find entity IDs for relation: {rel.source} -> {rel.target}")

            # Calculate overall confidence
            all_confidences = [e.confidence for e in entities] + [r.confidence for r in relations]
            overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

            extraction = KnowledgeGraphExtraction(
                chunk_id=chunk_id,
                entities=entities,
                relations=relations,
                confidence=overall_confidence,
                model_used=self._get_model_name(),
            )

            # Add processing time
            processing_time = int((time.time() - start_time) * 1000)
            extraction.processing_time_ms = processing_time

            logger.info(f"Extracted {len(extraction.entities)} entities and {len(extraction.relations)} relations from chunk {chunk_id} in {processing_time}ms")

            return extraction

        except Exception as e:
            logger.error(f"KG extraction failed for chunk {chunk_id}: {e}")
            if isinstance(e, ExtractionError):
                raise
            raise ExtractionError(f"Unexpected error during extraction: {e}", chunk_id)

    async def batch_extract_knowledge_graph(
        self,
        texts_and_ids: List[tuple[str, str]],
        language: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> List[KnowledgeGraphExtraction]:
        """Extract knowledge graphs from multiple texts in batch."""
        results = []

        for text, chunk_id in texts_and_ids:
            try:
                extraction = await self.extract_knowledge_graph(text, chunk_id, language, entity_types, relation_types)
                results.append(extraction)
            except Exception as e:
                logger.error(f"Batch extraction failed for chunk {chunk_id}: {e}")
                # Create error result
                error_extraction = KnowledgeGraphExtraction(
                    chunk_id=chunk_id,
                    error_message=str(e),
                    confidence=0.0,
                    model_used=self._get_model_name(),
                )
                results.append(error_extraction)

        return results

    def is_model_loaded(self) -> bool:
        """Check if the LLM model is loaded and ready."""
        # Attempt to load model if not already loaded
        if not self._model_loaded or self._llm is None:
            try:
                self._ensure_model_loaded()
            except Exception as e:
                logger.error(f"Failed to load model during check: {e}")
                return False

        return self._model_loaded and self._llm is not None

    def __del__(self) -> None:
        """Cleanup resources."""
        if self._llm is not None:
            del self._llm

    def _get_model_name(self) -> str:
        """Get the name of the currently loaded model."""
        if self._detected_model_name:
            return self._detected_model_name

        # Try to detect model name from path
        model_path_str = str(self.model_path).lower()
        if "tinyswallow" in model_path_str:
            self._detected_model_name = "TinySwallow-1.5B-Instruct"
        elif "elyza" in model_path_str or "llama-3" in model_path_str:
            self._detected_model_name = "Llama-3-ELYZA-JP-8B"
        else:
            self._detected_model_name = "Japanese-LLM-Unknown"

        return self._detected_model_name
