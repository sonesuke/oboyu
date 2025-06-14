"""ELYZA-japanese-Llama-2-7b-instruct implementation for KG extraction.

This module provides a concrete implementation of the KG extraction service
using the ELYZA Japanese language model for local CPU-based inference.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import jaconv
from llama_cpp import Llama

from oboyu.domain.models.knowledge_graph import (
    ENTITY_TYPES,
    RELATION_TYPES,
    Entity,
    KnowledgeGraphExtraction,
    Relation,
)
from oboyu.ports.services.kg_extraction_service import ExtractionError, KGExtractionService

logger = logging.getLogger(__name__)


class ELYZAKGExtractionService(KGExtractionService):
    """ELYZA-based knowledge graph extraction service.

    Uses ELYZA-japanese-Llama-2-7b-instruct model for local CPU inference
    with Japanese language support and business-oriented entity/relation extraction.
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
        """Initialize ELYZA KG extraction service.

        Args:
            model_path: Path to ELYZA GGUF model file
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

    def _ensure_model_loaded(self) -> None:
        """Ensure the LLM model is loaded."""
        if self._llm is None:
            if not self.model_path.exists():
                raise ExtractionError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading ELYZA model from {self.model_path}")
            try:
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=self.verbose,
                )
                self._model_loaded = True
                logger.info("ELYZA model loaded successfully")
            except Exception as e:
                raise ExtractionError(f"Failed to load ELYZA model: {e}")

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
                model_used="ELYZA-japanese-Llama-2-7b-instruct",
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
        """Extract knowledge graph from text using ELYZA model."""
        start_time = time.time()

        try:
            self._ensure_model_loaded()

            # Normalize Japanese text
            normalized_text = self._normalize_japanese_text(text)

            # Create extraction prompt
            prompt = self._create_extraction_prompt(normalized_text, entity_types, relation_types)

            # Run inference
            logger.debug(f"Starting KG extraction for chunk {chunk_id}")

            # Run LLM inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            if self._llm is None:
                raise ExtractionError("Model not loaded", chunk_id)

            response = await loop.run_in_executor(
                None,
                lambda: self._llm.create_completion(  # type: ignore[union-attr]
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=["</s>", "Human:", "Assistant:"],
                ),
            )

            # Extract text response
            if isinstance(response, dict) and "choices" in response:
                response_text = response["choices"][0]["text"].strip()
            else:
                raise ExtractionError("Invalid response format from LLM", chunk_id)

            # Parse response into structured format
            extraction = self._parse_llm_response(response_text, chunk_id)

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
                    model_used="ELYZA-japanese-Llama-2-7b-instruct",
                )
                results.append(error_extraction)

        return results

    def is_model_loaded(self) -> bool:
        """Check if the LLM model is loaded and ready."""
        return self._model_loaded and self._llm is not None

    async def validate_extraction_schema(self) -> bool:
        """Validate that the model can produce valid JSON schema output."""
        test_text = "山田太郎はトヨタ自動車株式会社のCEOです。"
        test_chunk_id = "test_validation"

        try:
            result = await self.extract_knowledge_graph(test_text, test_chunk_id)
            # Check if we got valid entities and relations
            return len(result.entities) > 0 and result.error_message is None
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def __del__(self) -> None:
        """Cleanup resources."""
        if self._llm is not None:
            del self._llm
