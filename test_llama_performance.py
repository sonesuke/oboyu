#!/usr/bin/env python3
"""Simple llama.cpp performance test script."""

import time
from pathlib import Path

from llama_cpp import Llama


def test_llama_performance() -> None:
    """Test basic llama.cpp performance with Japanese LLM models."""
    # Try downloading Gemma 3 1B model directly
    from huggingface_hub import hf_hub_download
    from xdg_base_dirs import xdg_cache_home

    cache_base = xdg_cache_home() / "oboyu" / "kg" / "models"
    cache_base.mkdir(parents=True, exist_ok=True)

    try:
        print("🔄 Downloading TinySwallow 1.5B model...")
        model_path = Path(
            hf_hub_download(
                repo_id="SakanaAI/TinySwallow-1.5B-Instruct-GGUF",
                filename="tinyswallow-1.5b-instruct-q5_k_m.gguf",
                cache_dir=cache_base,
                local_files_only=False,
            )
        )
        print(f"✅ Downloaded to: {model_path}")
    except Exception as e:
        print(f"❌ Failed to download TinySwallow model: {e}")
        print("🔄 Trying existing models...")

        # Fallback to any available Japanese LLM model
        model_path = None
        # Check for TinySwallow models
        for model_dir in cache_base.glob("models--SakanaAI--TinySwallow-1.5B-Instruct-GGUF*"):
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot_dir in snapshots_dir.iterdir():
                    for gguf_file in snapshot_dir.glob("*.gguf"):
                        model_path = gguf_file
                        break
                    if model_path:
                        break
            if model_path:
                break

        # Check for ELYZA models as fallback
        if not model_path:
            for model_dir in cache_base.glob("models--elyza--Llama-3-ELYZA-JP-8B-GGUF*"):
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        for gguf_file in snapshot_dir.glob("*.gguf"):
                            model_path = gguf_file
                            break
                        if model_path:
                            break
                if model_path:
                    break

    if not model_path:
        print("❌ No Japanese LLM models found in cache")
        return

    print(f"📁 Using model: {model_path}")
    print(f"📏 Model size: {model_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    # Initialize model with minimal settings for speed
    print("🚀 Loading model...")
    start_time = time.time()

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,  # Smaller context for speed
        n_threads=4,  # Use 4 threads
        verbose=False,  # Disable verbose output
        n_gpu_layers=-1,  # Use all GPU layers for Metal acceleration
    )

    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")

    # Test simple Japanese prompt
    simple_prompt = """以下のテキストから人名を抽出してください。

テキスト: 山田太郎は東京のトヨタ自動車株式会社でCEOとして働いています。

回答:"""

    print("🤖 Testing simple extraction...")
    start_time = time.time()

    response = llm.create_completion(
        simple_prompt,
        max_tokens=100,  # Limit tokens for speed
        temperature=0.1,
        top_p=0.95,
        stop=["</s>", "\n\n"],
    )

    inference_time = time.time() - start_time
    print(f"⏱️  Simple inference: {inference_time:.2f} seconds")

    if isinstance(response, dict) and "choices" in response:
        response_text = response["choices"][0]["text"].strip()
        print(f"📝 Response: {response_text}")

    # Test complex KG extraction prompt (similar to what KG system uses)
    kg_prompt = """あなたは知識グラフ抽出の専門家です。以下のテキストから構造化された知識グラフを抽出してください。

テキスト:
山田太郎は東京のトヨタ自動車株式会社でCEOとして働いています。

抽出ルール:
1. エンティティタイプ: PERSON, COMPANY, ORGANIZATION, POSITION, LOCATION
2. リレーションタイプ: WORKS_AT, CEO_OF, LOCATED_IN
3. 明確に記載されている事実のみを抽出
4. 推測や解釈は含めない
5. 信頼度スコア(0.0-1.0)を各要素に付与

出力形式（JSON）:
{
  "entities": [
    {
      "name": "エンティティ名",
      "entity_type": "PERSON|COMPANY|ORGANIZATION|POSITION|LOCATION",
      "definition": "エンティティの定義・説明",
      "confidence": 0.95
    }
  ],
  "relations": [
    {
      "source": "ソースエンティティ名",
      "target": "ターゲットエンティティ名",
      "relation_type": "WORKS_AT|CEO_OF|LOCATED_IN",
      "confidence": 0.9
    }
  ]
}

JSON形式のみで回答してください:"""

    print("🧠 Testing complex KG extraction...")
    start_time = time.time()

    response = llm.create_completion(
        kg_prompt,
        max_tokens=512,  # KG extraction needs more tokens
        temperature=0.1,
        top_p=0.95,
        stop=["</s>", "Human:", "Assistant:"],
    )

    kg_inference_time = time.time() - start_time
    print(f"⏱️  KG inference: {kg_inference_time:.2f} seconds")

    if isinstance(response, dict) and "choices" in response:
        response_text = response["choices"][0]["text"].strip()
        print(f"📝 KG Response ({len(response_text)} chars):")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)

    print("\n📊 Performance Summary:")
    print(f"   Model Loading: {load_time:.2f}s")
    print(f"   Simple Query:  {inference_time:.2f}s")
    print(f"   KG Extraction: {kg_inference_time:.2f}s")

    if kg_inference_time > 60:
        print(f"⚠️  KG extraction is slow ({kg_inference_time:.2f}s > 60s)")
        print("💡 Consider:")
        print("   - Reducing max_tokens (currently 512)")
        print("   - Using smaller model")
        print("   - Increasing n_threads")
        print("   - Using GPU acceleration")
    else:
        print("✅ Performance looks good!")


if __name__ == "__main__":
    test_llama_performance()
