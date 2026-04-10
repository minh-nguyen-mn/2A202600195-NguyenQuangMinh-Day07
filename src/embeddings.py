from __future__ import annotations

import hashlib
import math

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


# class OpenAIEmbedder:
#     """OpenAI embeddings API-backed embedder."""

#     def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
#         from openai import OpenAI

#         self.model_name = model_name
#         self._backend_name = model_name
#         self.client = OpenAI()

#     def __call__(self, text: str) -> list[float]:
#         response = self.client.embeddings.create(model=self.model_name, input=text)
#         return [float(value) for value in response.data[0].embedding]

class OpenAIEmbedder:
    """OpenRouter embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        import os
        import requests

        self.model_name = model_name
        self._backend_name = f"openrouter/{model_name}"

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY in environment")

        self.base_url = "https://openrouter.ai/api/v1/embeddings"
        self.session = requests.Session()

    def __call__(self, text: str) -> list[float]:
        print(f"🔎 Embedding chunk ({len(text)} chars)...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "AI-in-Action-Lab",
        }

        payload = {
            "model": self.model_name,
            "input": text,
        }

        response = self.session.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=30,
        )

        try:
            data = response.json()
        except Exception:
            raise RuntimeError(f"Invalid response: {response.text}")

        # ✅ HANDLE FAILURE GRACEFULLY
        if response.status_code != 200 or "data" not in data:
            print("⚠️ Embedding failed → fallback to mock")
            return _mock_embed(text)

        return [float(x) for x in data["data"][0]["embedding"]]
    
_mock_embed = MockEmbedder()
