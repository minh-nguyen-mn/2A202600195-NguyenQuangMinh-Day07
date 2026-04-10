import os
import requests
from dotenv import load_dotenv

load_dotenv()


class OpenRouterLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY in .env")

        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def __call__(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error: {response.text}")

        return response.json()["choices"][0]["message"]["content"]