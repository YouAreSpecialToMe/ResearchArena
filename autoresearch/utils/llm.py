"""Unified LLM interface supporting Anthropic and OpenAI."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    usage: dict


class LLMClient:
    """Thin wrapper over Anthropic / OpenAI chat APIs."""

    def __init__(self, provider: str, model: str, temperature: float = 0.7, max_tokens: int = 8192):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic()
        elif self.provider == "openai":
            import openai
            return openai.OpenAI()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, system: str, user: str) -> LLMResponse:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return LLMResponse(
                content=resp.content[0].text,
                usage={"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
            )
        else:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return LLMResponse(
                content=resp.choices[0].message.content,
                usage={"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens},
            )

    def generate_json(self, system: str, user: str) -> dict:
        """Generate and parse JSON output from the LLM."""
        resp = self.generate(system, user + "\n\nRespond ONLY with valid JSON, no markdown fences.")
        text = resp.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
