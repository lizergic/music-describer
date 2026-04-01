import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from system + user prompts."""


def get_provider(llm_config: dict) -> LLMProvider:
    """Instantiate the configured LLM provider."""
    provider_name = llm_config.get("provider", "claude")
    model = llm_config.get("model")
    api_key_env = llm_config.get("api_key_env")

    api_key = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)

    if provider_name == "claude":
        from .claude_provider import ClaudeProvider

        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        return ClaudeProvider(model=model or "claude-sonnet-4-6", api_key=api_key)

    elif provider_name == "openai":
        from .openai_provider import OpenAIProvider

        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAIProvider(model=model or "gpt-4o", api_key=api_key)

    elif provider_name == "ollama":
        from .ollama_provider import OllamaProvider

        base_url = llm_config.get("base_url", "http://localhost:11434")
        return OllamaProvider(model=model or "llama3", base_url=base_url)

    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
