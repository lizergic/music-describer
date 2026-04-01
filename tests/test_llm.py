import os
from unittest.mock import MagicMock, patch

from music_describer.llm.base import LLMProvider, get_provider
from music_describer.llm.prompt import SYSTEM_PROMPT, build_user_prompt


def test_get_provider_returns_claude_by_default():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        provider = get_provider({"provider": "claude"})
    from music_describer.llm.claude_provider import ClaudeProvider

    assert isinstance(provider, ClaudeProvider)


def test_get_provider_returns_openai():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = get_provider({"provider": "openai"})
    from music_describer.llm.openai_provider import OpenAIProvider

    assert isinstance(provider, OpenAIProvider)


def test_get_provider_returns_ollama():
    provider = get_provider({"provider": "ollama"})
    from music_describer.llm.ollama_provider import OllamaProvider

    assert isinstance(provider, OllamaProvider)


def test_get_provider_raises_on_unknown():
    try:
        get_provider({"provider": "nonexistent"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_claude_provider_generate():
    with patch("music_describer.llm.claude_provider.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="A rock song in E minor")]
        mock_client.messages.create.return_value = mock_response

        from music_describer.llm.claude_provider import ClaudeProvider

        provider = ClaudeProvider(model="claude-sonnet-4-6", api_key="test")
        result = provider.generate("system", "user")

        assert result == "A rock song in E minor"
        mock_client.messages.create.assert_called_once()


def test_openai_provider_generate():
    with patch("music_describer.llm.openai_provider.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.message.content = "An electronic track"
        mock_response = MagicMock()
        mock_response.choices = [mock_msg]
        mock_client.chat.completions.create.return_value = mock_response

        from music_describer.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        result = provider.generate("system", "user")

        assert result == "An electronic track"


def test_system_prompt_exists():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 50


def test_build_user_prompt_includes_analysis():
    analysis = {
        "rhythm": {"tempo": 120, "time_signature": "4/4"},
        "harmony": {"key": "C", "scale": "major"},
    }
    prompt = build_user_prompt(analysis)
    assert "120" in prompt
    assert "4/4" in prompt
    assert "C" in prompt
