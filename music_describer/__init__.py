from .analyzers import ALL_ANALYZERS
from .config import load_config
from .llm import get_provider
from .llm.prompt import SYSTEM_PROMPT, build_user_prompt
from .loader import load_audio


def analyze(audio_path: str) -> dict:
    """Analyze an audio file and return structured results from all analyzers.

    Does not call an LLM — returns raw structured data only.
    """
    waveform, sr = load_audio(audio_path)
    results = {}
    for analyzer_cls in ALL_ANALYZERS:
        analyzer = analyzer_cls()
        name = type(analyzer).__name__.replace("Analyzer", "").lower()
        results[name] = analyzer.analyze(waveform, sr)
    return results


def describe(audio_path: str, config_path: str | None = None) -> dict:
    """Analyze an audio file and generate a natural-language description.

    Returns both the structured analysis and the LLM-generated prose.
    """
    analysis = analyze(audio_path)
    config = load_config(config_path)
    provider = get_provider(config["llm"])
    description = provider.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(analysis),
    )
    return {
        "analysis": analysis,
        "description": description,
    }
