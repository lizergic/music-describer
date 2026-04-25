from .analyzers import ANALYZERS
from .config import load_config
from .llm import get_provider
from .llm.prompt import SYSTEM_PROMPT, build_user_prompt
from .loader import load_audio


def analyze(audio_path: str, analyzers: list[str] | None = None) -> dict:
    """Analyze an audio file and return structured results from the selected analyzers.

    Does not call an LLM — returns raw structured data only.

    Args:
        audio_path: Path to the audio file.
        analyzers: Optional list of analyzer names to run (e.g. ["rhythm", "harmony"]).
            Defaults to all analyzers. Valid names: rhythm, harmony, timbre, structure, energy.
    """
    selected = _resolve_analyzers(analyzers)
    waveform, sr = load_audio(audio_path)
    return {name: cls().analyze(waveform, sr) for name, cls in selected.items()}


def describe(
    audio_path: str,
    config_path: str | None = None,
    analyzers: list[str] | None = None,
) -> dict:
    """Analyze an audio file and generate a natural-language description.

    Returns both the structured analysis and the LLM-generated prose.

    Args:
        audio_path: Path to the audio file.
        config_path: Optional path to a config.yaml.
        analyzers: Optional subset of analyzer names; see analyze().
    """
    analysis = analyze(audio_path, analyzers=analyzers)
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


def _resolve_analyzers(analyzers: list[str] | None) -> dict:
    if analyzers is None:
        return ANALYZERS
    unknown = [name for name in analyzers if name not in ANALYZERS]
    if unknown:
        raise ValueError(
            f"Unknown analyzer(s): {unknown}. Available: {list(ANALYZERS)}"
        )
    return {name: ANALYZERS[name] for name in analyzers}
