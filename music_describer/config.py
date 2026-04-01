import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "llm": {
        "provider": "claude",
        "model": None,
        "api_key_env": None,
        "base_url": None,
    }
}


def load_config(config_path: str | None = None) -> dict:
    """Load config with cascading resolution: explicit path > cwd > home."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    paths = []
    if config_path:
        paths.append(Path(config_path))
    paths.append(Path.cwd() / "config.yaml")
    paths.append(Path.home() / ".music-describer" / "config.yaml")

    for path in paths:
        if path.exists():
            with open(path) as f:
                file_config = yaml.safe_load(f)
            if file_config:
                _deep_merge(config, file_config)
            break

    return config


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
