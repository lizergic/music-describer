import os
import tempfile

import yaml

from music_describer.config import load_config


def test_load_config_returns_defaults():
    # With no config file, should return defaults
    config = load_config(config_path="/nonexistent/path/config.yaml")
    assert "llm" in config
    assert config["llm"]["provider"] == "claude"


def test_load_config_from_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump({"llm": {"provider": "openai", "model": "gpt-4o"}}, f)
        f.flush()
        config = load_config(config_path=f.name)
    os.unlink(f.name)
    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["model"] == "gpt-4o"


def test_load_config_file_overrides_defaults():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump({"llm": {"provider": "ollama"}}, f)
        f.flush()
        config = load_config(config_path=f.name)
    os.unlink(f.name)
    # provider overridden, but model should still be None (default)
    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["model"] is None
