# music-describer

Analyze audio files and generate natural-language descriptions of musical style and composition.

Extracts structured features (rhythm, harmony, timbre, structure, energy) via [librosa](https://librosa.org/), then optionally synthesizes them into musician-level prose using an LLM.

## Installation

```bash
pip install -e .
```

To enable LLM-powered descriptions, install with your preferred provider:

```bash
pip install -e ".[claude]"    # Anthropic Claude
pip install -e ".[openai]"    # OpenAI
pip install -e ".[all]"       # Both
```

Ollama requires no extra Python packages -- just a running [Ollama](https://ollama.com/) server.

## Quick Start

### CLI

```bash
# Structured analysis only (no LLM needed)
music-describer song.mp3 --analysis-only

# Natural-language description (requires LLM provider)
export ANTHROPIC_API_KEY="your-key"
music-describer song.mp3

# Full JSON output (analysis + description)
music-describer song.mp3 --json

# Save to file
music-describer song.mp3 --analysis-only -o analysis.json
```

### Python API

```python
from music_describer import analyze, describe

# Structured analysis only
result = analyze("song.mp3")
print(result["rhythm"]["tempo"])      # e.g. 120.0
print(result["harmony"]["key"])       # e.g. "A"
print(result["energy"]["energy_arc"]) # e.g. "building"

# With LLM description (set ANTHROPIC_API_KEY or configure provider)
result = describe("song.mp3")
print(result["description"])
# "A warm, mid-tempo track in A major with a driving rhythmic feel..."
print(result["analysis"]["timbre"]["tonal_quality"])
# "neutral"
```

## Analyzers

| Analyzer | Output Fields |
|----------|--------------|
| **Rhythm** | `tempo`, `time_signature`, `feel` (straight/swung/driving/laid-back), `beat_strength` |
| **Harmony** | `key`, `scale` (major/minor), `mode` (ionian, dorian, etc.), `harmonic_complexity` |
| **Timbre** | `brightness`, `warmth`, `instrumentation_hints`, `has_vocals`, `tonal_quality` |
| **Structure** | `sections` (labeled with start/end times), `form_summary` (e.g. "intro -> verse -> chorus -> outro") |
| **Energy** | `overall_energy`, `dynamic_range`, `energy_arc` (building/fading/flat/etc.), `notable_moments` |

## Configuration

Create a `config.yaml` in your working directory, or at `~/.music-describer/config.yaml`:

```yaml
llm:
  # Provider: claude, openai, or ollama
  provider: claude

  # Model name (provider-specific)
  # model: claude-sonnet-4-6

  # Environment variable holding the API key
  # api_key_env: ANTHROPIC_API_KEY

  # Ollama only
  # base_url: http://localhost:11434
```

Config resolution order: `--config` flag > `./config.yaml` > `~/.music-describer/config.yaml` > defaults.

See `config.example.yaml` for the full template.

## Supported Formats

MP3, WAV, FLAC, and any format supported by [libsndfile](http://www.mega-nerd.com/libsndfile/) or [ffmpeg](https://ffmpeg.org/) (via audioread).

## Development

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
source venv/Scripts/activate  # Windows (Git Bash)

pip install -e ".[dev,all]"
pytest -v
```

## License

MIT
