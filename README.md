# music-describer

Analyze audio files and generate natural-language descriptions of musical style and composition.

Extracts structured features (rhythm, harmony, timbre, structure, energy) via [librosa](https://librosa.org/), then optionally synthesizes them into musician-level prose using an LLM.

**Approximate API cost per description (Anthropic): USD $0.010**

_Prompt can be modified for efficiency._

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
# Show all available flags
music-describer --help

# Structured analysis only (no LLM needed)
music-describer song.mp3 --analysis-only

# Natural-language description (requires LLM provider)
export ANTHROPIC_API_KEY="your-key"
music-describer song.mp3

# Full JSON output (analysis + description)
music-describer song.mp3 --json

# Save analysis to a file
music-describer song.mp3 --analysis-only -o analysis.json

# Save full JSON (analysis + prose) to a file
music-describer song.mp3 --json -o song.json

# Run only a subset of analyzers (comma-separated)
music-describer song.mp3 --analysis-only --analyzers rhythm,harmony

# Use a specific config file (overrides ./config.yaml and ~/.music-describer/config.yaml)
music-describer song.mp3 --config ./my-config.yaml

# OpenAI provider
export OPENAI_API_KEY="your-key"
music-describer song.mp3 --config ./openai.yaml

# Local Ollama (no API key needed; requires a running Ollama server)
music-describer song.mp3 --config ./ollama.yaml

# Pipe JSON output through jq for ad-hoc inspection
music-describer song.mp3 --analysis-only | jq '.rhythm.tempo'

# Batch a directory of files (bash)
for f in tracks/*.mp3; do
  music-describer "$f" --analysis-only -o "analysis/$(basename "$f" .mp3).json"
done
```

> By default all five analyzers run. Pass `--analyzers` (CLI) or `analyzers=[...]` (Python API) to run a subset. Valid names: `rhythm`, `harmony`, `timbre`, `structure`, `energy`.

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

# Run only a subset of analyzers
result = analyze("song.mp3", analyzers=["rhythm", "harmony"])
result = describe("song.mp3", analyzers=["rhythm", "harmony", "energy"])
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
