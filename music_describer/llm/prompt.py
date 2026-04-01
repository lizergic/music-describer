import json

SYSTEM_PROMPT = """\
You are a music analysis assistant. Given structured audio analysis data, \
write a natural, musician-level description of the song.

Describe:
- Overall genre and style
- Mood and emotional character
- Key, scale/mode, and harmonic qualities
- Tempo and rhythmic feel (including time signature if notable)
- Instrumentation and tonal qualities
- Song structure and how it flows between sections
- Dynamic arc and notable energy moments

Write 2-3 paragraphs of natural prose. Be specific but accessible to musicians. \
Do not describe lyrics. Do not describe production or mixing details. \
Do not list the raw numbers — interpret them musically. \
Do not mention the analysis process or that you received structured data.\
"""

FIELD_REFERENCE = {
    "rhythm": {
        "tempo": "Beats per minute",
        "time_signature": "Meter (e.g. 4/4, 3/4, 7/8)",
        "feel": "Rhythmic character: straight, swung, driving, or laid-back",
        "beat_strength": "Average onset strength (higher = more percussive)",
    },
    "harmony": {
        "key": "Musical key (e.g. C, F#, Bb)",
        "scale": "Major or minor",
        "mode": "Specific mode (ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian)",
        "harmonic_complexity": "simple / moderate / complex based on chroma entropy",
    },
    "timbre": {
        "brightness": "0-1 normalized spectral centroid (higher = brighter/trebly)",
        "warmth": "1 - brightness",
        "instrumentation_hints": "Rough guesses about instruments present",
        "has_vocals": "Whether vocal-like formant patterns were detected",
        "tonal_quality": "bright / neutral / warm / dark",
    },
    "structure": {
        "sections": "Detected sections with labels, start and end times",
        "form_summary": "Section flow as a string (e.g. intro -> verse -> chorus -> outro)",
    },
    "energy": {
        "overall_energy": "Mean RMS energy (higher = louder overall)",
        "dynamic_range": "Difference between loudest and quietest moments",
        "energy_arc": "Overall trajectory: building, fading, flat, peaks in middle, etc.",
        "notable_moments": "Detected breakdowns, climaxes, and energy peaks with timestamps",
    },
}


def build_user_prompt(analysis: dict) -> str:
    """Build the user prompt from structured analysis data."""
    prompt = "Here is the structured analysis of an audio track:\n\n"
    prompt += json.dumps(analysis, indent=2)
    prompt += "\n\nField reference (what each field means):\n\n"
    prompt += json.dumps(FIELD_REFERENCE, indent=2)
    prompt += "\n\nWrite a musician-level description of this track."
    return prompt
