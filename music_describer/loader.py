import os

import librosa
import numpy as np


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file and return (waveform, sample_rate).

    Supports MP3, WAV, FLAC via librosa/soundfile.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, sr = librosa.load(path, sr=sr, mono=True)
    return waveform.astype(np.float32), sr
