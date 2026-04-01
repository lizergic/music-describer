import os
import tempfile

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sine_440():
    """5-second 440Hz (A4) sine wave at 22050 Hz sample rate."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return waveform, sr


@pytest.fixture
def click_track_120bpm():
    """5-second click track at 120 BPM. Short impulses on each beat."""
    sr = 22050
    duration = 5.0
    samples = int(sr * duration)
    waveform = np.zeros(samples, dtype=np.float32)
    interval = int(sr * 60 / 120)  # samples per beat at 120 BPM
    click_len = int(sr * 0.005)  # 5ms click
    for i in range(0, samples, interval):
        end = min(i + click_len, samples)
        waveform[i:end] = 0.8
    return waveform, sr


@pytest.fixture
def rising_energy():
    """10-second tone that gets louder over time (for energy analysis)."""
    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = np.linspace(0.05, 1.0, len(t))
    waveform = (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)
    return waveform, sr


@pytest.fixture
def two_section_audio():
    """10-second audio: 5s quiet sine, 5s loud sine (for structure detection)."""
    sr = 22050
    duration_each = 5.0
    t = np.linspace(0, duration_each, int(sr * duration_each), endpoint=False)
    quiet = (np.sin(2 * np.pi * 330 * t) * 0.1).astype(np.float32)
    loud = (np.sin(2 * np.pi * 440 * t) * 0.9).astype(np.float32)
    waveform = np.concatenate([quiet, loud])
    return waveform, sr


@pytest.fixture
def white_noise():
    """3-second white noise (high brightness, no tonal content)."""
    sr = 22050
    duration = 3.0
    rng = np.random.default_rng(42)
    waveform = rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.5
    return waveform, sr


@pytest.fixture
def tmp_wav_file(sine_440):
    """Write a sine wave to a temporary WAV file, yield path, clean up."""
    waveform, sr = sine_440
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, waveform, sr)
    yield path
    os.unlink(path)
