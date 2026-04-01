import numpy as np

from music_describer.loader import load_audio


def test_load_wav_file(tmp_wav_file):
    waveform, sr = load_audio(tmp_wav_file)
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert sr == 22050
    assert len(waveform) > 0


def test_load_with_custom_sr(tmp_wav_file):
    waveform, sr = load_audio(tmp_wav_file, sr=16000)
    assert sr == 16000
    assert len(waveform) > 0


def test_load_nonexistent_file_raises():
    try:
        load_audio("nonexistent_file.wav")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass
