from music_describer.analyzers.rhythm import RhythmAnalyzer


def test_rhythm_returns_expected_keys(click_track_120bpm):
    waveform, sr = click_track_120bpm
    analyzer = RhythmAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert "tempo" in result
    assert "time_signature" in result
    assert "feel" in result
    assert "beat_strength" in result


def test_rhythm_tempo_near_120(click_track_120bpm):
    waveform, sr = click_track_120bpm
    analyzer = RhythmAnalyzer()
    result = analyzer.analyze(waveform, sr)
    # librosa's beat tracker should get close to 120 BPM
    assert 100 <= result["tempo"] <= 140


def test_rhythm_tempo_is_positive(sine_440):
    waveform, sr = sine_440
    analyzer = RhythmAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert result["tempo"] > 0


def test_rhythm_feel_is_valid_string(click_track_120bpm):
    waveform, sr = click_track_120bpm
    analyzer = RhythmAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert result["feel"] in ("straight", "swung", "driving", "laid-back")
