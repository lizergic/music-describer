from music_describer.analyzers.harmony import HarmonyAnalyzer


def test_harmony_returns_expected_keys(sine_440):
    waveform, sr = sine_440
    analyzer = HarmonyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert "key" in result
    assert "scale" in result
    assert "mode" in result
    assert "harmonic_complexity" in result


def test_harmony_detects_a_from_440hz(sine_440):
    waveform, sr = sine_440
    analyzer = HarmonyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    # 440 Hz is A4 — key detection should find A
    assert result["key"] == "A"


def test_harmony_key_is_valid_note(sine_440):
    waveform, sr = sine_440
    analyzer = HarmonyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    valid_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    assert result["key"] in valid_notes


def test_harmony_complexity_is_valid(sine_440):
    waveform, sr = sine_440
    analyzer = HarmonyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert result["harmonic_complexity"] in ("simple", "moderate", "complex")
