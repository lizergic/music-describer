from music_describer.analyzers.timbre import TimbreAnalyzer


def test_timbre_returns_expected_keys(sine_440):
    waveform, sr = sine_440
    analyzer = TimbreAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert "brightness" in result
    assert "warmth" in result
    assert "instrumentation_hints" in result
    assert "has_vocals" in result
    assert "tonal_quality" in result


def test_sine_is_darker_than_noise(sine_440, white_noise):
    analyzer = TimbreAnalyzer()
    sine_result = analyzer.analyze(*sine_440)
    noise_result = analyzer.analyze(*white_noise)
    # White noise has energy across all frequencies → higher brightness
    assert noise_result["brightness"] > sine_result["brightness"]


def test_instrumentation_hints_is_list(sine_440):
    analyzer = TimbreAnalyzer()
    result = analyzer.analyze(*sine_440)
    assert isinstance(result["instrumentation_hints"], list)


def test_tonal_quality_is_valid_string(sine_440):
    analyzer = TimbreAnalyzer()
    result = analyzer.analyze(*sine_440)
    assert result["tonal_quality"] in ("bright", "warm", "neutral", "dark")
