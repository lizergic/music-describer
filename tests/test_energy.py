from music_describer.analyzers.energy import EnergyAnalyzer


def test_energy_returns_expected_keys(sine_440):
    waveform, sr = sine_440
    analyzer = EnergyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert "overall_energy" in result
    assert "dynamic_range" in result
    assert "energy_arc" in result
    assert "notable_moments" in result


def test_energy_rising_audio_detected(rising_energy):
    waveform, sr = rising_energy
    analyzer = EnergyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert result["energy_arc"] == "building"


def test_energy_notable_moments_is_list(sine_440):
    waveform, sr = sine_440
    analyzer = EnergyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert isinstance(result["notable_moments"], list)


def test_energy_values_are_positive(sine_440):
    waveform, sr = sine_440
    analyzer = EnergyAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert result["overall_energy"] >= 0
    assert result["dynamic_range"] >= 0
