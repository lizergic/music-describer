from music_describer.analyzers.structure import StructureAnalyzer


def test_structure_returns_expected_keys(sine_440):
    waveform, sr = sine_440
    analyzer = StructureAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert "sections" in result
    assert "form_summary" in result


def test_structure_sections_have_required_fields(sine_440):
    waveform, sr = sine_440
    analyzer = StructureAnalyzer()
    result = analyzer.analyze(waveform, sr)
    for section in result["sections"]:
        assert "label" in section
        assert "start_time" in section
        assert "end_time" in section
        assert section["end_time"] > section["start_time"]


def test_structure_sections_cover_full_duration(sine_440):
    waveform, sr = sine_440
    analyzer = StructureAnalyzer()
    result = analyzer.analyze(waveform, sr)
    sections = result["sections"]
    assert len(sections) >= 1
    assert sections[0]["start_time"] == 0.0
    # Last section should end near the audio duration
    duration = len(waveform) / sr
    assert abs(sections[-1]["end_time"] - duration) < 1.0


def test_structure_detects_contrast(two_section_audio):
    waveform, sr = two_section_audio
    analyzer = StructureAnalyzer()
    result = analyzer.analyze(waveform, sr)
    # Should detect at least 2 distinct sections from quiet→loud transition
    assert len(result["sections"]) >= 2


def test_form_summary_is_string(sine_440):
    waveform, sr = sine_440
    analyzer = StructureAnalyzer()
    result = analyzer.analyze(waveform, sr)
    assert isinstance(result["form_summary"], str)
    assert len(result["form_summary"]) > 0
