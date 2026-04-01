from unittest.mock import MagicMock, patch

from music_describer import analyze, describe


def test_analyze_returns_all_sections(tmp_wav_file):
    result = analyze(tmp_wav_file)
    assert "rhythm" in result
    assert "harmony" in result
    assert "timbre" in result
    assert "structure" in result
    assert "energy" in result


def test_analyze_rhythm_has_tempo(tmp_wav_file):
    result = analyze(tmp_wav_file)
    assert "tempo" in result["rhythm"]
    assert result["rhythm"]["tempo"] > 0


def test_analyze_harmony_has_key(tmp_wav_file):
    result = analyze(tmp_wav_file)
    assert "key" in result["harmony"]


def test_describe_returns_analysis_and_description(tmp_wav_file):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "A melodic track in A major."

    with patch("music_describer.get_provider", return_value=mock_provider):
        with patch("music_describer.load_config", return_value={"llm": {}}):
            result = describe(tmp_wav_file)

    assert "analysis" in result
    assert "description" in result
    assert result["description"] == "A melodic track in A major."
    assert "rhythm" in result["analysis"]


def test_describe_calls_llm_with_analysis(tmp_wav_file):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "description"

    with patch("music_describer.get_provider", return_value=mock_provider):
        with patch("music_describer.load_config", return_value={"llm": {}}):
            describe(tmp_wav_file)

    mock_provider.generate.assert_called_once()
    call_args = mock_provider.generate.call_args
    # system_prompt and user_prompt should be passed
    assert len(call_args.kwargs) == 2 or len(call_args.args) == 2
