import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

from music_describer.cli import main


def test_cli_analysis_only(tmp_wav_file, capsys):
    with patch("sys.argv", ["music-describer", tmp_wav_file, "--analysis-only"]):
        main()
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "rhythm" in result
    assert "harmony" in result


def test_cli_json_mode(tmp_wav_file, capsys):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "A test description."

    with patch("sys.argv", ["music-describer", tmp_wav_file, "--json"]):
        with patch("music_describer.get_provider", return_value=mock_provider):
            with patch("music_describer.load_config", return_value={"llm": {}}):
                main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "analysis" in result
    assert "description" in result


def test_cli_prose_mode(tmp_wav_file, capsys):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "A beautiful melody in D minor."

    with patch("sys.argv", ["music-describer", tmp_wav_file]):
        with patch("music_describer.get_provider", return_value=mock_provider):
            with patch("music_describer.load_config", return_value={"llm": {}}):
                main()

    captured = capsys.readouterr()
    assert "A beautiful melody in D minor." in captured.out


def test_cli_output_to_file(tmp_wav_file, tmp_path):
    output_file = str(tmp_path / "result.txt")

    with patch(
        "sys.argv",
        ["music-describer", tmp_wav_file, "--analysis-only", "--output", output_file],
    ):
        main()

    with open(output_file) as f:
        content = f.read()
    result = json.loads(content)
    assert "rhythm" in result
