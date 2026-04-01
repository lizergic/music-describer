import librosa
import numpy as np

from .base import BaseAnalyzer

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Mode profiles (binary scale degrees relative to root)
MODE_PROFILES = {
    "ionian": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    "dorian": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    "phrygian": [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    "lydian": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    "mixolydian": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    "aeolian": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    "locrian": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
}


class HarmonyAnalyzer(BaseAnalyzer):
    """Detects key, scale, mode, and harmonic complexity."""

    def analyze(self, waveform: np.ndarray, sr: int) -> dict:
        chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        key, scale = self._detect_key(chroma_mean)
        mode = self._detect_mode(chroma_mean, key)
        complexity = self._harmonic_complexity(chroma)

        return {
            "key": key,
            "scale": scale,
            "mode": mode,
            "harmonic_complexity": complexity,
        }

    def _detect_key(self, chroma_mean: np.ndarray) -> tuple[str, str]:
        """Detect key using correlation with Krumhansl-Kessler profiles."""
        best_corr = -2.0
        best_key = "C"
        best_scale = "major"

        for i in range(12):
            rotated = np.roll(chroma_mean, -i)
            major_corr = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
            minor_corr = float(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = NOTE_NAMES[i]
                best_scale = "major"
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = NOTE_NAMES[i]
                best_scale = "minor"

        return best_key, best_scale

    def _detect_mode(self, chroma_mean: np.ndarray, key: str) -> str:
        """Attempt mode detection beyond major/minor."""
        root_idx = NOTE_NAMES.index(key)
        rotated = np.roll(chroma_mean, -root_idx)

        # Normalize to sum to 1
        total = np.sum(rotated)
        if total > 0:
            rotated = rotated / total

        best_mode = "ionian"
        best_score = -1.0

        for mode_name, profile in MODE_PROFILES.items():
            profile_arr = np.array(profile, dtype=float)
            profile_arr = profile_arr / np.sum(profile_arr)
            score = float(np.dot(rotated, profile_arr))
            if score > best_score:
                best_score = score
                best_mode = mode_name

        return best_mode

    def _harmonic_complexity(self, chroma: np.ndarray) -> str:
        """Measure harmonic complexity via chroma entropy."""
        # Normalize each frame
        frame_sums = np.sum(chroma, axis=0, keepdims=True) + 1e-10
        chroma_norm = chroma / frame_sums
        entropy = -np.sum(
            chroma_norm * np.log2(chroma_norm + 1e-10), axis=0
        )
        avg_entropy = float(np.mean(entropy))

        if avg_entropy < 2.0:
            return "simple"
        elif avg_entropy < 3.0:
            return "moderate"
        else:
            return "complex"
