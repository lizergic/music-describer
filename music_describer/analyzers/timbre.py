import librosa
import numpy as np

from .base import BaseAnalyzer


class TimbreAnalyzer(BaseAnalyzer):
    """Analyzes spectral characteristics, instrumentation hints, and vocal presence."""

    def analyze(self, waveform: np.ndarray, sr: int) -> dict:
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)

        brightness = float(np.mean(spectral_centroid)) / (sr / 2)  # normalize to 0-1
        warmth = 1.0 - brightness  # inverse of brightness

        tonal_quality = self._classify_tonal_quality(brightness)
        instrumentation_hints = self._guess_instrumentation(
            brightness, spectral_rolloff, mfccs, sr
        )
        has_vocals = self._detect_vocals(mfccs, spectral_centroid, sr)

        return {
            "brightness": round(brightness, 3),
            "warmth": round(warmth, 3),
            "instrumentation_hints": instrumentation_hints,
            "has_vocals": has_vocals,
            "tonal_quality": tonal_quality,
        }

    def _classify_tonal_quality(self, brightness: float) -> str:
        if brightness > 0.35:
            return "bright"
        elif brightness > 0.2:
            return "neutral"
        elif brightness > 0.1:
            return "warm"
        else:
            return "dark"

    def _guess_instrumentation(
        self,
        brightness: float,
        spectral_rolloff: np.ndarray,
        mfccs: np.ndarray,
        sr: int,
    ) -> list[str]:
        """Rough instrumentation guesses from spectral profile."""
        hints = []
        rolloff_mean = float(np.mean(spectral_rolloff)) / (sr / 2)

        # High brightness + high rolloff → likely cymbals/hi-hats or synths
        if brightness > 0.3 and rolloff_mean > 0.5:
            hints.append("bright synths or cymbals")

        # Low brightness + low rolloff → bass-heavy content
        if brightness < 0.15:
            hints.append("bass-heavy")

        # MFCC variance as a proxy for timbral variety
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
        if mfcc_var > 50:
            hints.append("varied instrumentation")
        elif mfcc_var < 10:
            hints.append("sparse or uniform timbre")

        # Mid-range brightness → guitars, piano, vocals
        if 0.15 <= brightness <= 0.3:
            hints.append("mid-range instruments (guitar, piano, or vocals)")

        if not hints:
            hints.append("mixed instrumentation")

        return hints

    def _detect_vocals(
        self,
        mfccs: np.ndarray,
        spectral_centroid: np.ndarray,
        sr: int,
    ) -> bool:
        """Simple vocal presence heuristic based on MFCC patterns."""
        # Vocals tend to have high variance in lower MFCCs (formant changes)
        # and spectral centroid in the 300-3000 Hz range
        lower_mfcc_var = float(np.mean(np.var(mfccs[1:5], axis=1)))
        centroid_hz = float(np.mean(spectral_centroid))

        has_formant_variation = lower_mfcc_var > 20
        in_vocal_range = 300 < centroid_hz < 3000

        return has_formant_variation and in_vocal_range
