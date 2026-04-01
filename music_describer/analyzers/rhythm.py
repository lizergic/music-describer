import librosa
import numpy as np

from .base import BaseAnalyzer


class RhythmAnalyzer(BaseAnalyzer):
    """Extracts tempo, time signature, rhythmic feel, and beat strength."""

    def analyze(self, waveform: np.ndarray, sr: int) -> dict:
        tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
        beat_strength = float(np.mean(onset_env))

        # Fallback: no rhythmic content detected; use tempogram-based estimate
        if tempo <= 0:
            tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])

        time_signature = self._estimate_time_signature(onset_env, tempo, sr)
        feel = self._estimate_feel(onset_env, beats)

        return {
            "tempo": round(tempo, 1),
            "time_signature": time_signature,
            "feel": feel,
            "beat_strength": round(beat_strength, 3),
        }

    def _estimate_time_signature(
        self, onset_env: np.ndarray, tempo: float, sr: int
    ) -> str:
        """Estimate time signature from onset autocorrelation patterns."""
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        if len(ac) < 4:
            return "4/4"

        # Normalize
        if ac[0] > 0:
            ac = ac / ac[0]

        # Expected lag for one beat (in onset envelope frames)
        hop_length = 512
        if tempo <= 0:
            return "4/4"
        frames_per_beat = (60.0 / tempo) * (sr / hop_length)

        if frames_per_beat < 1:
            return "4/4"

        # Check groupings of 3 vs 4 beats
        lag_3 = int(round(frames_per_beat * 3))
        lag_4 = int(round(frames_per_beat * 4))

        corr_3 = ac[lag_3] if lag_3 < len(ac) else 0
        corr_4 = ac[lag_4] if lag_4 < len(ac) else 0

        # Also check for odd meters (5, 7)
        lag_5 = int(round(frames_per_beat * 5))
        lag_7 = int(round(frames_per_beat * 7))
        corr_5 = ac[lag_5] if lag_5 < len(ac) else 0
        corr_7 = ac[lag_7] if lag_7 < len(ac) else 0

        best = max(
            ("3/4", corr_3),
            ("4/4", corr_4),
            ("5/4", corr_5),
            ("7/8", corr_7),
            key=lambda x: x[1],
        )
        return best[0]

    def _estimate_feel(self, onset_env: np.ndarray, beats: np.ndarray) -> str:
        """Estimate rhythmic feel from onset patterns."""
        if len(beats) < 4:
            return "straight"

        # Compute inter-beat intervals
        intervals = np.diff(beats)
        if len(intervals) < 2:
            return "straight"

        # Swing detection: in swung rhythms, alternating intervals have
        # a long-short pattern (ratio > 1.3 suggests swing)
        even_intervals = intervals[::2]
        odd_intervals = intervals[1::2]
        min_len = min(len(even_intervals), len(odd_intervals))
        if min_len > 0:
            ratio = np.mean(even_intervals[:min_len]) / (
                np.mean(odd_intervals[:min_len]) + 1e-10
            )
            if ratio > 1.3 or ratio < 0.7:
                return "swung"

        # Energy-based feel: high overall onset strength = driving
        mean_strength = float(np.mean(onset_env))
        if mean_strength > np.percentile(onset_env, 75):
            return "driving"

        # Low variance in onset strength = laid-back
        cv = float(np.std(onset_env)) / (mean_strength + 1e-10)
        if cv < 0.5:
            return "laid-back"

        return "straight"
