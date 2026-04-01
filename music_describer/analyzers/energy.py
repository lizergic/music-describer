import librosa
import numpy as np

from .base import BaseAnalyzer


class EnergyAnalyzer(BaseAnalyzer):
    """Analyzes energy dynamics: overall level, range, arc, and notable moments."""

    def analyze(self, waveform: np.ndarray, sr: int) -> dict:
        hop_length = 512
        rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]

        overall_energy = float(np.mean(rms))
        dynamic_range = float(np.max(rms) - np.min(rms))
        energy_arc = self._detect_arc(rms)
        notable_moments = self._detect_notable_moments(rms, sr, hop_length)

        return {
            "overall_energy": round(overall_energy, 4),
            "dynamic_range": round(dynamic_range, 4),
            "energy_arc": energy_arc,
            "notable_moments": notable_moments,
        }

    def _detect_arc(self, rms: np.ndarray) -> str:
        """Classify the overall energy trajectory."""
        n = len(rms)
        if n < 4:
            return "flat"

        # Split into thirds
        third = n // 3
        start_energy = float(np.mean(rms[:third]))
        mid_energy = float(np.mean(rms[third : 2 * third]))
        end_energy = float(np.mean(rms[2 * third :]))

        max_e = max(start_energy, mid_energy, end_energy)
        if max_e == 0:
            return "flat"

        # Relative differences
        start_to_mid = (mid_energy - start_energy) / (max_e + 1e-10)
        mid_to_end = (end_energy - mid_energy) / (max_e + 1e-10)

        threshold = 0.15

        if start_to_mid > threshold and mid_to_end > threshold:
            return "building"
        elif start_to_mid < -threshold and mid_to_end < -threshold:
            return "fading"
        elif mid_energy > start_energy * 1.2 and mid_energy > end_energy * 1.2:
            return "peaks in middle"
        elif start_to_mid > threshold and mid_to_end < -threshold:
            return "builds then fades"
        elif abs(start_to_mid) < threshold and abs(mid_to_end) < threshold:
            return "flat"
        else:
            return "dynamic"

    def _detect_notable_moments(
        self, rms: np.ndarray, sr: int, hop_length: int
    ) -> list[dict]:
        """Find notable energy events: sudden drops (breakdowns) and peaks (climaxes)."""
        moments = []
        n = len(rms)
        if n < 10:
            return moments

        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))

        # Look for sudden energy changes using diff
        energy_diff = np.diff(rms)
        threshold_drop = -(mean_rms + 2 * std_rms)
        threshold_spike = mean_rms + 2 * std_rms

        for i in range(len(energy_diff)):
            time = float(librosa.frames_to_time(i, sr=sr, hop_length=hop_length))
            if energy_diff[i] < threshold_drop:
                moments.append({"time": round(time, 1), "type": "breakdown"})
            elif energy_diff[i] > threshold_spike:
                moments.append({"time": round(time, 1), "type": "climax"})

        # Also detect overall peak
        peak_frame = int(np.argmax(rms))
        peak_time = float(
            librosa.frames_to_time(peak_frame, sr=sr, hop_length=hop_length)
        )
        if rms[peak_frame] > mean_rms + std_rms:
            # Avoid duplicate if already detected
            if not any(
                abs(m["time"] - peak_time) < 1.0 and m["type"] == "climax"
                for m in moments
            ):
                moments.append({"time": round(peak_time, 1), "type": "peak"})

        moments.sort(key=lambda m: m["time"])
        return moments
