import librosa
import numpy as np
from scipy.signal import find_peaks

from .base import BaseAnalyzer


class StructureAnalyzer(BaseAnalyzer):
    """Segments audio into sections and labels them by energy/spectral changes."""

    def analyze(self, waveform: np.ndarray, sr: int) -> dict:
        duration = len(waveform) / sr
        hop_length = 512

        # Compute features
        rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]
        chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr, hop_length=hop_length)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, hop_length=hop_length, n_mfcc=13)

        # Find boundaries from feature changes
        boundary_frames = self._find_boundaries(rms, chroma, mfcc)
        boundary_times = librosa.frames_to_time(
            boundary_frames, sr=sr, hop_length=hop_length
        )

        # Build section list with start=0 and end=duration
        all_times = np.concatenate([[0.0], boundary_times, [duration]])
        all_times = np.unique(np.sort(all_times))

        sections = self._label_sections(all_times, rms, sr, hop_length)
        form_summary = " -> ".join(s["label"] for s in sections)

        return {
            "sections": sections,
            "form_summary": form_summary,
        }

    def _find_boundaries(
        self,
        rms: np.ndarray,
        chroma: np.ndarray,
        mfcc: np.ndarray,
    ) -> np.ndarray:
        """Find section boundaries from combined feature novelty."""
        # Combine features
        n_frames = min(len(rms), chroma.shape[1], mfcc.shape[1])
        rms_trim = rms[:n_frames]
        chroma_trim = chroma[:, :n_frames]
        mfcc_trim = mfcc[:, :n_frames]

        features = np.vstack(
            [
                librosa.util.normalize(rms_trim.reshape(1, -1)),
                librosa.util.normalize(chroma_trim),
                librosa.util.normalize(mfcc_trim),
            ]
        )

        # Compute novelty from feature differences
        diff = np.sum(np.abs(np.diff(features, axis=1)), axis=0)

        if len(diff) < 40:
            return np.array([], dtype=int)

        # Smooth the novelty curve
        kernel_size = min(20, len(diff) // 4)
        if kernel_size > 0:
            kernel = np.ones(kernel_size) / kernel_size
            diff_smooth = np.convolve(diff, kernel, mode="same")
        else:
            diff_smooth = diff

        # Find peaks above threshold
        threshold = np.mean(diff_smooth) + 1.0 * np.std(diff_smooth)
        min_distance = max(40, n_frames // 10)  # minimum ~2 seconds between sections
        peaks, _ = find_peaks(
            diff_smooth, height=threshold, distance=min_distance
        )

        return peaks

    def _label_sections(
        self,
        boundary_times: np.ndarray,
        rms: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> list[dict]:
        """Label sections based on energy and position."""
        sections = []
        n_sections = len(boundary_times) - 1
        avg_energy = float(np.mean(rms))

        for i in range(n_sections):
            start = float(boundary_times[i])
            end = float(boundary_times[i + 1])
            segment_duration = end - start

            # Get mean energy for this section
            start_frame = librosa.time_to_frames(start, sr=sr, hop_length=hop_length)
            end_frame = librosa.time_to_frames(end, sr=sr, hop_length=hop_length)
            end_frame = min(end_frame, len(rms))
            if end_frame > start_frame:
                section_energy = float(np.mean(rms[start_frame:end_frame]))
            else:
                section_energy = 0.0

            label = self._guess_label(
                i, n_sections, section_energy, segment_duration, avg_energy
            )

            sections.append(
                {
                    "label": label,
                    "start_time": round(start, 1),
                    "end_time": round(end, 1),
                }
            )

        return sections

    def _guess_label(
        self,
        index: int,
        total: int,
        energy: float,
        duration: float,
        avg_energy: float,
    ) -> str:
        """Heuristic section labeling based on position and energy."""
        # First section: intro if short or quiet
        if index == 0 and (duration < 15 or energy < avg_energy * 0.7):
            return "intro"
        # Last section: outro if quiet or short
        if index == total - 1 and (energy < avg_energy * 0.7 or duration < 15):
            return "outro"
        # High energy = chorus-like
        if energy > avg_energy * 1.2:
            return "chorus"
        # Low energy in middle = bridge
        if energy < avg_energy * 0.6 and 0 < index < total - 1:
            return "bridge"
        # Default
        return "verse"
