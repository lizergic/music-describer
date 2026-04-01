from abc import ABC, abstractmethod

import numpy as np


class BaseAnalyzer(ABC):
    """Base class for all audio analyzers."""

    @abstractmethod
    def analyze(self, waveform: np.ndarray, sr: int) -> dict:
        """Analyze audio and return structured results."""
