from .energy import EnergyAnalyzer
from .harmony import HarmonyAnalyzer
from .rhythm import RhythmAnalyzer
from .structure import StructureAnalyzer
from .timbre import TimbreAnalyzer

ANALYZERS = {
    "rhythm": RhythmAnalyzer,
    "harmony": HarmonyAnalyzer,
    "timbre": TimbreAnalyzer,
    "structure": StructureAnalyzer,
    "energy": EnergyAnalyzer,
}

ALL_ANALYZERS = list(ANALYZERS.values())
