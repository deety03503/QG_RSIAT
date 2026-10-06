"""Streaming class statistics and drift estimation."""

from .accumulators import ClassStatistics, ClassStatisticsAccumulator
from .drift import estimate_drift, compensate_prototypes

__all__ = [
    "ClassStatistics",
    "ClassStatisticsAccumulator",
    "compensate_prototypes",
    "estimate_drift",
]
