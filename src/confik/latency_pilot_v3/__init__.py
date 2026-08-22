"""Validation-only latency profiling for the frozen paper-v2 models.

This package deliberately has no dependency on the paper-v2 evaluation entry
point.  It may read allow-listed training/validation artifacts, but it refuses
test-named datasets and writes only to ``outputs/latency_pilot_v3``.
"""

from .optimized_inference import (
    EagerSeedEngine,
    OptimizedSeedEngine,
    VectorizedHGBRiskModel,
    VectorizedSeedMLP,
)

__all__ = [
    "EagerSeedEngine",
    "OptimizedSeedEngine",
    "VectorizedHGBRiskModel",
    "VectorizedSeedMLP",
]
