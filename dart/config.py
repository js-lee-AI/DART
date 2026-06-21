"""Configuration dataclasses for the DART routers."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SCRouteConfig:
    """Stage-1 Self-Consistency Routing (SC-Route) configuration."""
    # Draft generation
    K_init: int = 2
    K_max: int = 3
    draft_temperature: float = 0.7
    draft_max_tokens: int = 512

    # Thinking generation (escalation)
    think_temperature: float = 0.6
    answer_max_tokens: int = 4096

    # Task type (affects answer extraction / equivalence)
    task_type: str = "math"

    # Optional conformal-calibration thresholds (extended variant)
    tau_high: float = 1.0
    tau_low: float = 0.67


@dataclass
class LTTCalibrationResult:
    """Budget stages for the SC-Budget router.

    In the paper these stages are set from draft entropy via a calibrator;
    here we keep a minimal carrier so the router is runnable with manual or
    calibrated ``budget_stages``.
    """
    budget_stages: List[int] = field(default_factory=lambda: [1024, 2048])
