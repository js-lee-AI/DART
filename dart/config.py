from dataclasses import dataclass, field
from typing import List

@dataclass
class SCRouteConfig:
    K_init: int = 2
    K_max: int = 3
    draft_temperature: float = 0.7
    draft_max_tokens: int = 512
    think_temperature: float = 0.6
    answer_max_tokens: int = 4096
    task_type: str = 'math'
    tau_high: float = 1.0
    tau_low: float = 0.67

@dataclass
class LTTCalibrationResult:
    budget_stages: List[int] = field(default_factory=lambda : [1024, 2048])
