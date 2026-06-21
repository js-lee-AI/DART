"""
SC-Budget v9: Self-Consistency Budget Probing Router.

Algorithm:
  1. Generate K=2 no-think drafts at T=0.7
  2. If unanimous → accept (budget=0, skip thinking)
  3. If disagree → speculative budget probing with calibrated stages
     a. For each budget stage: generate thinking with budget
     b. If thinking concludes naturally → accept AT answer
     c. If truncated but cross-verified with draft → accept AT answer
     d. If truncated and diverged → escalate to next stage
  4. All stages exhausted → NT fallback (overthinking prevention)

Key differences from v8:
  - LTT calibration instead of conformal quantile (handles overthinking)
  - Multi-stage budget probing instead of single budget
  - NT fallback instead of full-AT escalation (critical fix)
  - Natural conclusion as primary confidence signal
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .answer_extraction import AnswerExtractor
from .config import LTTCalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class SCBudgetConfig:
    """Configuration for SC-Budget v9 router."""
    K: int = 2
    draft_temperature: float = 0.7
    draft_max_tokens: int = 512
    think_temperature: float = 0.6
    answer_max_tokens: int = 8192
    # Budget stages (set by LTT calibrator or manually)
    budget_stages: List[int] = field(default_factory=lambda: [1024, 2048])
    # NT fallback temperature (greedy for consistency)
    nt_fallback_temperature: float = 0.0
    # Task type (for answer extraction)
    task_type: str = "math"


@dataclass
class SCBudgetResult:
    """Result from SC-Budget v9 routing."""
    answer: str
    text: str
    action: str  # ACCEPT_UNANIMOUS, BUDGETED_THINK, VERIFIED_THINK, NT_FALLBACK
    total_tokens: int
    thinking_tokens: int
    budget_stage_used: Optional[int]
    num_drafts: int
    consistency_score: float
    draft_answers: List[str] = field(default_factory=list)
    draft_tokens: int = 0
    naturally_concluded: bool = False
    cross_verified: bool = False
    num_probes: int = 0
    correct: Optional[bool] = None


class SCBudgetRouter:
    """Self-Consistency Budget Probing Router v9."""

    def __init__(self, model_client, config: SCBudgetConfig,
                 calibration: LTTCalibrationResult = None):
        self.model = model_client
        self.config = config
        self.extractor = AnswerExtractor()
        if calibration:
            self.config.budget_stages = calibration.budget_stages

    def route(self, question: str, task_type: str = None) -> SCBudgetResult:
        """Route a question through the SC-Budget v9 pipeline."""
        task_type = task_type or self.config.task_type

        # Stage 1: K=2 no-think drafts
        drafts = []
        for _ in range(self.config.K):
            draft = self.model.generate(
                prompt=question,
                enable_thinking=False,
                temperature=self.config.draft_temperature,
                max_tokens=self.config.draft_max_tokens,
            )
            drafts.append(draft)

        answers = [
            self.extractor.extract_and_normalize(d["text"], task_type)
            for d in drafts
        ]
        draft_tokens = sum(
            sum(d.get("draft_tokens", [0])) if isinstance(d.get("draft_tokens"), list)
            else d.get("total_tokens", 0)
            for d in drafts
        )

        # Stage 2: Check agreement
        agreed = self._all_equivalent(answers, task_type)

        if agreed:
            return SCBudgetResult(
                answer=answers[0],
                text=drafts[0]["text"],
                action="ACCEPT_UNANIMOUS",
                total_tokens=draft_tokens,
                thinking_tokens=0,
                budget_stage_used=0,
                num_drafts=self.config.K,
                consistency_score=1.0,
                draft_answers=answers,
                draft_tokens=draft_tokens,
            )

        # Stage 3: Speculative Budget Probing
        total_think_tokens = 0
        total_think_total = 0
        num_probes = 0

        for budget in self.config.budget_stages:
            num_probes += 1
            logger.debug(f"SC-Budget: probe budget={budget}, answers={answers}")

            think_resp = self.model.generate(
                prompt=question,
                enable_thinking=True,
                thinking_budget=budget,
                temperature=self.config.think_temperature,
                max_tokens=self.config.answer_max_tokens,
            )

            think_answer = self.extractor.extract_and_normalize(
                think_resp["text"], task_type
            )
            think_tokens = think_resp.get("thinking_tokens", 0)
            think_total = think_resp.get("total_tokens", 0)
            total_think_tokens += think_tokens
            total_think_total += think_total
            naturally_concluded = not think_resp.get("budget_truncated", False)

            # 3a: Natural conclusion → high confidence, accept
            if naturally_concluded:
                return SCBudgetResult(
                    answer=think_answer,
                    text=think_resp["text"],
                    action="BUDGETED_THINK",
                    total_tokens=draft_tokens + total_think_total,
                    thinking_tokens=total_think_tokens,
                    budget_stage_used=budget,
                    num_drafts=self.config.K,
                    consistency_score=0.0,
                    draft_answers=answers,
                    draft_tokens=draft_tokens,
                    naturally_concluded=True,
                    num_probes=num_probes,
                )

            # 3b: Truncated → cross-mode verification
            # Check against majority answer among drafts (not any draft)
            from collections import Counter
            answer_groups = []
            for a in answers:
                if not a:
                    continue
                matched = False
                for gidx, (rep, cnt) in enumerate(answer_groups):
                    if self.extractor.equivalent(a, rep, task_type):
                        answer_groups[gidx] = (rep, cnt + 1)
                        matched = True
                        break
                if not matched:
                    answer_groups.append((a, 1))
            # Majority = most frequent answer group
            majority_answer = max(answer_groups, key=lambda x: x[1])[0] if answer_groups else None
            converged = (
                majority_answer is not None
                and self.extractor.equivalent(think_answer, majority_answer, task_type)
            )
            if converged:
                return SCBudgetResult(
                    answer=think_answer,
                    text=think_resp["text"],
                    action="VERIFIED_THINK",
                    total_tokens=draft_tokens + total_think_total,
                    thinking_tokens=total_think_tokens,
                    budget_stage_used=budget,
                    num_drafts=self.config.K,
                    consistency_score=0.0,
                    draft_answers=answers,
                    draft_tokens=draft_tokens,
                    cross_verified=True,
                    num_probes=num_probes,
                )

            # 3c: Truncated + diverged → try next stage
            logger.debug(f"SC-Budget: probe {budget} truncated+diverged, escalating")

        # Stage 4: NT Fallback (Overthinking Prevention)
        # All probing stages exhausted. Model is spiraling.
        # NT empirically outperforms full AT on these problems
        # (40-53% vs 0% at budget ceiling).
        nt_resp = self.model.generate(
            prompt=question,
            enable_thinking=False,
            temperature=self.config.nt_fallback_temperature,
            max_tokens=self.config.answer_max_tokens,
        )
        nt_answer = self.extractor.extract_and_normalize(
            nt_resp["text"], task_type
        )
        nt_total = nt_resp.get("total_tokens", 0)

        return SCBudgetResult(
            answer=nt_answer,
            text=nt_resp["text"],
            action="NT_FALLBACK",
            total_tokens=draft_tokens + total_think_total + nt_total,
            thinking_tokens=total_think_tokens,
            budget_stage_used=None,
            num_drafts=self.config.K,
            consistency_score=0.0,
            draft_answers=answers,
            draft_tokens=draft_tokens,
            num_probes=num_probes,
        )

    def _all_equivalent(self, answers: List[str], task_type: str) -> bool:
        if len(answers) < 2:
            return False
        ref = answers[0]
        if not ref:
            return False
        return all(
            self.extractor.equivalent(ref, a, task_type)
            for a in answers[1:]
            if a
        )
