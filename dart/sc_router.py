"""
Self-Consistency Router (SC-Route) for Hybrid Reasoning Models.

Core algorithm:
  1. Generate K_init=2 no-think drafts (T=0.7 for diversity)
  2. Extract & normalize answers
  3. If all agree → accept (SC_ACCEPT_UNANIMOUS)
  4. If disagree → generate up to K_max=3 drafts for majority vote
  5. If 2/3+ agree → accept majority (SC_ACCEPT_MAJORITY)
  6. If no consensus → escalate to full thinking (SC_ESCALATE_FULL_THINK)
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from .answer_extraction import AnswerExtractor
from .config import SCRouteConfig

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    answer: str
    text: str
    action: str  # SC_ACCEPT_UNANIMOUS, SC_ACCEPT_MAJORITY, SC_ESCALATE_FULL_THINK
    total_tokens: int
    thinking_tokens: int
    num_drafts: int
    consistency_score: float
    correct: Optional[bool] = None
    draft_answers: List[str] = field(default_factory=list)
    draft_tokens: int = 0


class SCRouter:
    """Self-Consistency Router for think/no-think hybrid models."""

    def __init__(self, model_client, config: SCRouteConfig):
        self.model = model_client
        self.config = config
        self.extractor = AnswerExtractor()

    def route(self, question: str, task_type: str = None) -> RoutingResult:
        task_type = task_type or self.config.task_type

        # Stage 1: Generate K_init no-think drafts
        drafts = []
        for _ in range(self.config.K_init):
            draft = self.model.generate(
                prompt=question,
                enable_thinking=False,
                temperature=self.config.draft_temperature,
                max_tokens=self.config.draft_max_tokens,
            )
            drafts.append(draft)

        # Stage 2: Extract and normalize answers
        answers = []
        for d in drafts:
            raw = self.extractor.extract_and_normalize(d["text"], task_type)
            answers.append(raw)

        logger.debug(f"K={len(drafts)} answers: {answers}")

        # Stage 3: Check K_init agreement
        if self._all_equivalent(answers, task_type):
            draft_tokens = sum(d["total_tokens"] for d in drafts)
            return RoutingResult(
                answer=answers[0],
                text=drafts[0]["text"],
                action="SC_ACCEPT_UNANIMOUS",
                total_tokens=draft_tokens,
                thinking_tokens=0,
                num_drafts=self.config.K_init,
                consistency_score=1.0,
                draft_answers=answers,
                draft_tokens=draft_tokens,
            )

        # Stage 4: Adaptive escalation to K_max
        if self.config.K_init < self.config.K_max:
            for _ in range(self.config.K_max - self.config.K_init):
                extra = self.model.generate(
                    prompt=question,
                    enable_thinking=False,
                    temperature=self.config.draft_temperature,
                    max_tokens=self.config.draft_max_tokens,
                )
                drafts.append(extra)
                answers.append(
                    self.extractor.extract_and_normalize(extra["text"], task_type)
                )

            majority_answer, majority_count = self._majority_vote(answers)
            consistency_score = majority_count / len(answers)

            if majority_count >= 2:
                # Find the draft text matching the majority answer
                majority_text = self._find_draft_text(
                    drafts, answers, majority_answer, task_type
                )
                draft_tokens = sum(d["total_tokens"] for d in drafts)
                return RoutingResult(
                    answer=majority_answer,
                    text=majority_text,
                    action="SC_ACCEPT_MAJORITY",
                    total_tokens=draft_tokens,
                    thinking_tokens=0,
                    num_drafts=len(drafts),
                    consistency_score=consistency_score,
                    draft_answers=answers,
                    draft_tokens=draft_tokens,
                )

        # Stage 5: No consensus → full thinking
        think_resp = self.model.generate(
            prompt=question,
            enable_thinking=True,
            thinking_budget=None,
            temperature=self.config.think_temperature,
            max_tokens=self.config.answer_max_tokens,
        )

        draft_tokens = sum(d["total_tokens"] for d in drafts)
        think_tokens = think_resp.get("thinking_tokens", 0)
        think_total = think_resp.get("total_tokens", 0)

        return RoutingResult(
            answer=self.extractor.extract_and_normalize(
                think_resp["text"], task_type
            ),
            text=think_resp["text"],
            action="SC_ESCALATE_FULL_THINK",
            total_tokens=draft_tokens + think_total,
            thinking_tokens=think_tokens,
            num_drafts=len(drafts),
            consistency_score=0.0,
            draft_answers=answers,
            draft_tokens=draft_tokens,
        )

    def _all_equivalent(self, answers: List[str], task_type: str) -> bool:
        if len(answers) < 2:
            return False
        ref = answers[0]
        return all(
            self.extractor.equivalent(ref, a, task_type)
            for a in answers[1:]
        )

    def _majority_vote(self, answers: List[str]) -> tuple:
        counter = Counter(answers)
        return counter.most_common(1)[0]

    def _find_draft_text(
        self, drafts, answers, target, task_type
    ) -> str:
        for draft, ans in zip(drafts, answers):
            if self.extractor.equivalent(ans, target, task_type):
                return draft["text"]
        return drafts[0]["text"]
