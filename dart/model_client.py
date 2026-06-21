"""Model-client interface for the DART routers.

The routers are backend-agnostic: they only need a client exposing a single
``generate`` method. Implement this contract for your hybrid-reasoning model
(e.g. a local vLLM/transformers server, or a hosted think/no-think API).

Contract
--------
``generate(prompt, enable_thinking, temperature, max_tokens, thinking_budget=None)``
returns a ``dict`` with at least::

    {
        "text":            str,   # final answer text (post-thinking)
        "total_tokens":    int,   # tokens emitted for this call
        "thinking_tokens": int,   # tokens spent in the thinking trace (0 if no-think)
        "budget_truncated": bool, # True if the thinking trace hit `thinking_budget`
    }

When ``enable_thinking`` is False the model answers directly (no-think draft).
When True, ``thinking_budget`` (if given) caps the reasoning trace length.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ModelClient(ABC):
    """Abstract think/no-think model client."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        enable_thinking: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 512,
        thinking_budget: Optional[int] = None,
    ) -> dict:
        """Generate a completion. See module docstring for the return contract."""
        raise NotImplementedError


class MockModelClient(ModelClient):
    """Deterministic stub for tests and the example (no real model required).

    ``answers`` maps a prompt to the list of no-think draft answers it should
    emit (cycled across draft calls); ``think_answer`` is returned when thinking
    is enabled.
    """

    def __init__(self, answers: dict, think_answer: str = "42"):
        self._answers = answers
        self._think_answer = think_answer
        self._draft_idx = {}

    def generate(self, prompt, enable_thinking=False, temperature=0.7,
                 max_tokens=512, thinking_budget=None):
        if enable_thinking:
            return {
                "text": f"<think>...</think> The answer is \\boxed{{{self._think_answer}}}.",
                "total_tokens": 800,
                "thinking_tokens": 700,
                "budget_truncated": thinking_budget is not None and thinking_budget < 700,
            }
        drafts = self._answers.get(prompt, ["0"])
        i = self._draft_idx.get(prompt, 0)
        ans = drafts[i % len(drafts)]
        self._draft_idx[prompt] = i + 1
        return {
            "text": f"The answer is \\boxed{{{ans}}}.",
            "total_tokens": 40,
            "thinking_tokens": 0,
            "budget_truncated": False,
        }
