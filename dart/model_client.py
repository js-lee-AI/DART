from abc import ABC, abstractmethod
from typing import Optional

class ModelClient(ABC):

    @abstractmethod
    def generate(self, prompt: str, enable_thinking: bool=False, temperature: float=0.7, max_tokens: int=512, thinking_budget: Optional[int]=None) -> dict:
        raise NotImplementedError

class MockModelClient(ModelClient):

    def __init__(self, answers: dict, think_answer: str='42'):
        self._answers = answers
        self._think_answer = think_answer
        self._draft_idx = {}

    def generate(self, prompt, enable_thinking=False, temperature=0.7, max_tokens=512, thinking_budget=None):
        if enable_thinking:
            return {'text': f'<think>...</think> The answer is \\boxed{{{self._think_answer}}}.', 'total_tokens': 800, 'thinking_tokens': 700, 'budget_truncated': thinking_budget is not None and thinking_budget < 700}
        drafts = self._answers.get(prompt, ['0'])
        i = self._draft_idx.get(prompt, 0)
        ans = drafts[i % len(drafts)]
        self._draft_idx[prompt] = i + 1
        return {'text': f'The answer is \\boxed{{{ans}}}.', 'total_tokens': 40, 'thinking_tokens': 0, 'budget_truncated': False}
