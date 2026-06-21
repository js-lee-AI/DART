# DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets

Reference implementation of DART, a training-free router for hybrid reasoning
models (models that can answer directly or spend extra tokens on extended
"thinking"). DART uses cheap no-think drafts as a query-level difficulty probe:

* **Stage 1, SC-Route (draft agreement).** Sample `K=2` no-think drafts and
  accept the answer when they agree under a pluggable equivalence function, and
  route only disagreement cases to thinking.
* **Stage 2, budget probing.** For routed (disagreement) queries, allocate a
  query-specific thinking budget instead of always thinking to the cap.

No labeled difficulty data and no gradient updates are required, so DART works
with text-only API access to closed hybrid models.

> Paper: *DART: Draft-Agreement Routing for Training-Free Adaptive Thinking
> Budgets in Hybrid Reasoning Models.* A BibTeX entry will be added once the
> arXiv version is available.

## What's in this repository

```
dart/
  sc_router.py          Stage 1: Self-Consistency Routing (draft agreement)
  sc_budget_router.py   Stage 1 + Stage 2: budget-probing router
  answer_extraction.py  answer extraction and equivalence (math / mcq / code)
  config.py             router configuration dataclasses
  model_client.py       backend-agnostic model-client interface (plus a mock)
example.py              runnable demo using the mock client (no GPU needed)
```

## Installation

The core routers depend only on the Python standard library:

```bash
git clone https://github.com/js-lee-AI/DART.git && cd DART
python example.py
```

For real models, install a backend SDK (for example `pip install openai` for an
OpenAI-compatible vLLM server) and implement the model-client contract below.

## Usage

```python
from dart import SCRouter, SCRouteConfig

router = SCRouter(my_model_client, SCRouteConfig(task_type="math"))
result = router.route("What is the 7th prime number?")
print(result.action, result.answer, result.thinking_tokens)
```

### Model-client contract

The routers are backend-agnostic. Provide any object with a `generate` method:

```python
generate(prompt, enable_thinking, temperature, max_tokens, thinking_budget=None) -> {
    "text": str,              # final answer text
    "total_tokens": int,
    "thinking_tokens": int,   # 0 when enable_thinking is False
    "budget_truncated": bool, # True if the thinking trace hit thinking_budget
}
```

See `dart/model_client.py` for the abstract base class and a mock used by the
example. Implement it for your hybrid model (a local vLLM or transformers
server, or a hosted think/no-think API).

## Benchmarks

The paper evaluates on public datasets, which are **not redistributed here**.
Obtain them from their original sources under their respective licenses:
MATH-500, OlympiadBench, HumanEval, MBPP, and AIME 2024/2025.

## Results

DART matches or exceeds always-thinking (AT) accuracy on **13 of 14** model–benchmark pairs while reducing thinking tokens by **15–69%**. Accuracy in %, with the change vs AT in parentheses; **Think↓** is the thinking-token reduction vs AT. NT = no-think, AT = always-think.

| Model | Benchmark | NT | AT | DART (vs AT) | Think↓ |
|---|---|---|---|---|---|
| Qwen3-8B | MATH-500 | 76.6 | 85.6 | **88.2** (+2.6) | 67% |
| Qwen3-8B | OlympiadBench | 49.8 | 71.5 | 69.8 (−1.7) | 45% |
| Qwen3-8B | HumanEval | 60.4 | 59.1 | **78.7** (+19.6) | 55% |
| Qwen3-8B | MBPP | 60.7 | 64.2 | **68.9** (+4.7) | 58% |
| Qwen3-14B | MATH-500 | 81.2 | 87.6 | **87.6** (+0.0) | 37% |
| Qwen3-14B | OlympiadBench | 51.5 | 53.0 | **62.0** (+9.0) | 15% |
| Qwen3-14B | HumanEval | 71.3 | 66.5 | **78.7** (+12.2) | 60% |
| Qwen3-14B | MBPP | 64.6 | 64.6 | **68.1** (+3.5) | 51% |
| Qwen3-32B | MATH-500 | 82.2 | 86.2 | **88.5** (+2.3) | 69% |
| Qwen3-32B | OlympiadBench | 50.5 | 54.0 | **58.5** (+4.5) | 16% |
| Qwen3-32B | HumanEval | 79.9 | 72.6 | **95.1** (+22.5) | 63% |
| Qwen3-32B | MBPP | 65.8 | 65.8 | **71.2** (+5.4) | 51% |
| DeepSeek-V3.2 | MATH-500 | 84.8 | 88.4 | **92.6** (+4.2) | 56% |
| DeepSeek-V3.2 | OlympiadBench | 63.6 | 66.1 | **69.1** (+3.0) | 32% |

The only regression is Qwen3-8B on OlympiadBench (−1.7). DeepSeek-V3.2 is evaluated through the hosted API. No labeled data or gradient updates are used. See the paper for the SC-Route (Stage-1-only) and routing-rate columns.

## Citation

A BibTeX entry will be added once the arXiv version is available.

## License

The code in this repository is released under the [MIT License](LICENSE). The
paper itself is distributed under CC BY 4.0 via arXiv.
