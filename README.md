# DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets

Reference implementation of **DART**, a *training-free* router for hybrid
reasoning models (models that can answer directly or spend extra tokens on
extended "thinking"). DART uses cheap no-think drafts as a query-level
difficulty probe:

* **Stage 1 — SC-Route (draft agreement).** Sample `K=2` no-think drafts and
  accept the answer when they agree under a pluggable equivalence function;
  route only disagreement cases to thinking.
* **Stage 2 — budget probing.** For routed (disagreement) queries, allocate a
  query-specific thinking budget instead of always thinking to the cap.

No labeled difficulty data and no gradient updates are required, so DART works
with text-only API access to closed hybrid models.

> Paper: *DART: Draft-Agreement Routing for Training-Free Adaptive Thinking
> Budgets in Hybrid Reasoning Models.* A BibTeX entry will be added once the
> arXiv version is available.

## What's in this repository

```
dart/
  sc_router.py          # Stage 1: Self-Consistency Routing (draft agreement)
  sc_budget_router.py   # Stage 1 + Stage 2: budget-probing router
  answer_extraction.py  # answer extraction + equivalence (math / mcq / code)
  config.py             # router configuration dataclasses
  model_client.py       # backend-agnostic model-client interface (+ a mock)
example.py              # runnable demo using the mock client (no GPU needed)
```

## Installation

The core routers depend only on the Python standard library:

```bash
git clone https://github.com/js-lee-AI/DART.git && cd DART
python example.py        # runs with the built-in MockModelClient
```

For real models, install a backend SDK (e.g. `pip install openai` for an
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
    "text": str,             # final answer text
    "total_tokens": int,
    "thinking_tokens": int,  # 0 when enable_thinking is False
    "budget_truncated": bool, # True if the thinking trace hit thinking_budget
}
```

See `dart/model_client.py` for the abstract base class and a mock used by the
example. Implement it for your hybrid model (local vLLM/transformers server, or
a hosted think/no-think API).

## Benchmarks

The paper evaluates on public datasets, which are **not redistributed here** —
obtain them from their original sources under their respective licenses:
MATH-500, OlympiadBench, HumanEval, MBPP, and AIME 2024/2025.

## Citation

A BibTeX entry will be added once the arXiv version is available.

## License

The code in this repository is released under the [MIT License](LICENSE). The
paper itself is distributed under CC BY 4.0 via arXiv.
