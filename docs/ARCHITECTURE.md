# Architecture

## Overview

EvalBench is a modular LLM evaluation toolkit built around a pipeline of test cases, metrics, and reports.

## Module Structure

```
src/evalbench/
├── __init__.py      # Public API exports
├── __main__.py      # CLI entry point (Typer)
├── config.py        # Pydantic configuration models
├── core.py          # Evaluator, EvalSuite, TestCase, EvalResult, EvalReport
└── utils.py         # Text processing, n-gram extraction, scoring functions
```

## Data Flow

```
TestCase(s) + LLM Outputs
        │
        ▼
   ┌──────────┐
   │ Evaluator │──▶ Applies each metric to (expected, actual) pairs
   └──────────┘
        │
        ▼
   EvalResult (per case: individual metric scores + weighted score)
        │
        ▼
   EvalReport (aggregate stats, pass/fail rates, distributions)
```

## Metrics

| Metric              | Type       | Description                                   |
|---------------------|------------|-----------------------------------------------|
| BLEU                | Precision  | N-gram precision with brevity penalty          |
| ROUGE-L             | Recall     | Longest common subsequence recall              |
| Exact Match         | Binary     | Normalized string equality                     |
| Semantic Similarity | Similarity | TF-IDF cosine similarity                       |
| Length Ratio         | Ratio      | Hypothesis / reference token count (capped)    |

## Configuration

All settings are managed via Pydantic models in `config.py`:

- **MetricWeight**: enable/disable metrics and set relative weights
- **ThresholdConfig**: define pass/fail score boundaries
- **EvalConfig**: top-level config combining weights, thresholds, and runtime options

## Extensibility

Register custom metrics via `Evaluator.register_metric(name, fn)` where `fn` is any
callable with signature `(reference: str, hypothesis: str) -> float`.
