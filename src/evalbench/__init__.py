"""EvalBench — LLM evaluation and benchmarking toolkit."""

__version__ = "0.1.0"

from .core import (
    Evaluator,
    EvalResult,
    EvalReport,
    EvalSuite,
    TestCase,
)
from .config import EvalConfig, MetricWeight
from .utils import (
    extract_ngrams,
    normalize_text,
    compute_cosine_similarity,
)

__all__ = [
    "Evaluator",
    "EvalResult",
    "EvalReport",
    "EvalSuite",
    "TestCase",
    "EvalConfig",
    "MetricWeight",
    "extract_ngrams",
    "normalize_text",
    "compute_cosine_similarity",
]
