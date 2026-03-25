"""Core evaluation engine for EvalBench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import EvalConfig
from .utils import (
    aggregate_scores,
    bleu_score,
    compute_cosine_similarity,
    exact_match,
    length_ratio,
    rouge_l_score,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single evaluation test case."""

    input: str
    expected_output: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        return cls(
            input=data["input"],
            expected_output=data["expected_output"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvalResult:
    """Result for a single test case."""

    test_case: TestCase
    actual_output: str
    scores: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.test_case.input,
            "expected": self.test_case.expected_output,
            "actual": self.actual_output,
            "scores": self.scores,
            "weighted_score": round(self.weighted_score, 4),
            "passed": self.passed,
        }


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    results: List[EvalResult]
    aggregate: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pass_rate: float = 0.0
    total: int = 0
    passed_count: int = 0
    failed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "pass_rate": round(self.pass_rate, 4),
            "aggregate": self.aggregate,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# EvalSuite — collection of test cases
# ---------------------------------------------------------------------------

class EvalSuite:
    """A named collection of test cases."""

    def __init__(self, name: str, test_cases: Optional[List[TestCase]] = None):
        self.name = name
        self.test_cases: List[TestCase] = test_cases or []

    def add(self, test_case: TestCase) -> None:
        self.test_cases.append(test_case)

    def filter_by_tag(self, tag: str) -> List[TestCase]:
        return [tc for tc in self.test_cases if tag in tc.tags]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EvalSuite":
        data = json.loads(Path(path).read_text())
        cases = [TestCase.from_dict(tc) for tc in data["test_cases"]]
        return cls(name=data["name"], test_cases=cases)

    def __len__(self) -> int:
        return len(self.test_cases)


# ---------------------------------------------------------------------------
# Built-in metric registry
# ---------------------------------------------------------------------------

BUILTIN_METRICS: Dict[str, Callable[[str, str], float]] = {
    "bleu": bleu_score,
    "rouge_l": rouge_l_score,
    "exact_match": exact_match,
    "semantic_similarity": compute_cosine_similarity,
    "length_ratio": length_ratio,
}


# ---------------------------------------------------------------------------
# Evaluator — the main engine
# ---------------------------------------------------------------------------

class Evaluator:
    """Run evaluation suites against LLM outputs and produce reports."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self._custom_metrics: Dict[str, Callable[[str, str], float]] = {}

    # -- custom metrics -----------------------------------------------------

    def register_metric(
        self, name: str, fn: Callable[[str, str], float]
    ) -> None:
        """Register a custom metric function.

        The function must accept (reference, hypothesis) and return a float.
        """
        self._custom_metrics[name] = fn

    def _get_metric_fn(self, name: str) -> Optional[Callable[[str, str], float]]:
        if name in self._custom_metrics:
            return self._custom_metrics[name]
        return BUILTIN_METRICS.get(name)

    # -- evaluation ---------------------------------------------------------

    def evaluate_case(
        self, test_case: TestCase, actual_output: str
    ) -> EvalResult:
        """Evaluate a single test case."""
        scores: Dict[str, float] = {}
        enabled = self.config.enabled_metrics()
        # Also include any registered custom metrics
        all_metric_names = list(set(enabled) | set(self._custom_metrics.keys()))

        for metric_name in all_metric_names:
            fn = self._get_metric_fn(metric_name)
            if fn is not None:
                scores[metric_name] = fn(test_case.expected_output, actual_output)

        weighted_score = self._weighted_score(scores)
        passed = weighted_score >= self.config.thresholds.pass_score

        return EvalResult(
            test_case=test_case,
            actual_output=actual_output,
            scores=scores,
            weighted_score=weighted_score,
            passed=passed,
        )

    def evaluate_suite(
        self,
        suite: EvalSuite,
        outputs: List[str],
    ) -> EvalReport:
        """Evaluate all test cases in a suite against provided outputs.

        ``outputs`` must be the same length as ``suite.test_cases``.
        """
        if len(outputs) != len(suite.test_cases):
            raise ValueError(
                f"Number of outputs ({len(outputs)}) must match "
                f"number of test cases ({len(suite.test_cases)})"
            )

        results = [
            self.evaluate_case(tc, out)
            for tc, out in zip(suite.test_cases, outputs)
        ]
        return self._build_report(results)

    # -- reporting ----------------------------------------------------------

    def _build_report(self, results: List[EvalResult]) -> EvalReport:
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)

        # Aggregate per metric
        metric_names: set[str] = set()
        for r in results:
            metric_names.update(r.scores.keys())

        aggregate: Dict[str, Dict[str, float]] = {}
        for mn in sorted(metric_names):
            values = [r.scores.get(mn, 0.0) for r in results]
            aggregate[mn] = aggregate_scores(values)

        return EvalReport(
            results=results,
            aggregate=aggregate,
            pass_rate=passed_count / total if total > 0 else 0.0,
            total=total,
            passed_count=passed_count,
            failed_count=total - passed_count,
        )

    def _weighted_score(self, scores: Dict[str, float]) -> float:
        """Compute a single weighted score from individual metric scores."""
        total_weight = 0.0
        weighted_sum = 0.0
        for metric_name, value in scores.items():
            w = self.config.get_weight(metric_name)
            # Custom metrics default to weight 1.0 if not in config
            if w == 0.0 and metric_name in self._custom_metrics:
                w = 1.0
            weighted_sum += value * w
            total_weight += w
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight
