"""Tests for EvalBench core evaluation engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evalbench.config import EvalConfig
from evalbench.core import EvalReport, EvalResult, EvalSuite, Evaluator, TestCase
from evalbench.utils import (
    bleu_score,
    compute_cosine_similarity,
    exact_match,
    length_ratio,
    rouge_l_score,
)


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


class TestBleuScore:
    def test_identical_strings(self):
        score = bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_different(self):
        score = bleu_score("the cat sat on the mat", "dogs run in the park quickly")
        assert score < 0.3

    def test_partial_overlap(self):
        score = bleu_score("the cat sat on the mat", "the cat is on a mat", max_n=2)
        assert 0.1 < score < 1.0

    def test_empty_input(self):
        assert bleu_score("", "hello") == 0.0
        assert bleu_score("hello", "") == 0.0


class TestRougeLScore:
    def test_identical(self):
        score = rouge_l_score("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_subsequence(self):
        score = rouge_l_score("the cat sat on the mat", "cat sat on mat")
        assert score > 0.5

    def test_no_overlap(self):
        score = rouge_l_score("hello world", "goodbye universe")
        assert score == 0.0


class TestExactMatch:
    def test_exact(self):
        assert exact_match("Hello World!", "hello world") == 1.0

    def test_not_exact(self):
        assert exact_match("hello", "world") == 0.0


class TestSemanticSimilarity:
    def test_identical(self):
        score = compute_cosine_similarity("machine learning is great", "machine learning is great")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_similar(self):
        score = compute_cosine_similarity(
            "machine learning models are powerful",
            "deep learning models are very powerful",
        )
        assert score > 0.3

    def test_empty(self):
        assert compute_cosine_similarity("", "hello") == 0.0


class TestLengthRatio:
    def test_same_length(self):
        ratio = length_ratio("one two three", "four five six")
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_shorter(self):
        ratio = length_ratio("one two three four", "one two")
        assert ratio < 1.0

    def test_clamped(self):
        ratio = length_ratio("hi", "one two three four five six")
        assert ratio == 2.0


# ---------------------------------------------------------------------------
# EvalSuite tests
# ---------------------------------------------------------------------------


class TestEvalSuite:
    def test_create_and_add(self):
        suite = EvalSuite(name="test")
        suite.add(TestCase(input="q1", expected_output="a1", tags=["math"]))
        suite.add(TestCase(input="q2", expected_output="a2"))
        assert len(suite) == 2

    def test_filter_by_tag(self):
        suite = EvalSuite(name="test")
        suite.add(TestCase(input="q1", expected_output="a1", tags=["math"]))
        suite.add(TestCase(input="q2", expected_output="a2", tags=["code"]))
        assert len(suite.filter_by_tag("math")) == 1

    def test_save_and_load(self, tmp_path):
        suite = EvalSuite(name="persist_test")
        suite.add(TestCase(input="x", expected_output="y", tags=["t"]))
        path = tmp_path / "suite.json"
        suite.save(path)

        loaded = EvalSuite.load(path)
        assert loaded.name == "persist_test"
        assert len(loaded) == 1
        assert loaded.test_cases[0].expected_output == "y"


# ---------------------------------------------------------------------------
# Evaluator & Report tests
# ---------------------------------------------------------------------------


class TestEvaluator:
    def _make_suite(self):
        suite = EvalSuite(name="demo")
        suite.add(TestCase(input="Translate hello", expected_output="hola"))
        suite.add(TestCase(input="Capital of France", expected_output="Paris"))
        suite.add(TestCase(input="2+2", expected_output="4"))
        return suite

    def test_evaluate_case_perfect(self):
        ev = Evaluator()
        tc = TestCase(input="q", expected_output="the answer is 42")
        result = ev.evaluate_case(tc, "the answer is 42")
        assert result.scores["exact_match"] == 1.0
        assert result.passed is True

    def test_evaluate_suite(self):
        ev = Evaluator()
        suite = self._make_suite()
        outputs = ["hola", "Paris", "4"]
        report = ev.evaluate_suite(suite, outputs)
        assert report.total == 3
        assert report.passed_count == 3
        assert report.pass_rate == pytest.approx(1.0)

    def test_evaluate_suite_length_mismatch(self):
        ev = Evaluator()
        suite = self._make_suite()
        with pytest.raises(ValueError, match="Number of outputs"):
            ev.evaluate_suite(suite, ["one"])

    def test_custom_metric(self):
        def always_one(ref: str, hyp: str) -> float:
            return 1.0

        ev = Evaluator()
        ev.register_metric("custom_always_one", always_one)
        tc = TestCase(input="q", expected_output="a")
        result = ev.evaluate_case(tc, "b")
        assert "custom_always_one" in result.scores
        assert result.scores["custom_always_one"] == 1.0

    def test_report_json(self):
        ev = Evaluator()
        suite = self._make_suite()
        outputs = ["hola", "London", "five"]
        report = ev.evaluate_suite(suite, outputs)
        data = json.loads(report.to_json())
        assert "total" in data
        assert data["total"] == 3
        assert "aggregate" in data
        assert len(data["results"]) == 3
