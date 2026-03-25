"""Configuration models for EvalBench."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MetricWeight(BaseModel):
    """Weight assignment for a single metric."""

    name: str = Field(..., description="Metric identifier")
    weight: float = Field(1.0, ge=0.0, le=10.0, description="Relative weight")
    enabled: bool = Field(True, description="Whether this metric is active")


class ThresholdConfig(BaseModel):
    """Pass/fail thresholds for evaluation."""

    pass_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum passing score")
    warn_score: float = Field(0.3, ge=0.0, le=1.0, description="Warning threshold")


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    metric_weights: List[MetricWeight] = Field(
        default_factory=lambda: [
            MetricWeight(name="bleu", weight=1.0),
            MetricWeight(name="rouge_l", weight=1.0),
            MetricWeight(name="exact_match", weight=1.0),
            MetricWeight(name="semantic_similarity", weight=1.5),
            MetricWeight(name="length_ratio", weight=0.5),
        ],
        description="Weights for built-in metrics",
    )
    thresholds: ThresholdConfig = Field(
        default_factory=ThresholdConfig,
        description="Pass/fail thresholds",
    )
    max_workers: int = Field(4, ge=1, le=32, description="Parallel workers")
    output_format: str = Field("table", description="Output format: table | json | csv")
    suite_path: Optional[str] = Field(None, description="Path to suite JSON file")
    custom_metrics: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom metric name -> dotted import path",
    )

    def get_weight(self, metric_name: str) -> float:
        """Look up the weight for a metric by name."""
        for mw in self.metric_weights:
            if mw.name == metric_name and mw.enabled:
                return mw.weight
        return 0.0

    def enabled_metrics(self) -> List[str]:
        """Return names of all enabled metrics."""
        return [mw.name for mw in self.metric_weights if mw.enabled]
