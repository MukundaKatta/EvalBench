"""CLI entry point for EvalBench."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import EvalConfig
from .core import EvalSuite, Evaluator

app = typer.Typer(
    name="evalbench",
    help="EvalBench — LLM evaluation and benchmarking toolkit.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    suite_path: str = typer.Argument(..., help="Path to suite JSON file"),
    outputs_path: str = typer.Argument(..., help="Path to outputs JSON (list of strings)"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config JSON"),
    output_format: str = typer.Option("table", "--format", "-f", help="table | json"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Pass threshold"),
) -> None:
    """Run an evaluation suite against a set of LLM outputs."""
    suite = EvalSuite.load(suite_path)
    outputs = json.loads(Path(outputs_path).read_text())

    config = EvalConfig()
    if config_path:
        raw = json.loads(Path(config_path).read_text())
        config = EvalConfig(**raw)
    config.thresholds.pass_score = threshold

    evaluator = Evaluator(config=config)
    report = evaluator.evaluate_suite(suite, outputs)

    if output_format == "json":
        console.print_json(report.to_json())
    else:
        _render_table(report)


@app.command()
def report(
    report_path: str = typer.Argument(..., help="Path to saved report JSON"),
) -> None:
    """Display a previously saved evaluation report."""
    data = json.loads(Path(report_path).read_text())
    console.print_json(json.dumps(data, indent=2))


@app.command()
def create_suite(
    output: str = typer.Argument(..., help="Output path for suite JSON"),
    name: str = typer.Option("my_suite", "--name", "-n"),
) -> None:
    """Create an empty evaluation suite scaffold."""
    suite = EvalSuite(name=name)
    suite.save(output)
    console.print(f"[green]Created suite scaffold at {output}[/green]")


def _render_table(eval_report) -> None:  # noqa: ANN001
    """Render an EvalReport as a rich table."""
    table = Table(title="EvalBench Results", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Input", max_width=40)
    table.add_column("Expected", max_width=30)
    table.add_column("Actual", max_width=30)
    table.add_column("Score", justify="right")
    table.add_column("Pass", justify="center")

    for i, r in enumerate(eval_report.results, 1):
        score_str = f"{r.weighted_score:.3f}"
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(
            str(i),
            r.test_case.input[:40],
            r.test_case.expected_output[:30],
            r.actual_output[:30],
            score_str,
            status,
        )

    console.print(table)

    # Summary
    console.print(
        f"\n[bold]Total:[/bold] {eval_report.total}  "
        f"[green]Passed:[/green] {eval_report.passed_count}  "
        f"[red]Failed:[/red] {eval_report.failed_count}  "
        f"[bold]Pass Rate:[/bold] {eval_report.pass_rate:.1%}"
    )

    # Per-metric aggregates
    if eval_report.aggregate:
        agg_table = Table(title="Metric Aggregates")
        agg_table.add_column("Metric")
        agg_table.add_column("Mean", justify="right")
        agg_table.add_column("Min", justify="right")
        agg_table.add_column("Max", justify="right")
        agg_table.add_column("Std", justify="right")
        for metric, stats in eval_report.aggregate.items():
            agg_table.add_row(
                metric,
                f"{stats['mean']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['std']:.4f}",
            )
        console.print(agg_table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
