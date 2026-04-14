# EvalBench — LLM evaluation toolkit — BLEU, ROUGE, semantic similarity, and custom metrics for benchmarking AI outputs

LLM evaluation toolkit — BLEU, ROUGE, semantic similarity, and custom metrics for benchmarking AI outputs.

## Why EvalBench

EvalBench exists to make this workflow practical. Llm evaluation toolkit — bleu, rouge, semantic similarity, and custom metrics for benchmarking ai outputs. It favours a small, inspectable surface over sprawling configuration.

## Features

- CLI command `evalbench`
- `TestCase` — exported from `src/evalbench/core.py`
- `EvalResult` — exported from `src/evalbench/core.py`
- `EvalReport` — exported from `src/evalbench/core.py`
- Included test suite
- Dedicated documentation folder

## Tech Stack

- **Runtime:** Python
- **Frameworks:** Typer
- **Tooling:** Rich, Pydantic

## How It Works

The codebase is organised into `docs/`, `src/`, `tests/`. The primary entry points are `src/evalbench/core.py`, `src/evalbench/__init__.py`. `src/evalbench/core.py` exposes `TestCase`, `EvalResult`, `EvalReport` — the core types that drive the behaviour.

## Getting Started

```bash
pip install -e .
evalbench --help
```

## Usage

```bash
evalbench --help
```

## Project Structure

```
EvalBench/
├── .env.example
├── CONTRIBUTING.md
├── Makefile
├── README.md
├── docs/
├── pyproject.toml
├── src/
├── tests/
```