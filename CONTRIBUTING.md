# Contributing to EvalBench

Thank you for your interest in contributing to EvalBench! We welcome contributions of all kinds.

## Getting Started

1. Fork the repository and clone your fork
2. Install development dependencies:
   ```bash
   make dev
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

## Development Workflow

### Running Tests

```bash
make test
```

### Linting

```bash
make lint
```

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run `make format` before committing.

## Adding a New Metric

1. Implement the metric function in `src/evalbench/utils.py` with the signature:
   ```python
   def my_metric(reference: str, hypothesis: str) -> float:
       ...
   ```
2. Register it in `BUILTIN_METRICS` in `src/evalbench/core.py`
3. Add a default weight entry in `EvalConfig` in `src/evalbench/config.py`
4. Write tests in `tests/test_core.py`

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if applicable
- Ensure all tests pass before submitting

## Reporting Issues

Open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
