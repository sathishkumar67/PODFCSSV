# Contributing to PODFCSSV

This guide reflects the current single-file experiment setup.

## Setup

```bash
git clone https://github.com/sathishkumar67/PODFCSSV.git
cd PODFCSSV
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Current Entry Point

The repository now uses:

- `main.py`

Set `RUN_MODE` in `main.py` to:

- `federated`
- `baseline`

## Benchmark Design

Benchmark datasets:

- `EuroSAT`
- `GTSRB`
- `Food101`
- `Country211`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`

Retention-stress datasets used by both modes:

- `CIFAR10`
- `SVHN`
- `STL10`
- `CIFAR100`
- `Flowers102`
- `DTD`

The stress datasets are trained only to create stronger forgetting pressure and are excluded from the reported benchmark evaluation set.

## Development Rules

- keep `main.py` as the single source of truth for training and built-in forgetting evaluation
- keep the benchmark dataset list download-friendly
- keep the stress-dataset stream aligned between baseline and federated runs when making forgetting comparisons
- do not reintroduce manual-setup datasets into the default publishable schedule
- keep the baseline and federated training budgets aligned unless there is a strong experimental reason not to
- update docs whenever the benchmark, stage schedule, or tracked metrics change
- keep Python docstrings step-by-step and aligned with the current `main.py` execution flow

## Validation

Run at least:

```bash
python -m py_compile main.py src\__init__.py src\client.py src\server.py src\loss.py src\mae_with_adapter.py
```

If you want to verify dataset download paths without running training:

```bash
python test.py
```

If formatting tools are installed:

```bash
ruff check .
ruff format .
```

## Pull Requests

1. Create a focused branch from `main`.
2. Keep changes scoped to one pipeline change when possible.
3. Explain the experiment impact clearly.
4. Include the validation commands you ran.
5. Update documentation if behavior changed.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).


