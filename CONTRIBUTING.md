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

The repository now uses a single executable file:

- `main.py`

Set `RUN_MODE` in `main.py` to one of:

- `federated`
- `baseline`

The script does not take a command-line mode argument.

## Current Dataset Design

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

The full stage order is:

1. `EuroSAT` vs `GTSRB`
2. `CIFAR10` vs `SVHN`
3. `Food101` vs `Country211`
4. `STL10` vs `CIFAR100`
5. `Oxford-IIIT Pet` vs `FGVC Aircraft`
6. `Flowers102` vs `DTD`

## Split Rules

Benchmark training uses full train-side splits, except for `EuroSAT`, which is
fixed to `22000` training samples and `5000` held-out evaluation samples.

Stress datasets are merged into one self-supervised training pool per dataset:

- `CIFAR10`: `train + test`
- `STL10`: `train + test + unlabeled`
- `Flowers102`: `train + val + test`
- `SVHN`: `train + test + extra`
- `CIFAR100`: `train + test`
- `DTD`: `train + val + test`

## Development Rules

- keep `main.py` as the single source of truth for training and built-in evaluation
- keep the benchmark dataset list download-friendly
- keep the stress-dataset stream aligned between baseline and federated runs when making forgetting comparisons
- do not reintroduce manual-setup datasets into the default publishable schedule
- update docs whenever the benchmark, stage schedule, tracked metrics, or split policy changes
- keep Python docstrings and inline comments aligned with the exact execution flow in `main.py`
- preserve the two stage-wise evaluation passes: linear probe first, partial fine-tuning second

## Validation

Run at least:

```bash
python -m py_compile main.py src\__init__.py src\client.py src\server.py src\loss.py src\mae_with_adapter.py
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
