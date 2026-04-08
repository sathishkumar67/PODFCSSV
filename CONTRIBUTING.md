# Contributing to PODFCSSV

This guide reflects the current repository workflow. The active pipeline is intentionally centralized in `main.py`, so contributions should preserve that single-source-of-truth design unless a deliberate refactor is planned.

## Environment Setup

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

## Current Code Layout

The active files are:

- `main.py`: full experiment orchestration, dataset loading, training, linear-probe evaluation, plotting, final checkpoint export, and separate final-probe artifact export
- `src/mae_with_adapter.py`: frozen-backbone adapter injection
- `src/loss.py`: GPAD loss
- `src/client.py`: client-side continual state, local training, and local prototype maintenance
- `src/server.py`: global prototype merging and adapter aggregation

## Run Modes

The repository does not currently use a command-line mode flag. Instead, set `RUN_MODE` inside `main.py`:

- `federated`
- `baseline`

Launch the run with:

```bash
python main.py
```

## Current Dataset Design

### Benchmark Datasets

These datasets define the core benchmark portion of the continual stream:

- `EuroSAT`
- `GTSRB`
- `Food101`
- `Country211`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`

Benchmark schedule:

- Client 0: `EuroSAT -> Food101 -> Oxford-IIIT Pet`
- Client 1: `GTSRB -> Country211 -> FGVC Aircraft`

### Stress Datasets

These datasets are inserted between benchmark stages and after the final benchmark stage to strengthen distribution shift:

- `CIFAR10`
- `SVHN`
- `STL10`
- `CIFAR100`
- `Flowers102`
- `DTD`

Stress schedule:

- Client 0: `CIFAR10 -> STL10 -> Flowers102`
- Client 1: `SVHN -> CIFAR100 -> DTD`

Full stage order:

1. `EuroSAT` vs `GTSRB`
2. `CIFAR10` vs `SVHN`
3. `Food101` vs `Country211`
4. `STL10` vs `CIFAR100`
5. `Oxford-IIIT Pet` vs `FGVC Aircraft`
6. `Flowers102` vs `DTD`

## Split Rules

Benchmark training uses the full train-side split for each dataset except `EuroSAT`, which uses a fixed head/tail `22000 / 5000` train-eval split.

Current benchmark evaluation splits:

- `EuroSAT`: last `5000`
- `Food101`: `test`
- `Oxford-IIIT Pet`: `test`
- `GTSRB`: `test`
- `Country211`: `valid`
- `FGVC Aircraft`: `test`

Stress datasets are merged into a single self-supervised training pool per dataset:

- `CIFAR10`: `train + test`
- `STL10`: `train + test + unlabeled`
- `Flowers102`: `train + val + test`
- `SVHN`: `train + test + extra`
- `CIFAR100`: `train + test`
- `DTD`: `train + val + test`

The active dataloader worker policy is capped at `4` workers for both training and linear-probe evaluation.

## Behavioral Rules for Contributions

When editing the pipeline, preserve these expectations unless the change is intentional and fully documented:

- keep `main.py` as the active orchestration entrypoint
- keep the benchmark and stress streams aligned between baseline and federated modes when comparing final probe accuracy
- keep the active numeric path in `float32`
- keep device transfers explicit and avoid introducing silent mixed-device math
- keep adapter-only communication in the federated path
- keep the benchmark schedule download-friendly for a fresh environment
- do not reintroduce manual-setup datasets into the default publishable schedule
- update the docs whenever the stage order, split policy, evaluation logic, or tracked metrics change
- keep Python docstrings and high-signal inline comments aligned with the current execution flow

## Evaluation Expectations

The active pipeline uses one final benchmark-only linear-probe pass after training completes. Contributions should not reintroduce older intermediate evaluation branches unless they are intentionally restored everywhere.

The current evaluation flow is:

1. freeze the encoder,
2. disable MAE masking so full images are used,
3. extract frozen features for the benchmark datasets,
4. train one linear probe per dataset,
5. evaluate on the held-out split,
6. write one final comparison summary.

## Validation

At minimum, run:

```bash
python -m py_compile main.py src\__init__.py src\client.py src\server.py src\loss.py src\mae_with_adapter.py
```

If formatting and lint tools are available:

```bash
ruff check .
ruff format .
```

## Pull Requests

1. Create a focused branch from `main`.
2. Keep each change scoped to one pipeline improvement when possible.
3. Explain the experimental impact clearly in the PR description.
4. List the validation commands you ran.
5. Update documentation whenever the behavior changed.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
