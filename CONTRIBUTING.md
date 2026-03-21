# Contributing to PODFCSSV

Thank you for contributing to **PODFCSSV**. This guide reflects the current repository layout and the current training and evaluation flow.

## Setup

Clone the repository:

```bash
git clone https://github.com/sathishkumar67/PODFCSSV.git
cd PODFCSSV
```

Create and activate an environment:

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick import check:

```bash
python -c "from src import GPADLoss, ClientManager; print('Imports OK')"
```

## Current Entry Points

Run the current scripts like this:

```bash
python main.py
python base.py
python evaluate.py path\to\federated_final_model.pt path\to\base_final_model.pt
```

Current intent of each entrypoint:

- `main.py`: federated continual learning with 2 clients, 6 datasets, and GPAD
- `base.py`: single-model continual learning with reconstruction loss only
- `evaluate.py`: frozen-feature linear-probe comparison between saved checkpoints

## Shared Architecture

The main shared modules are:

| Module | Responsibility |
|---|---|
| `src/mae_with_adapter.py` | freeze ViT-MAE and inject residual adapters |
| `src/loss.py` | GPAD loss, adaptive thresholds, anchor masks |
| `src/client.py` | client-side MAE training, routing, novelty buffering, local prototypes |
| `src/server.py` | global prototype merging and adapter-weight averaging |

## Documentation Style

Python files use a clear, step-by-step documentation style.

When updating docstrings or comments:

- explain what stage of the pipeline the code belongs to
- explain the order of the important steps
- explain what state persists across rounds or dataset transitions
- explain why a branch exists when the behavior is not obvious
- prefer implementation clarity over dense research prose

Inline comments should be used sparingly and should explain the purpose of a block, not narrate each line.

## Development Guidelines

### Linting and Formatting

```bash
ruff check .
ruff format .
```

### Device and Dtype Handling

- respect the shared `dtype` and `device` settings
- avoid hardcoded dtype casts unless they are genuinely required
- keep tensor-device movement explicit
- remember that the repo now validates that CUDA can execute a real kernel before using a GPU

### Keeping the Entry Points Aligned

If you change one entrypoint, check the others:

- `main.py` and `base.py` should keep using the same adapter-injected MAE architecture
- `evaluate.py` should stay compatible with the checkpoint layout written by `main.py` and `base.py`
- README and the pipeline guide should be updated whenever the pipeline behavior changes

## Submitting Changes

1. Create a branch from `main`.
2. Make focused changes with clear commit messages.
3. Run at least the relevant syntax, lint, or smoke checks.
4. Update documentation when pipeline behavior changes.
5. Open a pull request that explains what changed, why it changed, and how it was tested.

## Reporting Issues

When reporting issues, include:

- a clear description of the problem
- exact reproduction steps
- Python, PyTorch, and CUDA versions
- GPU type if CUDA is involved
- relevant logs or stack traces

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
