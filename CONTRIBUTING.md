# Contributing to PODFCSSV

Thank you for considering contributing to **PODFCSSV**! This document provides guidelines for researchers and engineers looking to extend or improve the framework.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sathishkumar67/PODFCSSV.git
cd PODFCSSV
```

### 2. Set Up the Development Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies (with dev extras)
pip install -e ".[dev]"
```

### 3. Verify the Installation

```bash
python -c "from src import GPADLoss, ClientManager; print('Imports OK')"
```

### 4. Run the Pipeline

```bash
python main.py
```

This executes the full federated learning loop using `MockViTMAE` and synthetic data — no real datasets or checkpoints needed.

---

## Architecture Overview

Understanding the codebase requires familiarity with its four main modules:

| Module | Responsibility |
|---|---|
| `src/loss.py` | GPAD loss computation, adaptive thresholding, and anchor mask generation |
| `src/client.py` | Local training, per-embedding routing (anchored / local / novel), novelty buffer, K-Means clustering |
| `src/server.py` | Global prototype bank (merge-or-add with EMA), FedAvg weight aggregation |
| `src/mae_with_adapter.py` | Information-Bottleneck Adapter injection into frozen ViT-MAE blocks |

### Key Data Flow

```
Server broadcasts global prototypes + averaged weights
    → Client trains with MAE + GPAD (per-embedding routing)
        → Anchored embeddings: GPAD loss pulls toward global prototypes
        → Non-anchored, locally similar: EMA-update local prototype
        → Truly novel: Accumulate in novelty buffer → K-Means when full → Merge-or-Add
    → Client uploads local prototypes + adapter weights
→ Server merges prototypes (EMA merge-or-add) + averages weights (FedAvg)
```

---

## Development Workflow

### Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `ruff.toml`.

```bash
# Check for lint issues
ruff check .

# Auto-fix fixable issues
ruff check --fix .

# Format code
ruff format .
```

### Naming Conventions

Mathematical variable names (e.g., `K`, `N`, `D`, `X`, `Z`) follow standard research notation and are exempted from PEP 8 lowercase rules via `ruff.toml`. This is intentional — do not rename them.

### Docstrings

Every public class and method should have a researcher-grade docstring following NumPy/SciPy style with:
- **Summary line**: One-line description of what the component does.
- **Extended description** (optional): Mathematical formulations, algorithmic pipeline steps, and design rationale explaining *why* a particular approach was chosen.
- **Parameters section**: All arguments with types, defaults, valid ranges (as `Range: X–Y`), and behavioral descriptions. For example:
  ```
  merge_threshold : float
      Cosine similarity threshold for merging. Range: 0.5–0.85.
      Default: 0.7.
  ```
- **Returns section**: Return type, shape (for tensors), and description.
- **Notes section** (optional): Mathematical background, edge-case handling, and implementation details. Include tensor shape annotations (e.g., `[B, D]`) wherever applicable.

### Inline Comments

Inline comments should explain the **"why"**, not the **"what"**:
- Include tensor shape annotations: `# [B, K] similarity matrix`.
- Reference algorithmic context: `# EMA blend is not a unit vector → re-normalize`.
- Explain edge cases: `# Buffer may trigger with fewer samples than novelty_k`.
- Every hyperparameter used in code should reference its CONFIG key in a nearby comment.

### Hyperparameter Range Comments

All tunable hyperparameters in `CONFIG` (in `main.py`) include range comments for future tuning. When adding new hyperparameters:
1. Add the parameter to `CONFIG` with a descriptive comment and `Range: X–Y` or `Options: A, B, C`.
2. Wire it through the relevant component constructors.
3. Document it in the README.md CONFIG reference tables.

---

## Adding New Features

### Adding a New Loss Component

1. Implement the loss class in `src/loss.py`.
2. Add any new hyperparameters to `CONFIG` in `main.py` with descriptive comments.
3. Wire the loss into `FederatedClient.train_epoch()` in `src/client.py`.
4. Update `src/__init__.py` for public import access.

### Adding a New Routing Strategy

1. The per-embedding routing logic lives in `FederatedClient.train_epoch()` and `_route_non_anchored()`.
2. To add a new routing branch, modify the decision tree after the anchor mask check.
3. Document the routing decision clearly in docstrings.

### Adding a New Aggregation Strategy

1. Server-side aggregation happens in `src/server.py`.
2. `GlobalPrototypeBank` handles prototype merging; `FederatedModelServer` handles weight averaging.
3. To change aggregation (e.g., weighted FedAvg), modify `aggregate_weights()`.

---

## Submitting Changes

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. Ensure code passes `ruff check .` with no errors.
4. Run the full pipeline `python main.py` and confirm `Pipeline Finished Successfully.`
5. Open a **Pull Request** with a description of:
   - What changed and why.
   - Any new hyperparameters added to `CONFIG`.
   - Any new dependencies added.
   - How you tested the changes.

---

## Reporting Issues

Please use [GitHub Issues](https://github.com/sathishkumar67/PODFCSSV/issues) with:

- A clear title and description.
- Steps to reproduce the problem.
- Your environment (OS, Python version, PyTorch version, CUDA version).
- Relevant log output.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
