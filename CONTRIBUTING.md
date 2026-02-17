# Contributing to PODFCSSV

Thank you for considering contributing to **PODFCSSV**! This document provides guidelines to help you get started.

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

---

## Development Workflow

### Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint issues
ruff check .

# Auto-fix fixable issues
ruff check --fix .

# Format code
ruff format .
```

### Running the Pipeline

```bash
python main.py
```

---

## Submitting Changes

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. Ensure code passes `ruff check .` with no errors.
4. Open a **Pull Request** with a description of:
   - What changed and why.
   - Any new dependencies added.
   - How you tested the changes.

---

## Reporting Issues

Please use [GitHub Issues](https://github.com/sathishkumar67/PODFCSSV/issues) with:

- A clear title and description.
- Steps to reproduce the problem.
- Your environment (OS, Python version, PyTorch version).

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
