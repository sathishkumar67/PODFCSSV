# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-02-26

### Added

- **Per-Embedding Routing** (`src/client.py`): Three-stage decision tree in `train_epoch` — each embedding is classified as anchored (→ GPAD loss), locally known (→ EMA update), or truly novel (→ novelty buffer).
- **Novelty Buffer** (`src/client.py`): Accumulates genuinely unseen embeddings that fail both global and local similarity checks. Triggers fresh K-Means clustering when buffer reaches configurable threshold (`novelty_buffer_size`).
- **Merge-or-Add Strategy** (`src/client.py`): Buffer clustering centroids are compared against existing local prototypes — merged via EMA if similar (`local_update_threshold`), appended as new prototypes if distinct.
- **Anchor Mask** (`src/loss.py`): New `compute_anchor_mask()` method on `GPADLoss` for per-embedding anchor/non-anchor classification.
- **Lambda Proto Weighting** (`main.py`, `src/client.py`): Configurable `lambda_proto` scales GPAD loss contribution: `total = MAE + λ × GPAD`.
- **New Hyperparameters** (`main.py`): `lambda_proto` (1.0), `novelty_buffer_size` (500), `novelty_k` (20) added to CONFIG.
- **Conditional Orchestration** (`main.py`): Round 1 uses full K-Means via `generate_prototypes()`; Round ≥ 2 uses `get_local_prototypes()` maintained by routing and buffer clustering.
- **`get_local_prototypes()`** (`src/client.py`): Returns current live prototypes without re-clustering.
- **Project Configuration**: Added `pyproject.toml`, `CONTRIBUTING.md`, `CHANGELOG.md`, `ruff.toml`.

### Fixed

- **Empty Prototype Guard** (`src/loss.py`): `GPADLoss.forward()` now returns `tensor(0.0)` when the global bank is empty instead of crashing on `max()` of a zero-sized dimension.
- **K-Means K > N** (`src/client.py`): `_kmeans()` now clamps K to the number of available samples, preventing `IndexError` when samples < clusters.
- **Prototype Bank Normalization** (`src/server.py`): Added defensive re-normalization of global prototypes before dot-product similarity to maintain unit-sphere geometry.

### Changed

- **`FederatedClient.__init__`**: Accepts new hyperparameters (`lambda_proto`, `novelty_buffer_size`, `novelty_k`).
- **`ClientManager.__init__`**: Passes through all new hyperparameters to each `FederatedClient` instance.
- **`train_epoch`**: Fully rewritten for per-embedding routing with anchored/non-anchored branching.
- **Training Loop** (`main.py`): Split into Round 1 (full K-Means) vs. Round ≥ 2 (live prototype retrieval) paths.
- **`README.md`**: Comprehensive rewrite with routing diagrams, novelty buffer documentation, and complete CONFIG reference tables.

---

## [0.1.0] — 2025-02-17

### Added

- **GPAD Loss** (`src/loss.py`): Gated Prototype Anchored Distillation with entropy-adaptive thresholding and sigmoid gating.
- **Server Module** (`src/server.py`): `GlobalPrototypeBank` (EMA-based merge-or-add), `FederatedModelServer` (FedAvg), and `GlobalModel` wrapper.
- **Client Module** (`src/client.py`): `FederatedClient` with MAE + GPAD training, online EMA prototype refinement, and Spherical K-Means extraction. `ClientManager` for multi-GPU orchestration.
- **Adapter Injection** (`src/mae_with_adapter.py`): `IBA_Adapter` bottleneck module and `inject_adapters()` for parameter-efficient ViT-MAE fine-tuning.
- **Main Orchestrator** (`main.py`): Full round-based federated pipeline with `MockViTMAE` for dependency-free testing.
- **Documentation**: Comprehensive `README.md`, architecture diagrams, and `Complete-Pipeline-Guide.md`.
