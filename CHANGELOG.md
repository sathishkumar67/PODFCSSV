# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-02-27

### Added

- **New Hyperparameters in CONFIG** (`main.py`): 10 previously hardcoded or missing values are now centralized in the `CONFIG` dictionary with descriptive comments and tuning ranges:
  - `seed` (42) — Random seed for reproducibility.
  - `local_epochs` (1) — Number of local training epochs per round. Range: 1–10.
  - `dataloader_shuffle` (True) — Whether to shuffle the DataLoader between epochs.
  - `pretrained_model_name` ("facebook/vit-mae-base") — HuggingFace model identifier for the ViT-MAE backbone.
  - `adapter_dropout` (0.0) — Dropout rate for IBA adapters. Range: 0.0–0.5.
  - `max_global_prototypes` (50) — Maximum capacity of the global prototype bank. Range: 20–200.
  - `gpad_soft_assign_temp` (0.1) — Temperature for the soft assignment distribution in GPAD. Range: 0.05–0.5.
  - `gpad_epsilon` (1e-8) — Numerical epsilon for GPAD loss computation.
  - `kmeans_max_iters` (100) — Maximum K-Means iterations before forced termination.
  - `kmeans_tol` (1e-4) — Convergence tolerance for K-Means centroid shift.

- **Global Prototype Bank Capacity** (`src/server.py`): `GlobalPrototypeBank` now accepts a `max_prototypes` parameter. Once the bank reaches this limit, novel prototypes are rejected (merges still occur). This prevents unbounded bank growth in long-running federations.

- **Configurable GPAD Internals** (`src/loss.py`): `GPADLoss` now accepts `soft_assign_temp` and `epsilon` as constructor parameters (previously class-level constants), enabling external control via `CONFIG`.

- **Configurable K-Means** (`src/client.py`): `FederatedClient._kmeans()` now uses `self.kmeans_max_iters` and `self.kmeans_tol` from CONFIG instead of hardcoded values. These parameters are threaded through `ClientManager` → `FederatedClient`.

### Changed

- **8 Hyperparameter Defaults Updated** (`main.py`): Based on initial experimentation and tuning guidance, the following defaults were recalibrated:
  - `k_init_prototypes`: 5 → **10** (Range: 5–50)
  - `novelty_k`: 20 → **5** (Range: 3–10)
  - `merge_threshold`: 0.85 → **0.7** (Range: 0.5–0.85)
  - `gpad_lambda_entropy`: 0.1 → **0.3** (Range: 0.1–0.5)
  - `client_local_update_threshold`: 0.7 → **0.6** (Range: 0.4–0.8)
  - `lambda_proto`: 1.0 → **0.01** (Range: 0.001–0.1)
  - `novelty_buffer_size`: 500 → **256** (Options: 128, 256, 512)
  - `server_ema_alpha`: 0.1 → **0.05** (Range: 0.01–0.2)

- **Seed Setup** (`main.py`): Added explicit `torch.manual_seed()` and `torch.cuda.manual_seed_all()` calls based on `CONFIG["seed"]` for reproducibility.

### Documentation

- **Researcher-Grade Docstring Rewrite**: All module-level, class-level, and method-level docstrings across `src/loss.py`, `src/server.py`, `src/client.py`, and `main.py` were completely rewritten to provide:
  - Detailed mathematical formulations (GPAD loss equation, EMA blending, adaptive threshold derivation).
  - Per-stage pipeline explanations with data flow and tensor shapes.
  - Design rationale for every architectural choice (e.g., why sequential processing in prototype merging, why re-normalization after EMA blending).
  - Parameter documentation following NumPy/SciPy style with types, defaults, valid ranges, and behavioral descriptions.
  - References to relevant literature (FedAvg, Prototypical Networks, Masked Autoencoders).

- **Researcher-Grade Inline Comments**: All inline comments across 4 source files were rewritten to explain the "why" behind each operation, not just the "what". Each comment includes tensor shape notation, algorithmic context, and edge-case explanations.

- **Updated `README.md`**: CONFIG reference tables updated to reflect all new defaults and newly added hyperparameters.

- **Updated `CONTRIBUTING.md`**: Documentation style guidelines updated to reflect researcher-grade standards with range comments and design rationale requirements.

- **Version Bump**: `pyproject.toml` version bumped from `0.2.0` to `0.3.0`.

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
