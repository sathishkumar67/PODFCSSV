# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2025-02-17

### Added

- **GPAD Loss** (`src/loss.py`): Gated Prototype Anchored Distillation with entropy-adaptive thresholding and sigmoid gating.
- **Server Module** (`src/server.py`): `GlobalPrototypeBank` (EMA-based merge-or-add), `FederatedModelServer` (FedAvg), and `GlobalModel` wrapper.
- **Client Module** (`src/client.py`): `FederatedClient` with MAE + GPAD training, online EMA prototype refinement, and Spherical K-Means extraction. `ClientManager` for multi-GPU orchestration.
- **Adapter Injection** (`src/mae_with_adapter.py`): `IBA_Adapter` bottleneck module and `inject_adapters()` for parameter-efficient ViT-MAE fine-tuning.
- **Main Orchestrator** (`main.py`): Full round-based federated pipeline with `MockViTMAE` for dependency-free testing.
- **Documentation**: Comprehensive `README.md`, architecture diagrams, and `Complete-Pipeline-Guide.md`.
