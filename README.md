# PODFCSSV — Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision

A framework for **Federated Continual Self-Supervised Learning** that combines Masked Autoencoders (MAE) with Gated Prototype Anchored Distillation (GPAD) to enable privacy-preserving, communication-efficient visual representation learning across distributed clients.

<p align="center">
  <img src="docs/diagrams/proposed architecture.png" alt="Proposed Architecture" width="700"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Module Reference](#module-reference)
- [Algorithm Details](#algorithm-details)
- [License](#license)

---

## Overview

Training a shared visual model across distributed edge devices faces three simultaneous challenges:

| Challenge | Description |
|---|---|
| **Data Privacy** | Raw images can never leave the client device |
| **Catastrophic Forgetting** | New tasks overwrite previously learned representations |
| **Communication Cost** | Transmitting full model weights is bandwidth-prohibitive |

**PODFCSSV** addresses all three by:

1. Training a **frozen ViT-MAE backbone** with lightweight **Information-Bottleneck Adapters** (~1% trainable parameters).
2. Communicating only compact **prototype vectors** (K-Means centroids of local features) instead of raw data.
3. Using **GPAD loss** to anchor local representations against a global prototype bank, preventing forgetting.

---

## Key Features

- **Parameter-Efficient Fine-Tuning** — IBA adapters with zero-initialized up-projections for stable training.
- **Privacy-Preserving** — Only prototype vectors and adapter weights leave the client; raw data stays local.
- **Continual Learning** — EMA-based global prototype bank grows dynamically as new visual concepts emerge.
- **Adaptive Distillation** — Entropy-aware gating in GPAD suppresses noisy anchoring from ambiguous assignments.
- **Multi-GPU Parallelism** — 1:1 client-GPU mapping with `ThreadPoolExecutor` for true concurrent training.
- **Mock Simulation** — Full pipeline runs without downloading model checkpoints or real datasets.

---

## Architecture

The system operates in round-based communication cycles between a central **Server** and N distributed **Clients**:

```
┌──────────────────────────────────────────────────────────────┐
│                        SERVER                                │
│                                                              │
│  ┌──────────────────────┐    ┌───────────────────────────┐   │
│  │ Global Prototype Bank│    │ FedAvg Model Aggregator   │   │
│  │ (EMA merge-or-add)   │    │ (Arithmetic mean weights) │   │
│  └──────────┬───────────┘    └──────────┬────────────────┘   │
└─────────────┼───────────────────────────┼────────────────────┘
              │ Broadcast                 │ Broadcast
              ▼ Global Prototypes        ▼ Global Weights
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT i (Edge Device)                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Frozen ViT-MAE Backbone + IBA Adapters (trainable)     ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│              ┌───────────┼───────────┐                       │
│              ▼           ▼           ▼                       │
│        MAE Loss    GPAD Loss    EMA Proto                    │
│      (reconstruct) (distill)   (refine)                      │
│              └───────────┼───────────┘                       │
│                          ▼                                   │
│                K-Means Clustering → Local Prototypes          │
│                          │                                   │
│                    Upload to Server                           │
└──────────────────────────────────────────────────────────────┘
```

### Training Phases (per round)

| Phase | Step | Description |
|---|---|---|
| **A** | Broadcast | Server sends global prototypes to all clients |
| **B** | Local Training | Clients train with MAE loss (Round 1) or MAE + GPAD (Round > 1) |
| **C** | Prototype Extraction | Clients cluster local features via Spherical K-Means |
| **D** | Server Aggregation | EMA prototype merging + FedAvg weight averaging |
| **E** | Global Update | Updated model and prototype bank ready for next round |

---

## Repository Structure

```
PODFCSSV/
├── main.py                      # Federated Learning orchestrator & CONFIG
├── src/
│   ├── __init__.py
│   ├── mae_with_adapter.py      # IBA Adapter + ViT block wrapper + injection
│   ├── client.py                # FederatedClient + ClientManager
│   ├── server.py                # GlobalPrototypeBank + FedAvg + GlobalModel
│   └── loss.py                  # GPAD distillation loss
├── docs/
│   ├── diagrams/                # Architecture diagrams (PNG)
│   ├── svg/                     # Architecture diagrams (SVG)
│   └── markdowns/               # Complete Pipeline Guide
├── train.ipynb                  # Interactive training notebook
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── .gitignore
```

---

## Installation

### Prerequisites

- Python ≥ 3.8
- CUDA ≥ 11.7 (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/sathishkumar67/PODFCSSV.git
cd PODFCSSV

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Core deep learning framework |
| `torchvision` | ≥ 0.15.0 | Vision utilities |
| `transformers` | ≥ 4.30.0 | ViT-MAE backbone from Hugging Face |

---

## Quick Start

### Run the Full Pipeline (Mock Simulation)

```bash
python main.py
```

This runs the complete federated learning loop using a lightweight `MockViTMAE` model and synthetic data — no checkpoint downloads or real datasets required.

**Expected output:**

- Initialization logs for each client
- Per-round training losses
- Global prototype bank growth
- `Pipeline Finished Successfully.`

### Run with Real ViT-MAE

```python
from transformers import ViTMAEForPreTraining
from src.mae_with_adapter import inject_adapters

# Load pre-trained backbone
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

# Inject adapters (freezes backbone, adds ~1% trainable params)
model = inject_adapters(model, bottleneck_dim=64)
```

### Interactive Notebook

```bash
jupyter notebook train.ipynb
```

---

## Configuration

All hyperparameters are centralized in the `CONFIG` dictionary in `main.py`:

### System

| Parameter | Default | Description |
|---|---|---|
| `num_clients` | 2 | Number of simulated federated clients |
| `num_rounds` | 5 | Total communication rounds |
| `gpu_count` | 0 | GPUs available (auto-detected; 0 = CPU) |
| `dtype` | `float32` | Precision (`float32` or `bfloat16`) |

### Adapter

| Parameter | Default | Description |
|---|---|---|
| `adapter_bottleneck_dim` | 64 | Bottleneck dim (32–128 typical) |

### Server — Prototype Management

| Parameter | Default | Description |
|---|---|---|
| `merge_threshold` | 0.85 | Cosine similarity to merge vs. add a prototype |
| `server_ema_alpha` | 0.1 | EMA factor for global prototype updates |

### GPAD Loss

| Parameter | Default | Description |
|---|---|---|
| `gpad_base_tau` | 0.5 | Base threshold for confident anchoring |
| `gpad_temp_gate` | 0.1 | Sigmoid gate temperature |
| `gpad_lambda_entropy` | 0.1 | Entropy penalty scaling factor |

### Client Training

| Parameter | Default | Description |
|---|---|---|
| `k_init_prototypes` | 5 | Local prototypes per client per round |
| `client_lr` | 1e-4 | Optimizer learning rate |
| `client_weight_decay` | 0.05 | AdamW weight decay |
| `client_local_update_threshold` | 0.7 | EMA update similarity threshold |
| `client_local_ema_alpha` | 0.1 | Online prototype EMA factor |

---

## Module Reference

### `src/mae_with_adapter.py`

Implements parameter-efficient fine-tuning via **Information-Bottleneck Adapters (IBA)**.

| Component | Description |
|---|---|
| `IBA_Adapter` | Bottleneck MLP: Linear(D→d) → GELU → Linear(d→D) → Dropout, with zero-initialized up-projection |
| `ViTBlockWithAdapter` | Wraps a frozen encoder layer + adapter, handling HuggingFace return type polymorphism |
| `inject_adapters()` | Freezes backbone, injects adapters into every encoder layer, prints parameter audit |

### `src/client.py`

Defines federated client-side training and prototype generation.

| Component | Description |
|---|---|
| `FederatedClient` | Single edge device: local training (MAE ± GPAD), feature extraction, K-Means prototype generation, online EMA prototype refinement |
| `ClientManager` | Orchestrates N clients with parallel (multi-GPU via ThreadPoolExecutor) or sequential (CPU) execution |

### `src/server.py`

Server-side aggregation logic.

| Component | Description |
|---|---|
| `GlobalPrototypeBank` | Merge-or-Add prototype bank with EMA updates. Dynamically grows as new concepts emerge |
| `FederatedModelServer` | Standard FedAvg: arithmetic mean of client state dictionaries |
| `run_server_round()` | One-call server round: merges prototypes + averages weights |
| `GlobalModel` | Wrapper for loading real ViTMAEForPreTraining with adapter injection |

### `src/loss.py`

| Component | Description |
|---|---|
| `GPADLoss` | Gated Prototype Anchored Distillation loss with entropy-adaptive thresholding and soft sigmoid gating |

---

## Algorithm Details

### 1. Information-Bottleneck Adapter (IBA)

```
H_out = H + Dropout(W_up · σ(W_down · H))
```

- **Down-projection** compresses D → d (information bottleneck)
- **Up-projection** is zero-initialized at step 0 → adapter outputs Δh = 0 (identity init)
- Only adapter parameters are trainable (~1% of total for ViT-Base)

### 2. Gated Prototype Anchored Distillation (GPAD)

```
L_gpad(z) = Gate(z) × ‖z - v*‖₂

where:
  v*    = argmax_v cos(z, v)           (best-matching global prototype)
  τ(z)  = τ_base + λ · H_norm(z)      (entropy-adaptive threshold)
  Gate  = σ((cos(z, v*) - τ(z)) / T)  (soft sigmoid gate)
```

The entropy penalty `H_norm` raises the threshold when prototype assignment is ambiguous, ensuring only confident matches contribute to the distillation loss.

### 3. Global Prototype Bank (Merge-or-Add with EMA)

For each incoming local prototype `p`:

- If `max cos(p, G) ≥ threshold`: **Merge** via EMA — `G_best ← (1-α)·G_best + α·p`, then re-normalize
- If `max cos(p, G) < threshold`: **Add** — append `p` as a new global prototype

This allows the prototype bank to automatically discover new visual concepts while refining existing ones.

### 4. Spherical K-Means

Client-side prototype extraction uses K-Means on L2-normalized embeddings with cosine similarity as the distance metric. Features include:

- Random data-point initialization
- Empty-cluster re-seeding
- Convergence check (centroid shift < 1e-4)

---

## References

1. Houlsby, N. et al., *"Parameter-Efficient Transfer Learning for NLP"*, ICML 2019.
2. Tishby, N. et al., *"The Information Bottleneck Method"*, 2000.
3. He, K. et al., *"Masked Autoencoders Are Scalable Vision Learners"*, CVPR 2022.
4. McMahan, B. et al., *"Communication-Efficient Learning of Deep Networks from Decentralized Data"* (FedAvg), AISTATS 2017.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Copyright © 2025 Sathish Kumar R
