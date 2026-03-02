# PODFCSSV — Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision

A framework for **Federated Continual Self-Supervised Learning** that combines a frozen Masked Autoencoder (MAE) backbone, parameter-efficient Information-Bottleneck Adapters, per-embedding anchored routing, a novelty buffer for online concept discovery, and a global unsupervised prototype bank — enabling privacy-preserving, communication-efficient visual representation learning across distributed clients under non-IID, sequentially arriving data.

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
- [References](#references)
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

1. Loading a **pre-trained ViT-MAE backbone** and injecting lightweight **Information-Bottleneck Adapters** (~1% trainable parameters, backbone frozen).
2. Communicating only compact **prototype vectors** (K-Means centroids of local features) instead of raw data.
3. Using **GPAD loss** with per-embedding routing to anchor local representations against a global prototype bank, preventing forgetting.
4. Employing a **Novelty Buffer** on each client that accumulates genuinely unseen patterns and discovers new visual concepts via triggered K-Means clustering.

---

## Key Features

- **Parameter-Efficient Fine-Tuning** — IBA adapters with zero-initialized up-projections for stable, identity-first training.
- **Privacy-Preserving** — Only prototype vectors and adapter weights leave the client; raw data and novelty buffer contents stay local.
- **Continual Learning** — EMA-based global prototype bank grows dynamically as new visual concepts emerge across tasks.
- **Adaptive Distillation** — Entropy-aware gating in GPAD suppresses noisy anchoring from ambiguous prototype assignments.
- **Per-Embedding Routing** — Each embedding is dynamically classified as anchored (→ GPAD loss), locally known (→ EMA update), or truly novel (→ novelty buffer).
- **Novelty Buffer with Merge-or-Add** — Accumulated novel samples are clustered when a threshold is reached; resulting centroids are merged into existing local prototypes via EMA if similar, or added as new concepts if distinct.
- **Multi-GPU Parallelism** — 1:1 client-GPU mapping with `ThreadPoolExecutor` for true concurrent training.
- **Dtype Consistency** — All tensors respect the centralized `CONFIG["dtype"]` setting with no implicit float32 casts.
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
┌──────────────────────────────────────────────────────────────┐
│                     CLIENT i (Edge Device)                    │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Frozen ViT-MAE Backbone + IBA Adapters (trainable)      ││
│  └──────────────────────────────────────────────────────────┘│
│                          │                                    │
│              ┌───────────┼───────────────┐                    │
│              ▼           ▼               ▼                    │
│        MAE Loss    GPAD Loss     Per-Embedding Router         │
│      (reconstruct) (distill)   ┌─────────┴──────────┐        │
│              │           │     │ Local EMA  │ Novelty │        │
│              │           │     │  Update    │ Buffer  │        │
│              │           │     └────────────┴─────────┘        │
│              └───────────┼───────────┘                         │
│                          ▼                                     │
│            Local Prototypes → Upload to Server                 │
└───────────────────────────────────────────────────────────────┘
```

### Training Phases (per round)

| Phase | Step | Round 1 | Round ≥ 2 |
|---|---|---|---|
| **A** | Broadcast | Server sends initial model (no prototypes) | Server sends global prototypes + averaged weights |
| **B** | Local Training | MAE loss only (no global knowledge yet) | MAE + λ × GPAD loss with per-embedding routing |
| **C** | Prototype Extraction | Full K-Means on all local embeddings | Use live local prototypes (maintained via EMA + buffer clustering) |
| **D** | Server Aggregation | Concatenate all client prototypes into initial bank (no merge logic) | EMA merge-or-add prototypes + FedAvg weight averaging |
| **E** | Global Update | First global model and prototype bank ready | Updated model and bank broadcast to all clients |

### Per-Embedding Routing (Round ≥ 2)

For each embedding in a training batch, the client applies a three-stage decision tree:

```
                    ┌─────────────────────────────┐
                    │    Compute Anchor Mask       │
                    │  (GPAD adaptive threshold)   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              Anchored               Not Anchored
           (sim ≥ τ_adaptive)     (sim < τ_adaptive)
                    │                     │
           Apply GPAD Loss        Check Local Prototypes
           (pull toward global)          │
                                ┌────────┴────────┐
                                ▼                 ▼
                          Local Match        No Match
                       (sim ≥ τ_local)    (sim < τ_local)
                                │                 │
                        EMA Update         Add to Novelty
                     Local Prototype          Buffer
                                              │
                                    ┌─────────┴─────────┐
                                    │  Buffer ≥ thresh?  │
                                    └─────────┬─────────┘
                                              │ Yes
                                    ┌─────────┴──────────┐
                                    │  Fresh K-Means     │
                                    │  Merge-or-Add      │
                                    │  into local protos │
                                    └────────────────────┘
```

---

## Repository Structure

```
PODFCSSV/
├── main.py                      # Federated Learning orchestrator & CONFIG
├── src/
│   ├── __init__.py              # Clean public API imports
│   ├── mae_with_adapter.py      # IBA Adapter + ViT block wrapper + injection
│   ├── client.py                # FederatedClient + ClientManager
│   ├── server.py                # GlobalPrototypeBank + FedAvg + GlobalModel
│   └── loss.py                  # GPAD distillation loss + anchor mask
├── docs/
│   ├── diagrams/                # Architecture diagrams (PNG)
│   ├── svg/                     # Architecture diagrams (SVG)
│   └── markdowns/               # Complete Pipeline Guide
├── train.ipynb                  # Interactive training notebook
├── pyproject.toml               # Project metadata & build config
├── requirements.txt             # Python dependencies
├── ruff.toml                    # Linter/formatter configuration
├── CHANGELOG.md                 # Release history
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore patterns
```

---

## Installation

### Prerequisites

- Python ≥ 3.9
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

# Or install as editable package (includes dev dependencies)
pip install -e ".[dev]"
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Core deep learning framework |
| `torchvision` | ≥ 0.15.0 | Vision utilities and transforms |
| `transformers` | ≥ 4.30.0 | ViT-MAE backbone from Hugging Face |
| `numpy` | ≥ 1.21.0 | Numerical computing |
| `scikit-learn` | ≥ 1.0.0 | Additional ML utilities |

---

## Quick Start

### Run the Full Pipeline

```bash
python main.py
```

This loads the pre-trained `facebook/vit-mae-base` checkpoint, injects IBA adapters, and runs the full federated training loop on Tiny ImageNet. The backbone is frozen — only adapter parameters (~1%) are trained.

**Expected output:**

```
Round 1: MAE only  → loss ≈ 0.45 → 5 protos/client → initial global bank
Round 2: MAE+GPAD → loss ≈ 1.10 → per-embedding routing active
Round 3: MAE+GPAD → loss ≈ 1.10 → novelty buffer accumulating
Round 4: MAE+GPAD → loss ≈ 1.10 → buffer grows, local protos stable
Round 5: MAE+GPAD → loss ≈ 1.10 → pipeline finished successfully
```

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

| Parameter | Default | Range | Description |
|---|---|---|---|
| `seed` | 42 | — | Random seed for reproducibility (set to `None` to disable) |
| `num_clients` | 2 | — | Number of simulated federated clients |
| `num_rounds` | 5 | — | Total server-client communication rounds |
| `local_epochs` | 1 | 1–10 | Number of local training epochs per round |
| `gpu_count` | 0 | — | GPUs available (auto-detected; 0 = CPU mode) |
| `dtype` | `float32` | — | Floating-point precision (`float32` or `bfloat16`) |
| `dataloader_shuffle` | True | — | Whether to shuffle the DataLoader between epochs |

### Model & Adapter

| Parameter | Default | Range | Description |
|---|---|---|---|
| `pretrained_model_name` | `facebook/vit-mae-base` | — | HuggingFace model identifier for the ViT-MAE backbone |
| `adapter_bottleneck_dim` | 256 | 32–256 | Information bottleneck dimension of IBA adapters |
| `embedding_dim` | 768 | — | Hidden size of ViT-Base encoder (must match backbone) |
| `image_size` | 224 | — | Input image resolution (224×224 for ViT-Base) |

### Server — Global Prototype Bank

| Parameter | Default | Range | Description |
|---|---|---|---|
| `merge_threshold` | 0.15 | 0.05–0.4 | Cosine similarity threshold to merge vs. add a prototype (low because pre-trained ViT-MAE features are dense on the unit sphere) |
| `server_ema_alpha` | 0.1 | 0.01–0.2 | EMA interpolation factor for global prototype updates |
| `max_global_prototypes` | 500 | 20–200 | Maximum capacity of the global prototype bank |

### GPAD Loss

| Parameter | Default | Range | Description |
|---|---|---|---|
| `gpad_base_tau` | 0.4 | 0.3–0.7 | Base similarity threshold for confident anchoring |
| `gpad_temp_gate` | 0.1 | 0.05–0.5 | Sigmoid gate temperature (lower = sharper decision boundary) |
| `gpad_lambda_entropy` | 0.3 | 0.1–0.5 | Entropy penalty scaling factor (raises threshold for uncertain samples) |
| `gpad_soft_assign_temp` | 0.1 | 0.05–0.5 | Temperature for the soft assignment distribution in entropy calculation |
| `gpad_epsilon` | 1e-8 | — | Numerical epsilon for GPAD loss computation (prevents div-by-zero) |
| `lambda_proto` | 0.01 | 0.001–0.1 | GPAD loss weight: `total = MAE + λ × GPAD` |

### Client — Local Training & Prototype Management

| Parameter | Default | Range | Description |
|---|---|---|---|
| `k_init_prototypes` | 50 | 5–50 | Number of prototype centroids per client (Round 1 K-Means) |
| `client_lr` | 1e-4 | — | AdamW optimizer learning rate |
| `client_weight_decay` | 0.05 | — | AdamW L2 regularization weight decay |
| `client_local_update_threshold` | 0.2 | 0.05–0.3 | Cosine similarity threshold for EMA prototype updates and buffer merge decisions (low for pre-trained ViT-MAE) |
| `client_local_ema_alpha` | 0.1 | 0.05–0.3 | EMA interpolation factor for online prototype refinement |
| `kmeans_max_iters` | 100 | — | Maximum K-Means iterations before forced termination |
| `kmeans_tol` | 1e-4 | — | K-Means convergence tolerance (centroid shift below this = converged) |

### Client — Novelty Buffer

| Parameter | Default | Range / Options | Description |
|---|---|---|---|
| `novelty_buffer_size` | 256 | 128, 256, 512 | Number of novel embeddings to accumulate before triggering fresh K-Means |
| `novelty_k` | 10 | 3–10 | K for buffer K-Means clustering (independent of `k_init_prototypes`) |

---

## Module Reference

### `src/mae_with_adapter.py`

Implements parameter-efficient fine-tuning via **Information-Bottleneck Adapters (IBA)**.

| Component | Description |
|---|---|
| `IBA_Adapter` | Bottleneck MLP: Linear(D→d) → GELU → Linear(d→D) → Dropout, with zero-initialized up-projection for identity-first training |
| `ViTBlockWithAdapter` | Wraps a frozen encoder layer + adapter, handling HuggingFace return type polymorphism |
| `inject_adapters()` | Freezes backbone, injects adapters into every encoder layer, prints parameter audit |

### `src/loss.py`

Implements the **Gated Prototype Anchored Distillation (GPAD)** loss.

| Component | Description |
|---|---|
| `GPADLoss.forward()` | Full GPAD loss pipeline: similarity → adaptive threshold → gating → weighted distance. Returns `0.0` safely when the global bank is empty |
| `GPADLoss.compute_anchor_mask()` | Returns a boolean mask `[B]` indicating which embeddings are anchored to global prototypes (used by per-embedding routing) |

### `src/client.py`

Defines federated client-side training, per-embedding routing, and prototype management.

| Component | Description |
|---|---|
| `FederatedClient.train_epoch()` | Per-embedding routing: anchored → GPAD loss, non-anchored → local check → EMA or buffer |
| `FederatedClient._route_non_anchored()` | Routes non-anchored embeddings: updates matching local prototypes via EMA, or adds to novelty buffer |
| `FederatedClient._cluster_novelty_buffer()` | Clusters buffered novel embeddings via K-Means; merges similar centroids into existing local prototypes (EMA) or adds new ones |
| `FederatedClient.generate_prototypes()` | Full-dataset K-Means prototype extraction (used in Round 1 only) |
| `FederatedClient.get_local_prototypes()` | Returns current live local prototypes without re-clustering (used in Round ≥ 2) |
| `ClientManager` | Orchestrates N clients with parallel (multi-GPU via ThreadPoolExecutor) or sequential (CPU) execution |

### `src/server.py`

Server-side aggregation logic.

| Component | Description |
|---|---|
| `GlobalPrototypeBank` | Merge-or-Add prototype bank with EMA updates. Dynamically grows as new concepts emerge. Re-normalizes after merges to maintain unit-sphere geometry |
| `FederatedModelServer` | Standard FedAvg: element-wise arithmetic mean of client state dictionaries, preserving the original `CONFIG["dtype"]` without implicit float32 casting |
| `run_server_round()` | Single-call server round: merges prototypes + averages weights |
| `GlobalModel` | Wrapper for loading real ViTMAEForPreTraining with adapter injection |

---

## Algorithm Details

### 1. Information-Bottleneck Adapter (IBA)

Each frozen transformer block is paired with a lightweight adapter that learns a residual correction:

```
H_adapted = H + Dropout(W_up · σ(W_down · H))
```

- **Down-projection** compresses D → d (information bottleneck, retaining only task-relevant features)
- **Up-projection** is zero-initialized → adapter outputs Δh = 0 at initialization (identity-first training)
- Only adapter parameters are trainable (~1% of total for ViT-Base)

### 2. Gated Prototype Anchored Distillation (GPAD)

The GPAD loss enforces alignment between local embeddings and global prototypes:

```
L_GPAD(z) = Gate(z) × ‖z - v*‖

where:
  v*          = argmax_v cos(z, v)              (best-matching global prototype)
  p_j         = softmax(cos(z, v_j) / τ_p)     (soft assignment distribution)
  H_norm(z)   = -Σ p_j log(p_j) / log(M)       (normalized Shannon entropy)
  τ(z)        = τ_base + λ_entropy · H_norm(z)  (entropy-adaptive threshold)
  I_anchor    = 1 if cos(z, v*) ≥ τ(z), else 0  (binary anchor indicator)
  α_match     = σ((cos(z, v*) - τ(z)) / T)      (soft sigmoid gate)
  Gate(z)     = I_anchor × α_match               (final gating weight)
```

The entropy penalty `H_norm` raises the threshold for uncertain samples, ensuring only confident matches contribute to the distillation loss.

### 3. Per-Embedding Routing

Each embedding in a batch is routed through a three-stage decision tree:

1. **Global Anchor Check**: If the embedding is sufficiently similar to a global prototype (passes adaptive threshold), apply GPAD loss. Total loss = MAE + λ × GPAD.
2. **Local Prototype Check**: If not globally anchored, compare against local prototypes. If similar enough, update the closest local prototype via EMA. Only MAE loss applies.
3. **Novelty Buffer**: If the embedding matches neither global nor local prototypes, it is truly novel. Store it in the buffer. Only MAE loss applies.

### 4. Novelty Buffer Clustering with Merge-or-Add

When the novelty buffer accumulates enough samples (≥ `novelty_buffer_size`):

1. Run a **fresh K-Means** (K = `novelty_k`) on the buffered embeddings to find cluster centroids.
2. For each new centroid, compute similarity to existing local prototypes:
   - **If similar** (sim ≥ `local_update_threshold`): Merge into the matched prototype via EMA.
   - **If distinct** (sim < `local_update_threshold`): Add as a new local prototype.
3. Clear the buffer.

This prevents duplicate prototypes while allowing genuine new concept discovery.

### 5. Global Prototype Bank (Merge-or-Add with EMA)

**Round 1 (bank empty):** All incoming local prototypes from every client are simply concatenated to form the initial global bank. No Merge-or-Add logic is applied.

**Round ≥ 2:** For each incoming local prototype `p` uploaded by a client:

- If `max cos(p, G) ≥ merge_threshold`: **Merge** via EMA — `G_best ← (1-α)·G_best + α·p`, then re-normalise to unit sphere.
- If `max cos(p, G) < merge_threshold`: **Add** — append `p` as a new global prototype (subject to `max_prototypes` capacity).

This allows the prototype bank to automatically discover new visual concepts while refining existing ones.

### 6. Spherical K-Means

Client-side prototype extraction uses K-Means on L2-normalised embeddings with cosine similarity (dot product) as the distance metric:

- Random data-point initialisation (Forgy)
- Empty-cluster re-seeding with random data points
- Centroids are L2-normalised before computing the convergence shift, ensuring the check measures true displacement on the unit sphere
- Convergence check: centroid shift < `kmeans_tol`
- Automatic clamping of K to N when samples < clusters

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
