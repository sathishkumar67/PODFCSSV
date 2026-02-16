# Federated Continual Self-Supervised Learning (Simulated)

This repository implements a **Federated Continual Learning** pipeline that combines **Masked Autoencoders (MAE)** with **Prototype-based Knowledge Distillation**. The system allows distributed clients to learn continuously from their private data while sharing knowledge via a global prototype bank, preventing catastrophic forgetting without sharing raw data.

## 🚀 Key Features

- **Federated Architecture**: Decentralized training with a central server for aggregation.
- **Continual Learning**: Uses **Gated Prototype Anchored Distillation (GPAD)** to regularize local models using global knowledge.
- **Prototype Management**: implementing an **Exponential Moving Average (EMA)** strategies to merge and evolve global prototypes over time.
- **Phased Training**:
  - **Round 1**: Pure Self-Supervised Learning (MAE).
  - **Round >1**: MAE + GPAD (Distillation from Global Prototypes).
- **Simulation**: A `main.py` orchestrator that simulates the entire loop (Client -> Server -> Client) on CPU or GPU.

## 📂 Project Structure

```
├── main.py                # Entry point: Simulates the full Federated Learning loop
├── src
│   ├── client.py          # Federated Client logic (Training, K-Means Clustering)
│   ├── server.py          # Server logic (Global Prototype Bank, FedAvg Aggregation)
│   └── loss.py            # GPAD Loss function (Adaptive Thresholding, Gating)
└── README.md              # Project Documentation
```

## 🛠️ Architecture Overview

The system operates in communication rounds:

1. **Broadcast**: The Server sends the current **Global Prototypes** to all clients.
2. **Local Training**:
    - Clients train their local `ViT-MAE` model.
    - **Loss**: $L_{total} = L_{MAE} + \lambda \cdot L_{GPAD}$
    - $L_{GPAD}$ anchors local embeddings to the nearest global prototype if the match is confident.
3. **Prototype Generation**:
    - Clients run **K-Means** on their local feature space to generate *Local Prototypes*.
4. **Upload**: Clients upload *Local Prototypes* and *Model Weights* to the Server.
5. **Aggregation**:
    - **Prototypes**: Server merges local prototypes into the Global Bank using Cosine Similarity and EMA.
    - **Weights**: Server averages model weights using **FedAvg**.

## 📦 Installation

Ensure you have Python 3.8+ and PyTorch installed.

```bash
# Install PyTorch (Adjust for your CUDA version)
pip install torch torchvision

# Install Hugging Face Transformers (for ViT-MAE)
pip install transformers
```

## 🏃 Usage

To run the end-to-end simulation:

```bash
python main.py
```

### Configuration

You can modify the `CONFIG` dictionary in `main.py` to adjust hyperparameters:

```python
CONFIG = {
    "num_clients": 2,       # Number of clients
    "num_rounds": 5,        # Communication rounds
    "embedding_dim": 32,    # Feature dimension (mock)
    "merge_threshold": 0.85,# Similarity threshold for merging prototypes
    "ema_alpha": 0.1,       # EMA update factor
    "gpad_base_tau": 0.5,   # GPAD confidence threshold
}
```

## 🔍 Implementation Details

### GPAD Loss (`src/loss.py`)

Adaptive Distillation Loss that gates the regularization based on assignment uncertainty (Entropy).

- **High Entropy** (Uncertain match) → **High Threshold** → **Gate Closed** (No Loss).
- **Low Entropy** (Confident match) → **Low Threshold** → **Gate Open** (Anchor to Prototype).

### Global Prototype Bank (`src/server.py`)

- **Merge**: If a new local prototype represents a known concept (High Sim), it updates the existing global prototype via EMA.
- **Add**: If it represents a new concept (Low Sim), it is added to the bank.
