# Federated Continual Learning (FCL) Framework

This repository implements a robust **Federated Continual Learning** pipeline designed to prevent catastrophic forgetting in distributed environments. It leverages **Prototype Anchored Distillation (GPAD)** and **Masked Autoencoders (MAE)** with efficient **Adapter** tuning.

## 🚀 Key Features

* **Federated Learning**: Decentralized training across multiple clients (simulated).
* **Continual Learning**: Prevents forgetting by anchoring local models to a global "Prototype Bank".
* **GPAD Loss**: A novel loss function that uses an adaptive gating mechanism to distill knowledge only when the model is confident.
* **Online Prototype Updates**: Clients update their local prototypes on-the-fly using Exponential Moving Average (EMA) during training, ensuring stability.
* **Efficient Tuning**: Uses **Information Bottleneck Adapters** to fine-tune a frozen ViT-MAE backbone, minimizing communication overhead.

## 📂 Project Structure

```
├── main.py                 # Entry point: Orchestrates the entire training pipeline
├── src/
│   ├── server.py           # Server logic: Aggregation (FedAvg) and Global Prototype Management
│   ├── client.py           # Client logic: Local training loops and Prototype Generation (K-Means)
│   ├── loss.py             # GPAD Loss implementation with adaptive thresholding
│   └── mae_with_adapter.py # Model architecture: ViT-MAE with trainable Adapters
└── requirements.txt        # Dependencies
```

## 🛠️ Installation

Ensure you have Python 3.8+ and PyTorch installed.

```bash
pip install -r requirements.txt
# or manually:
pip install torch torchvision transformers numpy
```

## ▶️ Usage

Run the main simulation script:

```bash
python main.py
```

### Configuration

You can adjust hyperparameters directly in `main.py`:

* `num_clients`: Number of federated clients.
* `num_rounds`: Number of communication rounds.
* `embedding_dim`: Dimension of the prototype vectors.
* `merge_threshold`: Cosine similarity threshold for merging global prototypes.

## 🧠 How It Works

1. **Initialization**: The server initializes a global model (ViT-MAE) and an empty Prototype Bank.
2. **Local Training**:
    * **Round 1**: Clients train using pure **MAE Reconstruction Loss**.
    * **Round > 1**: Clients train using **MAE + GPAD Loss**. The GPAD loss anchors current embeddings to the nearest global prototype if the similarity is high (confident match).
    * **Online Update**: Simultaneously, clients update their *local* prototypes using EMA whenever a sample strongly matches an existing local cluster.
3. **Prototype Generation**: After training, clients run K-Means on their data to extract fresh local prototypes.
4. **Aggregation**:
    * **Weights**: The server averages adapter weights from all clients.
    * **Prototypes**: The server merges new local prototypes into the Global Bank, updating similar ones via EMA or adding new concepts.

## 📜 License

MIT
