"""
Federated Continual Self-Supervised Learning — Main Orchestrator.

This script simulates the complete lifecycle of a Federated Learning (FL)
system designed for Continual Self-Supervised Learning using Masked
Autoencoders (MAE) and Global Prototype Anchored Distillation (GPAD).

System Overview
---------------
The system consists of three actors:

1. **Server** (central coordinator):
   - Maintains a Global Prototype Bank — a collection of prototype vectors
     that represent the visual concepts discovered across all clients.
   - Aggregates client model weights via FedAvg.

2. **Clients** (edge devices):
   - Each client holds a private local dataset and an independent copy
     of the global model.
   - Clients train locally and never share raw data — only compact
     prototype vectors and model weights are communicated.

3. **Orchestrator** (this script):
   - Manages the round-based communication loop between server and clients.
   - Handles broadcasting, collection, and state updates.

Training Phases (per round)
---------------------------
Round 1 — Initialization:
    Clients train using only the MAE reconstruction loss. No global
    knowledge exists yet, so GPAD is skipped.

Round > 1 — Continual Learning:
    Clients train with MAE + GPAD. The GPAD loss regularizes each client's
    feature space against the global prototypes, preventing catastrophic
    forgetting and feature drift.

Within each round the following steps occur:

    Step A — Broadcast:   Server sends current global prototypes to clients.
    Step B — Training:    Clients train on their local data (MAE or MAE+GPAD).
    Step C — Extraction:  Clients cluster their features via K-Means to
                          produce local prototypes.
    Step D — Aggregation: Server merges local prototypes (EMA) and averages
                          model weights (FedAvg).
    Step E — Update:      The global model and prototype bank are updated.

Simulation Details
------------------
- Data:      Uses a mock TensorDataset with random noise to simulate
             image embeddings (no real images needed for pipeline testing).
- Model:     Uses ``MockViTMAE`` by default (a lightweight stand-in for
             ViTMAEForPreTraining that requires no checkpoint download).
- Execution: Supports both sequential (CPU) and parallel (multi-GPU) modes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import List, Dict, Any

# Import the project's server, client, and loss modules
from src.server import GlobalPrototypeBank, FederatedModelServer, run_server_round, GlobalModel
from src.client import ClientManager
from src.loss import GPADLoss


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DistillFed")


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # --- System Configuration ---
    # How many federated clients to simulate in this run
    "num_clients": 2,

    # Total number of server-client communication rounds
    "num_rounds": 5,

    # Number of GPUs available (0 = CPU-only sequential mode)
    # This value is auto-detected at runtime if CUDA is available
    "gpu_count": 0,

    # --- Model & Data Configuration ---
    # Dimensionality of the feature embedding space
    # (small value for mock simulation; real ViT-Base uses 768)
    "embedding_dim": 32,

    # --- Global Prototype Management (Server-Side) ---
    # Cosine similarity threshold: if a new local prototype is this similar
    # to an existing global prototype, they are merged via EMA instead of
    # being added as a new entry
    "merge_threshold": 0.85,

    # Exponential Moving Average factor for updating global prototypes
    # Lower values = slower, more stable updates
    "ema_alpha": 0.1,

    # --- GPAD Distillation Loss ---
    # Base adaptive threshold for confident anchoring in GPAD
    "gpad_base_tau": 0.5,

    # Temperature for the soft gating mechanism in GPAD
    "gpad_temp_gate": 0.1,

    # --- Client Local Training ---
    # Number of prototype centroids each client generates per round
    # via K-Means clustering
    "k_init_prototypes": 5,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK MODEL (Fallback for minimal-dependency pipeline testing)
# ═══════════════════════════════════════════════════════════════════════════════

class _MockViTEncoder(nn.Module):
    """
    Mock ViT encoder that simulates ``model.vit`` in ViTMAEForPreTraining.

    The real ``model.vit`` is a ViTMAEModel whose forward() returns an
    object with a ``.last_hidden_state`` of shape ``(B, L, D)``. This mock
    replicates that interface using a simple linear layer, so that the
    client's ``_extract_features()`` method works without modification.

    Parameters
    ----------
    linear : nn.Linear
        A shared linear layer (same as the parent MockViTMAE's encoder)
        to ensure weight sharing between the mock's forward pass and
        the mock's .vit access.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor, **kwargs) -> object:
        """
        Simulate a ViTMAEModel forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, D)``.

        Returns
        -------
        object
            An object with ``.last_hidden_state`` of shape ``(B, 1, D)``
            (sequence length = 1 for simplicity).
        """
        feat = self.linear(x)

        class EncoderOutput:
            pass

        out = EncoderOutput()
        # Unsqueeze to add a fake sequence dimension: (B, D) -> (B, 1, D)
        out.last_hidden_state = feat.unsqueeze(1)
        return out


class MockViTMAE(nn.Module):
    """
    Lightweight stand-in for ``ViTMAEForPreTraining``.

    This mock model is used to test the full FL orchestration pipeline
    without downloading a real 300MB+ checkpoint. It replicates the
    essential interfaces that the pipeline depends on:

    - ``model.forward(x)``  returns an object with ``.loss`` and ``.hidden_states``
    - ``model.vit``         exposes an encoder with ``.last_hidden_state`` output
    - ``model.config``      provides a ``.hidden_size`` attribute

    Parameters
    ----------
    dim : int
        The mock embedding dimension (default: 32 for fast testing).
    """

    def __init__(self, dim: int = 32):
        super().__init__()

        # Core linear layer acting as the "encoder"
        self.encoder = nn.Linear(dim, dim)

        # Identity head (placeholder for downstream projection)
        self.head = nn.Identity()

        # Mock config object matching HuggingFace's model.config interface
        self.config = type("Config", (), {"hidden_size": dim})()

        # Expose .vit to match ViTMAEForPreTraining structure
        # This is required by FederatedClient._extract_features()
        self.vit = _MockViTEncoder(self.encoder)

    def forward(self, x: torch.Tensor, output_hidden_states: bool = False, **kwargs) -> object:
        """
        Simulate a ViTMAEForPreTraining forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, D)``.
        output_hidden_states : bool
            Whether to include hidden states in the output (accepted for
            API compatibility but does not change behavior here).

        Returns
        -------
        object
            An object with:
            - ``.loss``: A scalar tensor (mean absolute value of features).
            - ``.hidden_states``: A list containing the features as
              ``(B, 1, D)`` to mimic real ViT output.
        """
        # Run the simple linear encoder
        feat = self.encoder(x)

        class Output:
            pass

        out = Output()

        # Simulate MAE reconstruction loss as the mean absolute value
        out.loss = feat.abs().mean()

        # Simulate hidden states: list of tensors with shape (B, 1, D)
        out.hidden_states = [feat.unsqueeze(1)]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Run the complete Federated Continual Learning simulation.

    This function executes the following pipeline:

    1. **Environment Setup**: Detect GPUs and configure execution mode.
    2. **Component Initialization**: Create the prototype bank, model server,
       base model, client manager, and loss function.
    3. **Data Setup**: Create a mock dataset and dataloaders.
    4. **Training Loop**: Execute ``num_rounds`` of federated training, each
       consisting of broadcast, local training, prototype extraction,
       server aggregation, and global model update.
    """
    logger.info("Initializing Federated Continual Learning Pipeline...")

    # ── Step 1: Environment Setup ────────────────────────────────────────────
    # Automatically detect available GPUs for parallel client simulation
    if torch.cuda.is_available():
        CONFIG["gpu_count"] = torch.cuda.device_count()
        logger.info(f"Detected {CONFIG['gpu_count']} GPUs. Parallel mode enabled if Clients <= GPUs.")
    else:
        logger.info("Using CPU Only (Sequential Mode).")

    # ── Step 2: Component Initialization ─────────────────────────────────────

    # 2A. Server-Side Global Prototype Bank
    #     Manages the global knowledge base of visual concepts.
    #     New local prototypes are merged into this bank each round via EMA.
    proto_bank = GlobalPrototypeBank(
        embedding_dim=CONFIG["embedding_dim"],
        merge_threshold=CONFIG["merge_threshold"],
        ema_alpha=CONFIG["ema_alpha"],
        device="cpu"  # Aggregation on CPU to save GPU memory for training
    )

    # 2B. Server-Side Model Aggregator
    #     Handles Federated Averaging (FedAvg) of client model weights.
    fed_server = FederatedModelServer()

    # 2C. Global Model Template
    #     In production, this would be a real ViTMAEForPreTraining model.
    #     Here we use MockViTMAE for pipeline testing without heavy dependencies.
    base_model = MockViTMAE(dim=CONFIG["embedding_dim"])

    # 2D. Client Manager
    #     Creates N independent clients, each with a deep copy of the base model.
    #     Handles device assignment (1:1 GPU mapping when available).
    client_manager = ClientManager(
        base_model=base_model,
        num_clients=CONFIG["num_clients"],
        gpu_count=CONFIG["gpu_count"],
        dtype=torch.float32
    )

    # 2E. GPAD Distillation Loss
    #     Used from Round 2 onwards to regularize client features against
    #     the global prototype bank, preventing catastrophic forgetting.
    gpad_loss = GPADLoss(
        base_tau=CONFIG["gpad_base_tau"],
        temp_gate=CONFIG["gpad_temp_gate"]
    )

    # ── Step 3: Data Setup (Mock) ────────────────────────────────────────────
    # Create a synthetic dataset of 20 random vectors with 32 dimensions.
    # In production, each client would have its own real image dataset.
    dataset = TensorDataset(torch.randn(20, CONFIG["embedding_dim"]))
    dataloader = DataLoader(dataset, batch_size=4)

    # For simulation, all clients share the same dataloader.
    # In a real system, each client would have its own private data.
    dataloaders = [dataloader] * CONFIG["num_clients"]

    # ── Step 4: Training Loop ────────────────────────────────────────────────
    # Round 1 starts with no global prototypes (pure MAE training).
    global_protos = None

    for round_idx in range(1, CONFIG["num_rounds"] + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"STARTING ROUND {round_idx}/{CONFIG['num_rounds']}")
        logger.info(f"{'='*40}")

        # ── Step A: Server Broadcast ─────────────────────────────────────
        # The server sends the current global prototypes to all clients.
        # In Round 1, there are no prototypes yet, so this step is skipped.
        if round_idx > 1:
            logger.info(f"Broadcasting {len(global_protos)} Global Prototypes to Clients.")

        # ── Step B: Client Local Training ────────────────────────────────
        # Each client trains on its local data for one epoch.
        # Round 1: Loss = MAE only (no global knowledge yet).
        # Round > 1: Loss = MAE + GPAD (regularized by global prototypes).
        logger.info(">> Clients Training...")
        losses = client_manager.train_round(
            dataloaders,
            global_prototypes=global_protos,
            gpad_loss_fn=gpad_loss
        )
        logger.info(f"Client Losses: {losses}")

        # ── Step C: Local Prototype Extraction ───────────────────────────
        # After training, each client extracts feature embeddings from its
        # data, clusters them via K-Means, and produces K local prototypes.
        # These prototypes summarize the client's data distribution without
        # revealing any raw images (privacy-preserving).
        logger.info(">> Generating Local Prototypes (K-Means)...")
        client_payloads = []

        for i, client in enumerate(client_manager.clients):
            # Extract K local prototype centroids via K-Means
            local_protos = client.generate_prototypes(
                dataloader,
                K_init=CONFIG["k_init_prototypes"]
            )

            # Collect model weights (moved to CPU for transmission)
            weights = {k: v.cpu() for k, v in client.model.state_dict().items()}

            # Pack the client's upload payload
            payload = {
                "client_id": f"client_{i}",
                "protos": local_protos.cpu(),
                "weights": weights,
            }
            client_payloads.append(payload)

        # ── Step D: Server Aggregation ───────────────────────────────────
        # The server receives all client payloads and performs two tasks:
        #   1. Merge local prototypes into the Global Bank using EMA.
        #   2. Average client model weights using FedAvg.
        logger.info(">> Server Aggregation...")
        server_result = run_server_round(
            proto_manager=proto_bank,
            model_server=fed_server,
            client_payloads=client_payloads
        )

        # Extract the updated global state for the next round
        global_protos = server_result["global_prototypes"]
        global_weights = server_result["global_weights"]

        # ── Step E: Global Model Update ──────────────────────────────────
        # Load the newly averaged weights into the base model.
        # In the next round, clients will deep-copy this updated model.
        base_model.load_state_dict(global_weights)

        # Log round summary
        if global_protos is not None:
            logger.info(f"Round {round_idx} Complete. Global Bank Size: {global_protos.shape[0]}")
        else:
            logger.error("Round Complete but Global Protos is None!")

    logger.info("\nPipeline Finished Successfully.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
