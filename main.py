"""
Federated Continual Self-Supervised Learning — Main Orchestrator.

This script is the top-level entry point that simulates the complete
lifecycle of a Federated Learning (FL) system designed for Continual
Self-Supervised Learning. It combines Masked Autoencoders (MAE) as the
self-supervised pretext task with Gated Prototype Anchored Distillation
(GPAD) as a forgetting-prevention mechanism.

System Architecture
-------------------
The system consists of three logical actors:

1. **Server** (central coordinator):
   - Maintains a Global Prototype Bank — a dynamically growing collection
     of L2-normalized prototype vectors on the unit hypersphere, where each
     vector represents a distinct visual concept discovered across the
     federation.
   - Aggregates client model weights via Federated Averaging (FedAvg),
     computing the element-wise arithmetic mean of all client state dicts.

2. **Clients** (edge devices):
   - Each client holds a private local dataset and an independent deep copy
     of the global model.
   - Privacy-preserving: clients NEVER share raw data — only compact
     prototype vectors and model weights are communicated over the network.
   - Training uses per-embedding routing to classify each feature vector as
     either "anchored" (known concept → GPAD loss) or "non-anchored"
     (novel concept → local prototype update or novelty buffer).

3. **Orchestrator** (this script):
   - Manages the round-based communication loop between server and clients.
   - Handles broadcasting, collection, and state updates.
   - Centrally defines ALL hyperparameters in the CONFIG dictionary.

Training Phases (per communication round)
------------------------------------------
Round 1 — Initialization:
    Clients train with MAE reconstruction loss only. No global knowledge
    exists yet: GPAD is skipped. After training, each client clusters its
    feature space via Spherical K-Means to produce the initial local
    prototypes.

Round > 1 — Continual Learning:
    Clients train with MAE + GPAD. The GPAD loss regularizes each client's
    evolving feature space against the current global prototype bank,
    preventing catastrophic forgetting while allowing plasticity for new
    visual concepts.

Within each round, the following five steps execute sequentially:

    Step A — Broadcast:   Server sends current global prototypes to all clients.
    Step B — Training:    Clients train locally (MAE-only or MAE+GPAD).
    Step C — Extraction:  Round 1: K-Means prototype generation.
                          Round >1: Retrieve online-maintained prototypes.
    Step D — Aggregation: Server merges local prototypes (Merge-or-Add + EMA)
                          and averages model weights (FedAvg).
    Step E — Update:      Load the FedAvg weights into the base model for the
                          next round.

Simulation Details
------------------
- **Data**: Uses a synthetic ``TensorDataset`` with random noise to simulate
  image embeddings. No real images or pretrained checkpoints are needed for
  pipeline testing.
- **Model**: Uses ``MockViTMAE`` — a lightweight stand-in for
  ``ViTMAEForPreTraining`` that replicates the essential interfaces without
  requiring HuggingFace Transformers or checkpoint downloads.
- **Execution**: Supports both sequential (CPU) and parallel (multi-GPU)
  modes. GPU mode uses ``ThreadPoolExecutor`` with strict 1:1 Client-GPU
  mapping.

Hyperparameter Centralization
-----------------------------
Every tunable value across the entire pipeline is centralized in the
``CONFIG`` dictionary defined at the module level. This avoids magic numbers
scattered across source files and enables straightforward hyperparameter
sweeps. Each entry includes a descriptive comment and valid range for tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import List, Dict, Any

# ==========================================================================
# Project Imports
# ==========================================================================
# Import the project's server, client, and loss modules. These contain the
# core FL components: prototype bank, FedAvg server, client manager, and
# the GPAD distillation loss.
from src.server import GlobalPrototypeBank, FederatedModelServer, run_server_round, GlobalModel
from src.client import ClientManager
from src.loss import GPADLoss


# ==========================================================================
# Logging Configuration
# ==========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DistillFed")


# ==========================================================================
# HYPERPARAMETERS & CONFIGURATION
#
# Every tunable value across the entire pipeline is centralized here.
# This avoids magic numbers scattered in source files and makes
# hyperparameter sweeps straightforward.
#
# Each entry includes:
#   - A descriptive comment explaining the parameter's role.
#   - The valid range (as a comment) for future hyperparameter tuning.
#   - The default value chosen based on initial experimentation.
# ==========================================================================

CONFIG = {
    # ── System ────────────────────────────────────────────────────────────────
    # Random seed for reproducibility (set to None to disable)
    "seed": 42,

    # How many federated clients to simulate in this run
    "num_clients": 2,

    # Total number of server-client communication rounds to simulate.
    # Each round consists of: broadcast -> train -> extract -> aggregate -> update.
    "num_rounds": 5,

    # Number of local training epochs each client runs per round
    # Range: 1–10
    "local_epochs": 1,

    # Number of GPUs available (0 = CPU-only sequential mode)
    # This value is auto-detected at runtime if CUDA is available
    "gpu_count": 0,

    # Primary computation device for training and inference.
    # Set to "cuda" to use the first available GPU, or "cuda:N" for a
    # specific GPU. Auto-updated at runtime when CUDA is detected.
    "device": "cpu",

    # Floating-point precision used for model parameters and input tensors.
    # torch.float32 (single precision) is the safe default.
    # torch.bfloat16 offers ~2x speedup on Ampere+ GPUs with minimal
    # accuracy loss for most vision models.
    "dtype": torch.float32,

    # ── Model & Data ─────────────────────────────────────────────────────────
    # Pretrained model name/path for HuggingFace ViT-MAE backbone
    "pretrained_model_name": "facebook/vit-mae-base",

    # Dimensionality of the feature embedding space
    # (small value for mock simulation; real ViT-Base uses 768)
    "embedding_dim": 32,

    # Number of synthetic samples in the mock dataset (ignored in production)
    "num_samples": 20,

    # Mini-batch size for local training and prototype extraction
    "batch_size": 4,

    # Whether to shuffle the DataLoader between epochs
    "dataloader_shuffle": True,

    # ── Adapter (mae_with_adapter.py) ────────────────────────────────────────
    # Bottleneck dimension of the IBA adapters injected into the ViT encoder.
    # Smaller = fewer params / faster, larger = more capacity.
    # Typical values: 32–128. At dim=64 with ViT-Base the adapters add ~1 %
    # trainable parameters.
    "adapter_bottleneck_dim": 64,

    # Dropout rate for IBA adapters (regularization)
    # Range: 0.0–0.5
    "adapter_dropout": 0.0,

    # ── Global Prototype Management (server.py) ──────────────────────────────
    # Server-side global merge threshold: cosine similarity required to merge
    # a local prototype into an existing global one via EMA
    # Range: 0.5–0.85
    "merge_threshold": 0.7,

    # Server-side EMA alpha for global prototype updates
    # Lower values = slower, more stable updates
    # Range: 0.01–0.2
    "server_ema_alpha": 0.05,

    # Maximum capacity of the global prototype bank.
    # New prototypes are not added once this limit is reached.
    # Range: 20–200
    "max_global_prototypes": 50,

    # ── GPAD Distillation Loss (loss.py) ─────────────────────────────────────
    # Base similarity threshold for global anchoring in GPAD
    # Higher = stricter gating, fewer anchors activated
    # Range: 0.3–0.7
    "gpad_base_tau": 0.5,

    # Sigmoid gate temperature for steepness control in GPAD
    # Lower = sharper (near step-function), higher = smoother
    # Range: 0.05–0.5
    "gpad_temp_gate": 0.1,

    # Uncertainty scaling factor for the entropy-based adaptive threshold
    # Higher lambda = threshold rises more steeply as assignment entropy increases
    # Range: 0.1–0.5
    "gpad_lambda_entropy": 0.3,

    # Temperature for the soft assignment distribution in GPAD
    # Controls sharpness of the softmax used in entropy calculation
    # Range: 0.05–0.5
    "gpad_soft_assign_temp": 0.1,

    # Numerical epsilon for GPAD loss computation (prevents div-by-zero)
    "gpad_epsilon": 1e-8,

    # ── Client Local Training (client.py) ────────────────────────────────────
    # Number of prototype centroids each client generates via K-Means (Round 1)
    # Range: 5–50
    "k_init_prototypes": 10,

    # Optimizer learning rate for local client training
    "client_lr": 1e-4,

    # AdamW weight decay for L2 regularization
    "client_weight_decay": 0.05,

    # Local merge threshold: cosine-similarity for online EMA prototype updates.
    # Only samples more similar than this to their nearest local prototype
    # trigger an update — prevents noisy refinements.
    # Range: 0.4–0.8
    "client_local_update_threshold": 0.6,

    # EMA interpolation factor for local non-anchored updates and
    # local buffer centroid merges.
    # 0 = no update, 1 = full replacement.
    # Range: 0.05–0.3
    "client_local_ema_alpha": 0.1,

    # ── GPAD Loss Weighting ──────────────────────────────────────────────────
    # Weight of GPAD distillation loss relative to the main reconstruction loss.
    # total_loss = mae_loss + lambda_proto * gpad_loss.
    # Range: 0.001–0.1
    "lambda_proto": 0.01,

    # ── Novelty Buffer (client.py) ───────────────────────────────────────────
    # Novelty buffer capacity before triggering clustering.
    # Number of truly novel embeddings (failing both global and local
    # threshold checks) to accumulate before triggering a fresh K-Means
    # clustering to discover new visual concepts.
    # Options: 128, 256, 512
    "novelty_buffer_size": 256,

    # K for the buffer K-Means clustering.
    # This is independent of k_init_prototypes (Round 1 full clustering).
    # Range: 3–10
    "novelty_k": 5,

    # ── K-Means (client.py) ──────────────────────────────────────────────────
    # Maximum number of K-Means iterations before stopping
    "kmeans_max_iters": 100,

    # Convergence tolerance for K-Means (centroid shift below this = converged)
    "kmeans_tol": 1e-4,
}


# ==========================================================================
# MOCK MODEL — Lightweight stand-in for ViTMAEForPreTraining
#
# These classes replicate the essential interfaces of HuggingFace's
# ViTMAEForPreTraining model, enabling full pipeline testing without
# downloading a 300MB+ checkpoint or installing the `transformers` library.
# ==========================================================================


class _MockViTEncoder(nn.Module):
    """
    Mock ViT encoder that simulates ``model.vit`` in ViTMAEForPreTraining.

    In the real HuggingFace model, ``model.vit`` is a ``ViTMAEModel`` whose
    ``forward()`` returns a ``BaseModelOutput`` with a ``.last_hidden_state``
    attribute of shape ``[B, L, D]``, where L is the number of visible
    (non-masked) patch tokens.

    This mock replicates that interface using a shared linear layer, so that
    ``FederatedClient._extract_features()`` works identically whether the
    underlying model is real or mock. The sequence dimension is set to 1
    (a single "token") for simplicity.

    Attributes
    ----------
    linear : nn.Linear
        A shared linear layer (same instance as ``MockViTMAE.encoder``) to
        ensure consistent weight sharing between ``model.forward()`` and
        ``model.vit.forward()`` calls.
    """

    def __init__(self, linear: nn.Linear):
        """
        Initialize the mock ViT encoder.

        Parameters
        ----------
        linear : nn.Linear
            A pre-existing linear layer to share with the parent MockViTMAE.
        """
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor, **kwargs) -> object:
        """
        Simulate a ViTMAEModel forward pass.

        Applies the shared linear transformation and wraps the output in an
        object with a ``.last_hidden_state`` attribute to match the real
        ViT encoder's output format.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, D]`` (batch of feature vectors).

        Returns
        -------
        object
            An object with ``.last_hidden_state`` of shape ``[B, 1, D]``.
            The sequence length is 1 (a single mock patch token).
        """
        # Apply the shared linear transformation: [B, D] → [B, D]
        feat = self.linear(x)

        # Wrap in an output object mimicking HuggingFace's BaseModelOutput.
        class EncoderOutput:
            pass

        out = EncoderOutput()
        # Add a fake sequence dimension: [B, D] → [B, 1, D]
        # This allows mean-pooling in _extract_features() to be a no-op,
        # keeping the feature extraction code generic.
        out.last_hidden_state = feat.unsqueeze(1)
        return out


class MockViTMAE(nn.Module):
    """
    Lightweight stand-in for ``ViTMAEForPreTraining``.

    Used to test the full Federated Continual Learning pipeline without
    requiring a real pretrained checkpoint (300MB+) or the HuggingFace
    ``transformers`` library. This mock replicates the three essential
    interfaces that the pipeline code depends on:

    1. ``model.forward(x)`` → returns object with ``.loss`` and ``.hidden_states``
    2. ``model.vit``        → exposes an encoder whose output has ``.last_hidden_state``
    3. ``model.config``     → provides ``.hidden_size`` attribute

    The mock "loss" is computed as the mean absolute value of the encoder
    output, providing a simple differentiable scalar that enables gradient
    flow for testing the backward pass.

    Attributes
    ----------
    encoder : nn.Linear
        The core linear layer acting as the "encoder". Shape: [D, D].
    head : nn.Identity
        Placeholder for a downstream projection head (identity = no-op).
    config : object
        Mock configuration object with ``.hidden_size = dim``, matching
        HuggingFace's model config interface.
    vit : _MockViTEncoder
        Mock ViT encoder exposing ``.last_hidden_state`` in its output,
        matching the real ``model.vit`` interface used by
        ``FederatedClient._extract_features()``.
    """

    def __init__(self, dim: int = 32):
        """
        Initialize the mock ViT-MAE model.

        Parameters
        ----------
        dim : int
            Mock embedding dimension. Default: 32 (small for fast testing;
            real ViT-Base uses 768).
        """
        super().__init__()

        # Core linear layer acting as the "encoder".
        self.encoder = nn.Linear(dim, dim)

        # Identity head — placeholder for downstream projection.
        self.head = nn.Identity()

        # Mock config matching HuggingFace's model.config interface.
        self.config = type("Config", (), {"hidden_size": dim})()

        # Expose .vit to match ViTMAEForPreTraining's structure.
        # FederatedClient._extract_features() accesses model.vit directly
        # to get encoder-only representations (without the decoder).
        self.vit = _MockViTEncoder(self.encoder)

    def forward(
        self, x: torch.Tensor, output_hidden_states: bool = False, **kwargs
    ) -> object:
        """
        Simulate a ViTMAEForPreTraining forward pass.

        Runs the input through the linear encoder and produces a mock
        reconstruction loss (mean absolute value) and hidden states.
        The loss is differentiable, enabling full backward-pass testing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, D]`` (batch of feature vectors).
        output_hidden_states : bool
            Accepted for API compatibility with HuggingFace models.
            Does not change behavior in the mock.

        Returns
        -------
        object
            An object with two attributes:
            - ``.loss``: Scalar tensor — mean absolute value of encoder output.
              Serves as a simple differentiable proxy for MAE reconstruction loss.
            - ``.hidden_states``: List containing the encoder features as a
              ``[B, 1, D]`` tensor, mimicking the real ViT's multi-layer output.
        """
        # Forward through the mock encoder: [B, D] → [B, D]
        feat = self.encoder(x)

        class Output:
            pass

        out = Output()

        # Mock reconstruction loss: mean(|encoder_output|). This provides
        # a simple, non-trivial differentiable scalar for gradient testing.
        out.loss = feat.abs().mean()

        # Mock hidden states: list of (B, 1, D) tensors. In the real ViT,
        # this would contain one entry per encoder layer.
        out.hidden_states = [feat.unsqueeze(1)]
        return out


# ==========================================================================
# MAIN ORCHESTRATOR
# ==========================================================================


def main():
    """
    Run the complete Federated Continual Self-Supervised Learning simulation.

    This function is the top-level orchestrator that executes the full FL
    pipeline end-to-end. It performs four major phases:

    Phase 1 — Environment Setup:
        Detect available GPUs, configure the execution mode (sequential CPU
        or parallel multi-GPU), and set the random seed for reproducibility.

    Phase 2 — Component Initialization:
        Instantiate all FL components from the centralized CONFIG:
        (a) GlobalPrototypeBank — server-side prototype aggregation.
        (b) FederatedModelServer — FedAvg weight aggregation.
        (c) MockViTMAE — base model template (or real ViT-MAE in production).
        (d) ClientManager — spawns N independent federated clients.
        (e) GPADLoss — distillation loss module for Rounds > 1.

    Phase 3 — Data Setup:
        Create a synthetic TensorDataset and DataLoaders. In production, each
        client would have its own private image dataset; here all clients
        share the same mock data for pipeline validation.

    Phase 4 — Federated Training Loop:
        For each round r = 1, ..., num_rounds:
            (A) Broadcast global prototypes to all clients (skip Round 1).
            (B) Clients train locally — MAE only (Round 1) or MAE+GPAD (Round > 1).
            (C) Extract local prototypes — K-Means (Round 1) or online protos (Round > 1).
            (D) Server aggregates prototypes (Merge-or-Add + EMA) and weights (FedAvg).
            (E) Load aggregated weights into the base model for the next round.
    """
    logger.info("Initializing Federated Continual Learning Pipeline...")

    # ── Phase 1: Environment Setup ────────────────────────────────────────────
    # Automatically detect available CUDA GPUs and configure the execution
    # mode. Multi-GPU mode enables parallel client training with strict 1:1
    # Client-GPU mapping. CPU mode falls back to sequential execution.
    if torch.cuda.is_available():
        CONFIG["gpu_count"] = torch.cuda.device_count()
        CONFIG["device"] = "cuda"
        logger.info(
            f"Detected {CONFIG['gpu_count']} GPU(s). "
            f"Device set to '{CONFIG['device']}'. "
            f"Parallel mode enabled if Clients <= GPUs."
        )
    else:
        CONFIG["device"] = "cpu"
        logger.info(
            f"No CUDA GPUs found. Device='{CONFIG['device']}' (Sequential Mode)."
        )

    logger.info(f"Dtype: {CONFIG['dtype']}")

    # Set random seed for reproducibility. This ensures deterministic weight
    # initialization and data generation across runs with the same seed.
    if CONFIG["seed"] is not None:
        torch.manual_seed(CONFIG["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CONFIG["seed"])
        logger.info(f"Random seed set to {CONFIG['seed']}")

    # ── Phase 2: Component Initialization ─────────────────────────────────────
    # All components read their hyperparameters from the centralized CONFIG
    # dictionary, ensuring consistency and enabling easy sweeps.

    # 2A. Server-Side Global Prototype Bank
    # Manages the shared knowledge base of visual concepts. Local prototypes
    # from all clients are merged into this bank each round using the
    # Merge-or-Add strategy with EMA. The bank lives on the configured
    # device and has a capacity limit (max_global_prototypes).
    proto_bank = GlobalPrototypeBank(
        embedding_dim=CONFIG["embedding_dim"],
        merge_threshold=CONFIG["merge_threshold"],
        ema_alpha=CONFIG["server_ema_alpha"],
        device=CONFIG["device"],
        max_prototypes=CONFIG["max_global_prototypes"],
    )

    # 2B. Server-Side Model Aggregator (FedAvg)
    # Computes the element-wise arithmetic mean of all client model state
    # dicts to produce a global consensus model. Stateless — no
    # hyperparameters needed.
    fed_server = FederatedModelServer()

    # 2C. Global Model Template
    # In production, this would be a real ViTMAEForPreTraining model loaded
    # from CONFIG["pretrained_model_name"]. Here we use MockViTMAE — a
    # lightweight stand-in that replicates the essential model interfaces
    # without requiring checkpoint downloads or the transformers library.
    base_model = MockViTMAE(dim=CONFIG["embedding_dim"])

    # 2D. Client Manager
    # Factory that spawns N independent FederatedClient instances, each with
    # a deep copy of the base model. Handles device assignment (1:1 GPU
    # mapping when available) and dispatches training commands either
    # sequentially (CPU) or in parallel (multi-GPU via ThreadPoolExecutor).
    # All client-side hyperparameters are forwarded from CONFIG.
    client_manager = ClientManager(
        base_model=base_model,
        num_clients=CONFIG["num_clients"],
        gpu_count=CONFIG["gpu_count"],
        dtype=CONFIG["dtype"],
        optimizer_kwargs={
            "lr": CONFIG["client_lr"],
            "weight_decay": CONFIG["client_weight_decay"],
        },
        local_update_threshold=CONFIG["client_local_update_threshold"],
        local_ema_alpha=CONFIG["client_local_ema_alpha"],
        lambda_proto=CONFIG["lambda_proto"],
        novelty_buffer_size=CONFIG["novelty_buffer_size"],
        novelty_k=CONFIG["novelty_k"],
        kmeans_max_iters=CONFIG["kmeans_max_iters"],
        kmeans_tol=CONFIG["kmeans_tol"],
    )

    # 2E. GPAD Distillation Loss
    # The Gated Prototype Anchored Distillation loss module. Used from
    # Round 2 onwards to regularize client features against the global
    # prototype bank, preventing catastrophic forgetting while allowing
    # local plasticity for novel concepts.
    gpad_loss = GPADLoss(
        base_tau=CONFIG["gpad_base_tau"],
        temp_gate=CONFIG["gpad_temp_gate"],
        lambda_entropy=CONFIG["gpad_lambda_entropy"],
        soft_assign_temp=CONFIG["gpad_soft_assign_temp"],
        epsilon=CONFIG["gpad_epsilon"],
    )

    # ── Phase 3: Data Setup (Mock) ────────────────────────────────────────────
    # Create a synthetic dataset with random noise to simulate feature
    # embeddings. In production, each client would have its own private
    # image dataset (e.g., a shard of ImageNet or a domain-specific corpus).
    dataset = TensorDataset(
        torch.randn(CONFIG["num_samples"], CONFIG["embedding_dim"])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=CONFIG["dataloader_shuffle"],
    )

    # For simulation simplicity, all clients share the same dataloader.
    # In a real federated system, each client's DataLoader would wrap a
    # distinct, private dataset that never leaves the edge device.
    dataloaders = [dataloader] * CONFIG["num_clients"]

    # ── Phase 4: Federated Training Loop ──────────────────────────────────────
    # Round 1 starts with no global prototypes → pure MAE training.
    # Subsequent rounds use MAE + GPAD for continual learning.
    global_protos = None

    for round_idx in range(1, CONFIG["num_rounds"] + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"STARTING ROUND {round_idx}/{CONFIG['num_rounds']}")
        logger.info(f"{'='*40}")

        # ── Step A: Server Broadcast ─────────────────────────────────────
        # Send the current global prototype bank to all clients so they can
        # use it for GPAD anchoring. In Round 1, no prototypes exist yet.
        if round_idx > 1:
            logger.info(
                f"Broadcasting {len(global_protos)} Global Prototypes to Clients."
            )

        # ── Step B: Client Local Training ────────────────────────────────
        # Each client trains on its private data for one epoch:
        #   Round 1:  Loss = MAE reconstruction loss only.
        #   Round >1: Loss = MAE + lambda_proto × GPAD (anchored embeddings).
        logger.info(">> Clients Training...")
        losses = client_manager.train_round(
            dataloaders,
            global_prototypes=global_protos,
            gpad_loss_fn=gpad_loss,
        )
        logger.info(f"Client Losses: {losses}")

        # ── Step C: Local Prototype Extraction ───────────────────────────
        # Round 1:  Run full K-Means from scratch to produce initial
        #           client prototypes (k_init_prototypes clusters).
        # Round >1: Prototypes are maintained online via per-embedding
        #           routing (EMA updates + novelty buffer clustering).
        #           Retrieve them with get_local_prototypes().
        client_payloads = []

        if round_idx == 1:
            # --- Round 1: Full K-Means prototype initialization ---
            logger.info(">> Generating Initial Local Prototypes (K-Means)...")
            for i, client in enumerate(client_manager.clients):
                local_protos = client.generate_prototypes(
                    dataloader, K_init=CONFIG["k_init_prototypes"]
                )
                # Move weights to CPU for server-side aggregation (avoids
                # GPU memory retention across communication boundaries).
                weights = {
                    k: v.cpu() for k, v in client.model.state_dict().items()
                }
                payload = {
                    "client_id": f"client_{i}",
                    "protos": local_protos.cpu(),
                    "weights": weights,
                }
                client_payloads.append(payload)
        else:
            # --- Round > 1: Collect online-maintained prototypes ---
            logger.info(">> Collecting Local Prototypes (from routing/buffer)...")
            for i, client in enumerate(client_manager.clients):
                local_protos = client.get_local_prototypes()
                weights = {
                    k: v.cpu() for k, v in client.model.state_dict().items()
                }
                payload = {
                    "client_id": f"client_{i}",
                    "weights": weights,
                }
                # Include prototypes only if the client has generated them.
                if local_protos is not None:
                    payload["protos"] = local_protos.cpu()
                    logger.info(
                        f"  Client {i}: {local_protos.shape[0]} local protos, "
                        f"buffer={len(client.novelty_buffer)}"
                    )
                else:
                    logger.info(f"  Client {i}: No local prototypes yet")
                client_payloads.append(payload)

        # ── Step D: Server Aggregation ───────────────────────────────────
        # The server receives all client payloads and executes two tasks:
        #   (1) Merge local prototypes into the Global Prototype Bank using
        #       the Merge-or-Add strategy with EMA updates.
        #   (2) Average client model weights using FedAvg (element-wise mean
        #       of all client state dicts).
        logger.info(">> Server Aggregation...")
        server_result = run_server_round(
            proto_manager=proto_bank,
            model_server=fed_server,
            client_payloads=client_payloads,
        )

        # Extract the updated global state from the server's result.
        global_protos = server_result["global_prototypes"]
        global_weights = server_result["global_weights"]

        # ── Step E: Global Model Update ──────────────────────────────────
        # Load the FedAvg-aggregated weights into the base model. In the
        # next round, each client will receive a deep copy of this updated
        # model (via ClientManager), ensuring all clients start from the
        # latest global consensus.
        base_model.load_state_dict(global_weights)

        # Log round completion with the current global bank size.
        if global_protos is not None:
            logger.info(
                f"Round {round_idx} Complete. "
                f"Global Bank Size: {global_protos.shape[0]}"
            )
        else:
            logger.error("Round Complete but Global Protos is None!")

    logger.info("\nPipeline Finished Successfully.")


# ==========================================================================
# Entry Point
# ==========================================================================

if __name__ == "__main__":
    main()