import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import List, Dict, Any

from src.server import GlobalPrototypeBank, FederatedModelServer, run_server_round, GlobalModel
from src.client import ClientManager
from src.loss import GPADLoss

"""
Federated Continual Learning Orchestrator.

This script simulates the entire lifecycle of a Federated Learning system
designed for Continual Self-Supervised Learning. It integrates the client-side
training logic (MAE + GPAD), the server-side aggregation (EMA Prototypes + FedAvg),
and manages the communication rounds between them.

Key Workflows:
1.  **System Initialization**: Sets up the Global Model, Prototype Bank, and Client Manager.
2.  **Round-Based Training Loop**:
    a.  **Broadcast**: Server sends global prototypes to clients (from Round 2 onwards).
    b.  **Local Training**: Clients train on their private data.
        -   Round 1: Pure Masked Autoencoding (MAE).
        -   Round > 1: MAE + GPAD (Distillation from Global Prototypes).
    c.  **Prototype Generation**: Clients cluster their features to find local prototypes.
    d.  **Aggregation**: Server merges local prototypes into the Global Bank (EMA) 
        and averages model weights (FedAvg).
    e.  **Global Update**: The global model and bank are updated for the next round.

Simulation Details:
-   Data: Uses mock TensorDataset (random noise) to simulate image embeddings.
-   Model: Uses `MockViTMAE` if transformers is not installed, or `ViTMAEForPreTraining`.
-   Execution: Supports both Sequential (CPU) and Parallel (Multi-GPU) simulation modes.
"""

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DistillFed")

# =============================================================================
# HYPERPARAMETERS & CONFIGURATION
# =============================================================================
CONFIG = {
    # --- System Configuration ---
    "num_clients": 2,       # Number of federated clients to simulate
    "num_rounds": 5,        # Total number of communication rounds
    "gpu_count": 0,         # 0 for CPU, >0 for GPU parallelization (Auto-detected below)
    
    # --- Model & Data Configuration ---
    "embedding_dim": 32,    # Dimensionality of the feature space (Small for mock simulation)
    
    # --- Global Prototype Management (Server) ---
    "merge_threshold": 0.85, # Cosine similarity threshold to merge local proto into global
    "ema_alpha": 0.1,       # Exponential Moving Average factor for updating prototypes
    
    # --- Distillation Loss Configuration (GPAD) ---
    "gpad_base_tau": 0.5,   # Base adaptive threshold for confident anchoring
    "gpad_temp_gate": 0.1,  # Temperature for the soft gating mechanism
    
    # --- Client Local Training ---
    "k_init_prototypes": 5, # Number of local prototypes each client generates per round
}

# =============================================================================
# MOCK MODEL (Fallback for minimal dependency execution)
# =============================================================================
class MockViTMAE(nn.Module):
    """
    A lightweight mock model acting as a stand-in for `ViTMAEForPreTraining`.
    
    Used for:
    1.  Testing the orchestration pipeline without downloading heavy checkpoints.
    2.  Verifying logic flow when `transformers` library is missing.
    
    Behaves like a standard HuggingFace model:
    -   Input: `x` (Batch, Dim) or `pixel_values`
    -   Output: Object with `loss` and `hidden_states` attributes.
    """
    def __init__(self, dim=32):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.head = nn.Identity()
        # Mock configuration object often accessed by HF utils
        self.config = type('Config', (), {'hidden_size': dim})()
        
    def forward(self, x, output_hidden_states=False, **kwargs):
        """
        Simulates a forward pass.
        
        Args:
            x (Tensor): Input tensor (Batch, Dim).
            output_hidden_states (bool): Whether to return hidden states (Required for GPAD).
        """
        # Simple linear transformation
        feat = self.encoder(x)
        
        # Create a dummy return object similar to ModelOutput
        class Output:
            pass
        out = Output()
        
        # Simulate MAE Loss (e.g., L1 norm of features just to have a scalar)
        out.loss = feat.abs().mean()
        
        # Simulate hidden states: [feat]
        # In real ViT, this is (Batch, SeqLen, Dim). 
        # Here our input is (B, D), so we unsqueeze to (B, 1, D) to mimic sequence length 1.
        out.hidden_states = [feat.unsqueeze(1)] 
        return out

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    """
    Main function to run the Federated Learning simulation.
    """
    logger.info("Initializing Federated Continual Learning Pipeline...")
    
    # 1. Environment Setup
    # Automatically detect available GPUs to enable parallel simulation
    if torch.cuda.is_available():
        CONFIG["gpu_count"] = torch.cuda.device_count()
        logger.info(f"Detected {CONFIG['gpu_count']} GPUs. Parallel mode enabled if Clients <= GPUs.")
    else:
        logger.info("Using CPU Only (Sequential Mode).")

    # 2. Instantiate Components
    
    # A. Server-Side Prototype Bank
    # Manages the global knowledge base (concepts)
    proto_bank = GlobalPrototypeBank(
        embedding_dim=CONFIG["embedding_dim"],
        merge_threshold=CONFIG["merge_threshold"],
        ema_alpha=CONFIG["ema_alpha"],
        device="cpu" # We perform aggregation on CPU to save GPU memory for training
    )
    
    # B. Server-Side Model Aggregator
    # Handles FedAvg logic
    fed_server = FederatedModelServer()
    
    # C. Global Model Initialization
    # In a full deployment, `GlobalModel` loads the real ViT-MAE.
    # Here, we instantiate the base model object manually to pass to clients.
    base_model = MockViTMAE(dim=CONFIG["embedding_dim"])
    
    # D. Client Manager
    # Spawns and manages the simulated clients (Edge Devices)
    client_manager = ClientManager(
        base_model=base_model,
        num_clients=CONFIG["num_clients"],
        gpu_count=CONFIG["gpu_count"]
    )
    
    # E. Loss Function
    # The GPAD loss used during the continual learning phase
    gpad_loss = GPADLoss(
        base_tau=CONFIG["gpad_base_tau"],
        temp_gate=CONFIG["gpad_temp_gate"]
    )

    # 3. Data Setup (Mock)
    # Simulating a dataset of 20 samples with 32 dimensions
    dataset = TensorDataset(torch.randn(20, CONFIG["embedding_dim"]))
    dataloader = DataLoader(dataset, batch_size=4)
    # Assign the same dataloader to all clients for this simulation
    dataloaders = [dataloader] * CONFIG["num_clients"]

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    global_protos = None # Round 1 starts with no global knowledge
    
    for round_idx in range(1, CONFIG["num_rounds"] + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"STARTING ROUND {round_idx}/{CONFIG['num_rounds']}")
        logger.info(f"{'='*40}")
        
        # --- A. SERVER BROADCAST ---
        # In a real system, the server sends the `global_weights` and `global_protos`
        # to selected clients here. In this simulation, we pass `global_protos` explicitly.
        if round_idx > 1:
            logger.info(f"Broadcasting {len(global_protos)} Global Prototypes to Clients.")
        
        # --- B. CLIENT LOCAL TRAINING ---
        # Clients train on their local data. 
        # Round 1: Only MAE. 
        # Round >1: MAE + GPAD (using the broadcasted prototypes).
        logger.info(">> Clients Training...")
        losses = client_manager.train_round(
            dataloaders,
            global_prototypes=global_protos,
            gpad_loss_fn=gpad_loss
        )
        logger.info(f"Client Losses: {losses}")
        
        # --- C. PROTOTYPE EXTRACTION ---
        # After training, clients identify representative features (Cluster Centroids)
        # to share with the server.
        logger.info(">> Generating Local Prototypes (K-Means)...")
        client_payloads = []
        
        for i, client in enumerate(client_manager.clients):
            # 1. Extract Local Protos
            local_protos = client.generate_prototypes(
                dataloader, 
                K_init=CONFIG["k_init_prototypes"]
            )
            
            # 2. Extract Weights
            # Ensure weights are moved to CPU before transmission
            weights = {k: v.cpu() for k, v in client.model.state_dict().items()}
            
            # 3. Construct Payload
            payload = {
                'client_id': f"client_{i}",
                'protos': local_protos.cpu(),
                'weights': weights
            }
            client_payloads.append(payload)
            
        # --- D. SERVER AGGREGATION ---
        # Server calls `run_server_round` to:
        # 1. Merge new prototypes into the Global Bank (EMA).
        # 2. Average client weights (FedAvg).
        logger.info(">> Server Aggregation...")
        server_result = run_server_round(
            proto_manager=proto_bank,
            model_server=fed_server,
            client_payloads=client_payloads
        )
        
        # Update Global State for the next round
        global_protos = server_result['global_prototypes']
        global_weights = server_result['global_weights']
        
        # --- E. GLOBAL MODEL UPDATE ---
        # Update the server's central model with the new consensus weights
        base_model.load_state_dict(global_weights)
        
        # Log Summary
        if global_protos is not None:
            logger.info(f"Round {round_idx} Complete. Global Bank Size: {global_protos.shape[0]}")
        else:
            logger.error("Round Complete but Global Protos is None!")

    logger.info("\nPipeline Finished Successfully.")

if __name__ == "__main__":
    main()
