from __future__ import annotations
import logging
from typing import List, Dict, Optional, Any, Tuple

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class ServerPrototypeManager:
    """
    Server-side coordinator for Global Prototype Management.
    
    This class implements an online clustering approach (Discovery-Update) to maintain 
    a shared codebook of visual features across the federated network. It handles
    incoming local prototypes from decentralized clients using a 'Merge-or-Add' strategy
    driven by Exponential Moving Average (EMA).
    
    Technical Details:
    - Discovery: Uses Cosine Similarity to determine if a client-submitted prototype 
    is a known concept or a novel contribution.
    - Update: Uses EMA for smooth global state transitions, ensuring the global 
    representations evolve steadily rather than oscillating due to biased local client data.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 768, 
        merge_threshold: float = 0.8, 
        ema_alpha: float = 0.1,
        device: str = "cpu"
    ) -> None:
        """
        Args:
            embedding_dim: Dimensionality of the latent feature space (D).
            merge_threshold: Similarity score (0 to 1) required to anchor a local
                            prototype to an existing global entry.
            ema_alpha: Smoothing factor for the update rule. Higher values 
                    give more weight to the most recent client update.
            device: Hardware device to store and compute global prototypes.
        """
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)
        
        # State: Store prototypes on the unit hypersphere
        # Shape: [M, D] where M is the dynamic number of global prototypes.
        self.global_prototypes = torch.zeros(0, embedding_dim, device=self.device)

    @torch.no_grad()
    def aggregate_prototypes(
        self, 
        local_protos_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Aggregates a batch of decentralized prototypes into the global state.

        Algorithmic Pipeline:
        1. Flatten all incoming local prototypes into a single tensor.
        2. Filter and project them onto the unit hypersphere.
        3. Iterate through each local prototype:
        - If best match > threshold: Perform EMA update on the global centroid.
        - If best match < threshold: Add as a new global visual concept.

        Args:
            local_protos_list: Collection of prototype tensors from N clients.
                            List of shapes [Num_Local_i, D].

        Returns:
            torch.Tensor: The updated Global Prototype Bank [M_updated, D].
        """
        if not local_protos_list:
            logger.warning("Prototype aggregation invoked with empty payload.")
            return self.global_prototypes.clone()

        # Batch-process incoming data onto the server device
        incoming_protos = torch.cat(local_protos_list, dim=0).to(self.device)

        # L2-Normalization is prerequisite for Cosine Similarity during mm/mv ops
        incoming_protos = F.normalize(incoming_protos, p=2, dim=1)

        for i in range(incoming_protos.size(0)):
            p_new = incoming_protos[i]

            # Lazy-init: The first observed prototype defines the starting global state
            if self.global_prototypes.size(0) == 0:
                self.global_prototypes = p_new.unsqueeze(0)
                continue

            # Batch similarity check: New vector against all existing centroids
            similarities = torch.mv(self.global_prototypes, p_new)
            max_sim, best_idx = similarities.max(dim=0)

            if max_sim > self.merge_threshold:
                # REFINEMENT: Update nearest representative via EMA
                self._update_ema(best_idx, p_new)
            else:
                # DISCOVERY: Introduce a novel high-level concept to the bank
                self._expand_bank(p_new)

        return self.global_prototypes

    def _update_ema(self, idx: int, p_new: torch.Tensor) -> None:
        """
        Applies a non-linear EMA update to a specific global centroid.
        Rule: Global = (1-alpha)*Global + alpha*New
        """
        state_old = self.global_prototypes[idx]
        
        # Linear blend followed by projection back to unit length
        updated = (1 - self.ema_alpha) * state_old + self.ema_alpha * p_new
        self.global_prototypes[idx] = F.normalize(updated, p=2, dim=0)

    def _expand_bank(self, p_new: torch.Tensor) -> None:
        """Appends a new distinct feature vector to the codebook."""
        self.global_prototypes = torch.cat(
            [self.global_prototypes, p_new.unsqueeze(0)], 
            dim=0
        )


class FederatedModelServer:
    """
    Coordinator for Global Model Parameter Aggregation.
    
    Implements the standard Federated Averaging (FedAvg) protocol.
    This class is framework-agnostic regarding the backbone and focuses 
    exclusively on the weight-averaging logic across N client updates.
    """
    
    @torch.no_grad()
    def aggregate_weights(
        self, 
        client_weights_map: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates decentralized model weights into a centralized consensus model.
        
        This method processes ALL parameters in the state_dict, including:
        - Weights (conv, linear, attention)
        - Biases
        - LayerNorm / BatchNorm parameters (running_mean, running_var, num_batches_tracked)
        
        Args:
            client_weights_map: Dictionary mapping ClientID -> state_dict.

        Returns:
            Dict: The aggregated global state_dict (averaged per layer).
        """
        if not client_weights_map:
            logger.warning("No client weights received for aggregation.")
            return {}

        client_ids = list(client_weights_map.keys())
        num_clients = len(client_ids)
        
        # Use simple pass-through if only one client
        if num_clients == 1:
            return client_weights_map[client_ids[0]]

        # Use the first client's state_dict as the structural reference
        reference_client = client_ids[0]
        reference_state = client_weights_map[reference_client]
        reference_keys = reference_state.keys()
        
        global_state: Dict[str, torch.Tensor] = {}

        for key in reference_keys:
            # Check if key exists in all clients to avoid errors with partial updates
            tensors = []
            for cid in client_ids:
                if key in client_weights_map[cid]:
                    tensors.append(client_weights_map[cid][key].float())
            
            if len(tensors) == num_clients:
                # Perform component-wise averaging across all client tensors for this layer
                # Conversion to float() ensures compatibility with half-precision local training
                stacked = torch.stack(tensors, dim=0)
                global_state[key] = stacked.mean(dim=0)
            else:
                logger.warning(f"Layer {key} missing in some clients. Skipping aggregation for this layer.")

        return global_state


def run_server_round(
    proto_manager: ServerPrototypeManager, 
    model_server: FederatedModelServer, 
    client_payloads: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Orchestrates the server-side logic for a single communication round.

    Args:
        proto_manager: Instance of the Global Prototype controller.
        model_server: Instance of the Federated Aggregator.
        client_payloads: List of dictionaries from clients. Each payload 
                        must contain 'client_id', 'protos' and 'weights'.

    Returns:
        Dict: Final global update containing merged prototypes and weights.
    """
    if not client_payloads:
        return {}

    # 1. Aggregate Prototypes
    # Extract prototype lists (ignoring client IDs for prototypes as they are treated as a pool)
    protos = [p['protos'] for p in client_payloads if 'protos' in p]
    global_protos = proto_manager.aggregate_prototypes(protos)
    
    # 2. Aggregate Model Weights
    # Construct the map: ClientID -> StateDict
    # We expect 'client_id' to be present. If not, generate a dummy one.
    client_weights_map = {}
    for i, p in enumerate(client_payloads):
        if 'weights' in p:
            cid = p.get('client_id', f"unknown_client_{i}")
            client_weights_map[cid] = p['weights']
            
    global_weights = model_server.aggregate_weights(client_weights_map)
    
    return {
        "global_prototypes": global_protos,
        "global_weights": global_weights
    }


class GlobalModel:
    """
    Wrapper for the Server-Side Global Model.
    
    Manages the lifecycle of the central model, including initialization
    and parameter updates from federated aggregation rounds.
    """
    
    def __init__(self, device: str = "cpu") -> None:
        """
        Args:
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        logger.info(f"Initializing Global Model on {self.device}...")
        
        # Load the backbone (ViT-MAE)
        try:
            from transformers import ViTMAEForPreTraining
            from src.mae_with_adapter import inject_adapters
            
            # Initialize with standard pretrained weights
            self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
            
            # Inject the same adapters structure as clients
            self.model = inject_adapters(self.model)
            
            self.model.to(self.device)
            # Set to eval mode as the server primarily aggregates/evaluates
            self.model.eval() 
            
            logger.info("Global Model successfully initialized with Adapters.")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def update_model_weights(self, aggregated_weights: Dict[str, torch.Tensor]) -> None:
        """
        Updates the global model with averaged weights from the round.
        
        Args:
            aggregated_weights: Dictionary of aggregated parameters (full or partial).
        """
        if not aggregated_weights:
            logger.warning("Received empty weight update. Skipping.")
            return

        try:
            # We use strict=False because clients might only send back trainable parameters (adapters),
            # especially if the backbone is frozen.
            keys = self.model.load_state_dict(aggregated_weights, strict=False)
            logger.info(f"Global Model Updated. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
            
        except Exception as e:
            logger.error(f"Failed to update global model weights: {e}")
            raise


# =============================================================================
# Main Execution Block (For Testing)
# =============================================================================
if __name__ == "__main__":
    # Setup test logging
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n{'='*60}")
    print(f"[Test] Starting Server Logic Verification")
    print(f"{'='*60}")

    try:
        # 1. Initialize Server Components
        print("\n[Step 1] Initializing Server Components...")
        pm = ServerPrototypeManager(embedding_dim=768)
        fms = FederatedModelServer()
        global_model = GlobalModel(device="cpu")
        
        # 2. Simulate Client Payloads
        print("\n[Step 2] Simulating Client Updates...")
        
        # Mocking 2 clients
        # We simulate that clients send a subset of weights for testing, but we include both weights and biases
        # to demonstrate that all parameter types are handled.
        ref_state_dict = global_model.model.state_dict()
        
        # Client 1: Add 0.01 to everything
        c1_weights = {}
        for k, v in ref_state_dict.items():
            if 'adapter' in k: # Simulate sending only adapters for bandwidth efficiency in this test
                c1_weights[k] = v.clone() + 0.01

        # Client 2: Subtract 0.01 from everything
        c2_weights = {}
        for k, v in ref_state_dict.items():
            if 'adapter' in k:
                c2_weights[k] = v.clone() - 0.01
        
        # Manually introduce a bias term if not present in adapter (adapters usually have bias)
        # Just to be sure, let's verify if we are processing biases.
        # IBA_Adapter has up_project/down_project which are Linear layers (have bias by default)
        
        c1_proto = torch.randn(5, 768)
        c2_proto = torch.randn(3, 768)
        
        clients = [
            {'client_id': 'client_1', 'protos': c1_proto, 'weights': c1_weights},
            {'client_id': 'client_2', 'protos': c2_proto, 'weights': c2_weights}
        ]
        
        # 3. Run Server Round
        print("\n[Step 3] Running Aggregation Round...")
        updates = run_server_round(pm, fms, clients)
        
        print(f"   -> Global Prototypes Shape: {updates['global_prototypes'].shape}")
        
        # 4. Update Global Model
        print("\n[Step 4] Updating Global Model Weights...")
        global_model.update_model_weights(updates['global_weights'])
        
        print("\n[Success] All server tests passed successfully.")
        
    except Exception as e:
        print(f"\n[Error] Test Failed: {e}")
        import traceback
        traceback.print_exc()