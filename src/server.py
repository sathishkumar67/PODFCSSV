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

class GlobalPrototypeBank:
    """
    Central Repository for Global Visual Prototypes.
    
    This class manages the lifecycle of global prototypes, which serve as the shared
    knowledge base for the Federated Learning system. It implements an online 
    clustering mechanism that continually updates the prototype bank as new data 
    arrives from clients each round.

    Core Mechanism: 'Merge-or-Add' with EMA
    ---------------------------------------
    When a client sends a set of local prototypes, the server processes them one by one:
    
    1.  **Similarity Check**: It compares the local prototype (P) against all existing 
        global prototypes (G) using Cosine Similarity.
    
    2.  **Merge Decision**:
        -   If the similarity to the best match (G_best) is >= `merge_threshold`, 
            P is considered an observation of an existing concept.
            Action: Update G_best using Exponential Moving Average (EMA).
            
            G_best_new = Normalize( (1 - alpha) * G_best_old + alpha * P )
            
            This ensures that the global prototype evolves smoothly towards the most 
            recent observations without drastic jumps.

        -   If the similarity is < `merge_threshold`, P is considered a NEW concept.
            Action: Add P to the bank as a new global prototype.

    This approach allows the system to automatically discover new classes/features 
    (Add) while refining known ones (Merge), without needing a pre-defined number of classes.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 768, 
        merge_threshold: float = 0.8, 
        ema_alpha: float = 0.1,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the Global Prototype Bank.

        Args:
            embedding_dim (int): The dimensionality of the feature vectors.
            merge_threshold (float): The cosine similarity score [0, 1] required to 
                                    merge a local prototype into an existing global one.
                                    High value (e.g., 0.9) = stricter merging (more new prototypes).
                                    Low value (e.g., 0.6) = looser merging (fewer prototypes).
            ema_alpha (float): The interpolation factor for EMA updates.
                            Range: (0, 1].
                            - Small alpha (0.1): Global proto changes slowly (more stable).
                            - Large alpha (0.9): Global proto changes quickly (more responsive).
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)
        
        # Tensor storing all global prototypes.
        # Shape: [M, D] where M is the current number of prototypes.
        # All vectors are always kept L2-normalized.
        self.prototypes = torch.zeros(0, embedding_dim, device=self.device)

    @torch.no_grad()
    def merge_local_prototypes(
        self, 
        local_protos_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Integrates a batch of new local prototypes from multiple clients into the global bank.

        Algorithm Steps:
        1.  Concatenate all incoming local prototypes into a single batch.
        2.  Normalize them to the unit hypersphere.
        3.  Iterate through each local prototype sequentially:
            a.  Calculate Cosine Similarity with all current global prototypes.
            b.  Find the Best Match (Max Similarity).
            c.  If Max Sim >= Threshold:
                    UPDATE the best match using EMA.
            d.  Else:
                    APPEND the local prototype as a new global prototype.
        
        Args:
            local_protos_list (List[torch.Tensor]): A list where each element is a tensor 
                                                    of prototypes from one client.
                                                    Shapes: [K_local, Dim]

        Returns:
            torch.Tensor: The updated state of the Global Prototype Bank [M_new, Dim].
        """
        if not local_protos_list:
            return self.prototypes

        # 1. Prepare batch: Flatten list of tensors into one large tensor
        incoming = torch.cat(local_protos_list, dim=0).to(self.device)
        
        # Ensure incoming vectors are normalized for valid cosine similarity
        incoming = F.normalize(incoming, p=2, dim=1)
        
        # 2. Sequential Merge Process
        # We process sequentially rather than in parallel to handle cases where 
        # multiple incoming prototypes might belong to the same (possibly new) cluster.
        for i in range(incoming.size(0)):
            p_new = incoming[i]
            
            # Case 0: Bank is empty (First round/initialization)
            if self.prototypes.size(0) == 0:
                self.prototypes = p_new.unsqueeze(0)
                continue
                
            # Compute similarities: Dot product of (M, D) and (D,) -> (M,)
            sims = torch.mv(self.prototypes, p_new)
            max_sim, best_idx = sims.max(dim=0)
            
            # Decision Logic
            if max_sim >= self.merge_threshold:
                # MATCH found: Refine the existing global concept
                # EMA Update: New = (1-a)*Old + a*Incoming
                old_vec = self.prototypes[best_idx]
                new_vec = (1 - self.ema_alpha) * old_vec + self.ema_alpha * p_new
                
                # Renormalize immediately to stay on hypersphere
                self.prototypes[best_idx] = F.normalize(new_vec, p=2, dim=0)
            else:
                # NO MATCH: This represents a novel feature/concept
                # Append to the bank
                self.prototypes = torch.cat(
                    [self.prototypes, p_new.unsqueeze(0)], 
                    dim=0
                )
                
        return self.prototypes

    def get_prototypes(self) -> torch.Tensor:
        """
        Retrieve the current set of global prototypes.
        
        Returns:
            torch.Tensor: The global prototype bank [M, D].
        """
        return self.prototypes


class FederatedModelServer:
    """
    Federated Averaging (FedAvg) Aggregator.
    
    This class is responsible for aggregating the model parameters (weights and biases)
    from multiple clients to produce a global consensus model. It implements the 
    standard FedAvg algorithm, where the global parameters are the arithmetic mean 
    of the client parameters.
    """
    
    @torch.no_grad()
    def aggregate_weights(
        self, 
        client_weights_map: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates model weights from all participating clients.

        Logic:
        1.  Identify the common set of parameter keys (layer names).
        2.  For each layer:
            a.  Collect the weight tensors from all clients.
            b.  Stack them and compute the Mean (Average).
            c.  Store in the global state dictionary.
        
        Assumption:
        - All clients return the same model architecture structure (same keys).
        - If a client is missing a key, that layer is skipped (logged as warning).

        Args:
            client_weights_map (Dict): Mapping of ClientID -> Model State Dict.
                                    Each state dict maps LayerName -> WeightTensor.

        Returns:
            Dict[str, torch.Tensor]: The aggregated Global State Dict.
        """
        if not client_weights_map:
            logger.warning("No client weights received for aggregation.")
            return {}

        client_ids = list(client_weights_map.keys())
        num_clients = len(client_ids)
        
        # Optimization: If only one client, no averaging needed.
        if num_clients == 1:
            return client_weights_map[client_ids[0]]

        # Use the first client's weights as a template for structure
        reference_client = client_ids[0]
        reference_state = client_weights_map[reference_client]
        reference_keys = reference_state.keys()
        
        global_state: Dict[str, torch.Tensor] = {}

        for key in reference_keys:
            # Collect this layer's weights from all clients
            tensors = []
            for cid in client_ids:
                if key in client_weights_map[cid]:
                    # Convert to float for high-precision averaging (avoids overflow/underflow)
                    tensors.append(client_weights_map[cid][key].float())
            
            if len(tensors) == num_clients:
                # Compute Mean: Sum(Weights) / N
                stacked = torch.stack(tensors, dim=0)
                global_state[key] = stacked.mean(dim=0)
            else:
                logger.warning(f"Layer {key} missing in some clients. Skipping aggregation for this layer.")

        return global_state


def run_server_round(
    proto_manager: GlobalPrototypeBank, 
    model_server: FederatedModelServer, 
    client_payloads: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Server Round Orchestrator.
    
    This function executes the server-side logic for a single communication round:
    1.  Extracts Local Prototypes from client payloads and merges them into the Global Bank.
    2.  Extracts Local Model Weights and aggregates them using FedAvg.
    
    Args:
        proto_manager (GlobalPrototypeBank): The prototype manager instance.
        model_server (FederatedModelServer): The weight aggregator instance.
        client_payloads (List[Dict]): Data received from clients.
            Expected format: {'client_id': str, 'protos': Tensor, 'weights': Dict}

    Returns:
        Dict: The updated global state to be broadcast to clients next round.
            keys: 'global_prototypes', 'global_weights'
    """
    if not client_payloads:
        return {}

    # 1. Aggregate Prototypes
    # We collect all prototype tensors into a list. Client identity doesn't matter 
    # for clustering, as prototypes are treated as independent feature observations.
    protos = [p['protos'] for p in client_payloads if 'protos' in p]
    global_protos = proto_manager.merge_local_prototypes(protos)
    
    # 2. Aggregate Model Weights
    # We map weights to client IDs for structured aggregation.
    client_weights_map = {}
    for i, p in enumerate(client_payloads):
        if 'weights' in p:
            # Fallback ID generation if missing
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
    
    This class encapsulates the global model instance (e.g., ViT-MAE).
    It handles initialization (loading pretrained backbones) and updating 
    its internal state with aggregated weights from the server.
    """
    
    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the Global Model.
        
        Attempts to load `ViTMAEForPreTraining` from Hugging Face.
        If dependencies are missing, it falls back to a mock model for testing purposes.
        
        Args:
            device (str): target device ('cpu', 'cuda').
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
            logger.warning(f"Failed to import required modules for real model: {e}. Using Mock Model.")
            # Create a simple mock for testing logic flow without HuggingFace dependencies
            self.model = torch.nn.Linear(10, 10).to(self.device)
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