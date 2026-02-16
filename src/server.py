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
        client_params_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates decentralized model weights into a centralized consensus model.

        Args:
            client_params_list: List of state_dicts collected from clients.

        Returns:
            Dict: The aggregated global state_dict.
        """
        if not client_params_list:
            return {}

        num_clients = len(client_params_list)
        if num_clients == 1:
            return client_params_list[0]

        # Use the first client's state_dict as the structural reference
        reference_keys = client_params_list[0].keys()
        global_state: Dict[str, torch.Tensor] = {}

        for key in reference_keys:
            # Perform component-wise averaging across all client tensors for this layer
            # Conversion to float() ensures compatibility with half-precision local training
            tensors = torch.stack(
                [client_dict[key].float() for client_dict in client_params_list], 
                dim=0
            )
            global_state[key] = tensors.mean(dim=0)

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
                        must contain 'protos' and 'weights'.

    Returns:
        Dict: Final global update containing merged prototypes and weights.
    """
    if not client_payloads:
        return {}

    # Extract heterogeneous data types from the payload stream
    # Note: Counts were removed in v2.0 in favor of EMA-only updates
    protos = [p['protos'] for p in client_payloads if 'protos' in p]
    global_protos = proto_manager.aggregate_prototypes(protos)
    
    weights = [p['weights'] for p in client_payloads if 'weights' in p]
    global_weights = model_server.aggregate_weights(weights)
    
    return {
        "global_prototypes": global_protos,
        "global_weights": global_weights
    }