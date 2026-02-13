
from __future__ import annotations
import logging
from typing import List, Dict, Optional, Any, Tuple

import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerPrototypeManager:
    """
    Manages the global prototype bank on the server using a merge-or-add strategy.
    
    This class maintains the 'collective knowledge' by aggregating and merging 
    local prototypes into a global dictionary of visual concepts.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 768, 
        merge_threshold: float = 0.8, 
        device: str = "cpu"
    ) -> None:
        """
        Initializes the global prototype bank.

        Args:
            embedding_dim: Dimensionality of feature vectors.
            merge_threshold: Cosine similarity threshold for merging.
            device: Computation device.
        """
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.device = torch.device(device)
        
        # State initialization
        self.global_prototypes = torch.zeros(0, embedding_dim, device=self.device)
        self.global_counts = torch.zeros(0, device=self.device)

    @torch.no_grad()
    def aggregate_prototypes(
        self, 
        local_protos_list: List[torch.Tensor], 
        local_counts_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates local prototypes into the global bank.

        Args:
            local_protos_list: List of local prototype tensors [Num_Local, D].
            local_counts_list: List of confidence counts [Num_Local].

        Returns:
            Tuple of updated global prototypes and counts.
        """
        if not local_protos_list or not local_counts_list:
            logger.warning("No prototypes provided for aggregation.")
            return self.global_prototypes.clone(), self.global_counts.clone()

        # Input validation: check for length mismatch
        if len(local_protos_list) != len(local_counts_list):
            raise ValueError("Size mismatch between prototypes and counts.")

        # Batch processing: Flatten all incoming data
        incoming_protos = torch.cat(local_protos_list, dim=0).to(self.device)
        incoming_counts = torch.cat(local_counts_list, dim=0).to(self.device)

        # Normalize once for batch similarity computation
        incoming_protos = F.normalize(incoming_protos, p=2, dim=1)

        for i in range(incoming_protos.size(0)):
            p_new = incoming_protos[i]
            c_new = incoming_counts[i]

            # Lazy initialization or empty bank check
            if self.global_prototypes.size(0) == 0:
                self.global_prototypes = p_new.unsqueeze(0)
                self.global_counts = c_new.unsqueeze(0)
                continue

            # Compute similarity to all global prototypes
            sims = torch.mv(self.global_prototypes, p_new)
            max_sim, best_idx = sims.max(dim=0)

            if max_sim > self.merge_threshold:
                # Merge existing concept
                self._merge_prototype(best_idx, p_new, c_new)
            else:
                # Add novel concept
                self._add_prototype(p_new, c_new)

        return self.global_prototypes, self.global_counts

    def _merge_prototype(self, idx: int, p_new: torch.Tensor, c_new: torch.Tensor) -> None:
        """Internal helper for weighted prototype merging."""
        count_old = self.global_counts[idx]
        count_total = count_old + c_new
        
        # Weighted mean update
        merged = (self.global_prototypes[idx] * count_old + p_new * c_new) / count_total
        self.global_prototypes[idx] = F.normalize(merged, p=2, dim=0)
        self.global_counts[idx] = count_total

    def _add_prototype(self, p_new: torch.Tensor, c_new: torch.Tensor) -> None:
        """Internal helper for appending a new concept."""
        self.global_prototypes = torch.cat([self.global_prototypes, p_new.unsqueeze(0)], dim=0)
        self.global_counts = torch.cat([self.global_counts, c_new.unsqueeze(0)], dim=0)


class FederatedModelServer:
    """
    Coordinator for model weight synchronization across clients.
    """
    
    @torch.no_grad()
    def aggregate_weights(
        self, 
        client_params_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Performs FedAvg on a list of state_dicts.

        Args:
            client_params_list: List of state_dicts from clients.

        Returns:
            The averaged global state_dict.
        """
        if not client_params_list:
            return {}

        num_clients = len(client_params_list)
        if num_clients == 1:
            return client_params_list[0]

        # Use the first client's dict as the reference for structure
        reference_dict = client_params_list[0]
        global_state: Dict[str, torch.Tensor] = {}

        for key in reference_dict.keys():
            # Filter non-torch types if any exist in the dict
            stacked_tensors = torch.stack(
                [client_dict[key].float() for client_dict in client_params_list], 
                dim=0
            )
            global_state[key] = stacked_tensors.mean(dim=0)

        return global_state


def run_server_round(
    proto_manager: ServerPrototypeManager, 
    model_server: FederatedModelServer, 
    payloads: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Orchestrates the server update round logic.

    Args:
        proto_manager: Instance of ServerPrototypeManager.
        model_server: Instance of FederatedModelServer.
        payloads: List of dictionaries from clients containing 
                'protos', 'counts', and 'weights'.

    Returns:
        Consolidated global update dictionary.
    """
    # Defensive check
    if not payloads:
        return {}

    # 1. Aggregate Knowledge Base
    protos = [p['protos'] for p in payloads if 'protos' in p]
    counts = [p['counts'] for p in payloads if 'counts' in p]
    
    global_protos, global_confs = proto_manager.aggregate_prototypes(protos, counts)
    
    # 2. Aggregate Model Intelligence
    weights = [p['weights'] for p in payloads if 'weights' in p]
    global_weights = model_server.aggregate_weights(weights)
    
    return {
        "global_prototypes": global_protos,
        "global_confidences": global_confs,
        "global_weights": global_weights
    }