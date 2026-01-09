import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any, Tuple, Union

class FederatedServer:
    """
    Orchestrates the server-side logic for Federated Learning with Prototypes.
    
    This class handles global prototype aggregation, clustering (to remove redundancy),
    metadata management, and the preparation of broadcast messages for clients.

    Attributes:
        global_model (nn.Module): The central model architecture.
        config (Dict[str, Any]): Configuration dictionary containing hyperparameters.
        global_prototypes (Optional[torch.Tensor]): Aggregated global prototypes [K_global, dim].
        prototype_counts (Optional[torch.Tensor]): Sample counts contributing to each prototype [K_global].
        prototype_confidence (Optional[torch.Tensor]): Reliability score of each prototype [K_global].
        prototype_created_round (Optional[torch.Tensor]): Round ID when prototype was created.
        prototype_last_used (Optional[torch.Tensor]): Round ID when prototype was last updated.
        task_per_prototype (Optional[torch.Tensor]): Task ID association for each prototype.
    """
    
    def __init__(self, global_model: nn.Module, config: Dict[str, Any]):
        """
        Initializes the Federated Server.

        Args:
            global_model (nn.Module): The global model instance.
            config (Dict[str, Any]): Dictionary containing settings like 'prototype_merge_threshold'.
        """
        self.global_model = global_model
        self.config = config
        
        # Global prototypes and metadata
        self.global_prototypes: Optional[torch.Tensor] = None
        self.prototype_counts: Optional[torch.Tensor] = None
        self.prototype_confidence: Optional[torch.Tensor] = None
        
        # Lifecycle tracking
        self.prototype_created_round: Optional[torch.Tensor] = None
        self.prototype_last_used: Optional[torch.Tensor] = None
        self.task_per_prototype: Optional[torch.Tensor] = None 
    
    @torch.no_grad()
    def bootstrap_global_prototypes(
        self, 
        client_protos_list: List[torch.Tensor], 
        client_counts_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates and merges local prototypes from clients to form the global prototype bank.
        
        Process:
        1. Concatenate all local prototypes.
        2. Perform greedy clustering to merge similar prototypes based on Cosine Similarity.
        3. Compute weighted averages for merged prototypes.
        
        Args:
            client_protos_list (List[torch.Tensor]): List of tensors, each shape [K_local, embedding_dim].
            client_counts_list (List[torch.Tensor]): List of tensors, each shape [K_local] (counts).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - global_prototypes: [K_global, embedding_dim]
                - global_counts: [K_global]
        
        Raises:
            ValueError: If input lists are empty.
        """
        if not client_protos_list:
            raise ValueError("No client prototypes received. Cannot bootstrap.")

        # 1. Collect & Concatenate
        # Ensure all tensors are on the same device (e.g., CPU) for processing
        device = client_protos_list[0].device
        
        all_protos = torch.cat(client_protos_list, dim=0)  # [Total_K, Dim]
        
        # Flatten counts list
        all_counts_list = []
        for counts in client_counts_list:
            all_counts_list.extend(counts.tolist())
        all_counts = torch.tensor(all_counts_list, device=device)
        
        # Normalize inputs to ensure cosine similarity works correctly
        all_protos = F.normalize(all_protos, dim=1)
        
        print(f"[Server] Processing {len(all_protos)} local prototypes from {len(client_protos_list)} clients...")
        
        # 2. Redundancy Removal via Greedy Clustering
        merged_protos_list = []
        merged_counts_list = []
        used_indices = set()
        
        threshold = self.config.get('prototype_merge_threshold', 0.85)
        num_protos = len(all_protos)

        for i in range(num_protos):
            if i in used_indices:
                continue
            
            # Calculate similarity of proto 'i' against all *original* protos
            # Note: We calculate against 'all_protos' to find neighbors. 
            # Optimization: could mask 'used_indices' here, but strictly not required for correctness.
            sims = F.cosine_similarity(all_protos[i:i+1], all_protos)
            
            # Identify neighbors (including self)
            similar_indices = (sims > threshold).nonzero(as_tuple=True)[0].tolist()
            
            # 3. Merge Strategy: Weighted Average
            proto_acc = torch.zeros_like(all_protos[0])
            count_acc = 0.0
            
            for j in similar_indices:
                if j not in used_indices:
                    weight = all_counts[j].float()
                    proto_acc += all_protos[j] * weight
                    count_acc += weight.item()
                    used_indices.add(j)
            
            if count_acc > 0:
                # Normalize the weighted sum to get the mean direction
                merged_proto = proto_acc / count_acc
                merged_proto = F.normalize(merged_proto, dim=0)
                
                merged_protos_list.append(merged_proto)
                merged_counts_list.append(count_acc)
        
        # Stack results
        self.global_prototypes = torch.stack(merged_protos_list)
        self.prototype_counts = torch.tensor(merged_counts_list, device=device)
        
        # Initialize Metadata containers matching the new size
        k_global = len(self.global_prototypes)
        self.prototype_confidence = torch.ones(k_global, device=device)
        self.prototype_created_round = torch.zeros(k_global, device=device)
        self.prototype_last_used = torch.zeros(k_global, device=device)
        self.task_per_prototype = torch.zeros(k_global, dtype=torch.long, device=device)
        
        print(f"[Server] Redundancy removal complete. Reduced {num_protos} -> {k_global} global prototypes.")
        
        return self.global_prototypes, self.prototype_counts
    
    def compute_prototype_confidence(self) -> torch.Tensor:
        """
        Computes a confidence score for each global prototype.
        
        Formula:
            Confidence = (count / total_count) * (1 - normalized_variance)
            
        Note: currently assumes unit variance until variance tracking is implemented.
        
        Returns:
            torch.Tensor: Confidence scores [K_global].
        """
        if self.global_prototypes is None or self.prototype_counts is None:
            raise RuntimeError("Global prototypes not initialized. Run bootstrap first.")
        
        # Avoid division by zero
        total_count = self.prototype_counts.sum()
        if total_count == 0:
            count_conf = torch.zeros_like(self.prototype_counts)
        else:
            count_conf = self.prototype_counts / total_count
        
        # Variance-based confidence (Placeholder: assume 1.0/High confidence)
        # TODO: Implement actual variance tracking during aggregation
        variance_conf = torch.ones_like(count_conf)
        
        self.prototype_confidence = count_conf * variance_conf
        
        return self.prototype_confidence

    def prepare_server_broadcast(
        self, 
        current_round: int, 
        aggregated_adapters: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Prepares the data package to be broadcast to selected clients for the next round.
        
        Optimizes bandwidth by sending only:
        1. Aggregated Adapter weights (small).
        2. Global Prototypes (medium).
        3. Metadata (tiny).
        
        It explicitly excludes the frozen encoder backbone weights.

        Args:
            current_round (int): The current federated learning round ID.
            aggregated_adapters (Dict[str, torch.Tensor]): The averaged adapter state dict.

        Returns:
            Dict[str, Any]: The payload dictionary to send to clients.
        """
        if self.global_prototypes is None:
            # Fallback if prototypes aren't ready (e.g., Round 0)
            print("[Server] Warning: Broadcasting without global prototypes (not initialized).")
            prototypes = None
        else:
            prototypes = self.global_prototypes

        server_message = {
            'round_id': current_round,
            
            # --- Knowledge Distillation Data ---
            'global_prototypes': prototypes,                # [K, 768]
            'prototype_counts': self.prototype_counts,      # [K]
            'prototype_confidence': self.prototype_confidence,
            'prototype_task_id': self.task_per_prototype,
            
            # --- Model Parameters ---
            # Only sending adapters (e.g., ~0.5MB) vs Full Model (~350MB)
            'adapter_params': aggregated_adapters,
            
            # Explicitly None to signal clients to use their local frozen backbone
            'encoder_weights': None, 
        }
        
        # Calculate approximate payload size for logging
        total_bytes = 0
        
        # Size of Prototypes
        if prototypes is not None:
            total_bytes += prototypes.numel() * prototypes.element_size()
            
        # Size of Adapters
        if aggregated_adapters:
            for tensor in aggregated_adapters.values():
                total_bytes += tensor.numel() * tensor.element_size()
        
        kb_size = total_bytes / 1024
        print(f"[Server] Broadcasting payload prepared. Approx size: {kb_size:.1f} KB per client.")
        
        return server_message