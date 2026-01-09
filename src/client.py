import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any, Optional


# Assuming LocalPrototypeManager and ExperienceReplayBuffer are imported from previous cells
# from modules import LocalPrototypeManager, ExperienceReplayBuffer 

class ClientNode:
    """
    Represents a Federated Learning Client Node.
    
    This class manages local data, the local model copy, prototype computation,
    and the local training loop.
    
    Attributes:
        client_id (str): Unique identifier for the client.
        local_loader (DataLoader): PyTorch DataLoader for local data.
        model (nn.Module): Local copy of the global model.
        config (Dict): Configuration parameters.
        device (torch.device): Computation device (CPU/GPU).
        prototype_manager (LocalPrototypeManager): Helper for clustering embeddings.
        optimizer (torch.optim.Optimizer): Optimizer for trainable parameters (Adapters).
    """
    
    def __init__(
        self, 
        client_id: str, 
        local_loader: DataLoader, 
        vit_model: nn.Module, 
        config: Dict[str, Any]
    ):
        """
        Initialize the Client Node.

        Args:
            client_id (str): Unique ID.
            local_loader (DataLoader): Data provider.
            vit_model (nn.Module): The initial global model (will be deepcopied).
            config (Dict[str, Any]): Hyperparameters (learning rate, K, etc.).
        """
        self.client_id = client_id
        self.local_loader = local_loader
        self.config = config
        
        # device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Deepcopy ensures this client has its own independent weights
        self.model = copy.deepcopy(vit_model)
        self.model.to(self.device)
        
        # Components
        # Note: Assuming LocalPrototypeManager is defined as in previous steps
        self.prototype_manager = LocalPrototypeManager(
            K=config.get('K', 20),
            alpha_ema=config.get('alpha_ema', 0.1),
            novelty_threshold=config.get('novelty_threshold', 0.4)
        )
        
        # Placeholder for Replay Buffer (if used in later rounds)
        self.replay_buffer = None # ExperienceReplayBuffer(max_size=1000)
        
        # Optimizer Setup: Only optimize Adapters (which require_grad=True)
        # We filter parameters to ensure we don't accidentally train the frozen backbone
        adapter_params = [
            p for name, p in self.model.named_parameters() 
            if 'adapter' in name and p.requires_grad
        ]
        
        if not adapter_params:
            print(f"[Client {self.client_id}] Warning: No trainable adapter parameters found.")
        
        self.optimizer = torch.optim.Adam(
            adapter_params, 
            lr=config.get('lr', 0.001)
        )

    @torch.no_grad()
    def align_prototypes_to_global(
        self, 
        global_prototypes: torch.Tensor, 
        global_confidence: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Client Alignment (Passive Phase).
        
        Performs "Semantic Translation" by aligning local representations to the 
        global shared space. This step calculates the nearest global prototype 
        for every local prototype.

        Purpose:
        - Establishes a mapping: Local_Proto_ID -> Global_Proto_ID.
        - Allows the client to know "what the server thinks this concept looks like."
        - Prepares metadata for the upcoming active training phase (distillation).
        
        Process:
        1. Normalize vectors.
        2. Compute Cosine Similarity Matrix [K_local, K_global].
        3. Assign each local prototype to its nearest global neighbor (Argmax).
        
        Args:
            global_prototypes (torch.Tensor): Tensor of shape [K_global, Embedding_Dim].
            global_confidence (torch.Tensor): Confidence scores of global prototypes [K_global].
            
        Returns:
            Dict[str, Any]: The alignment metadata dictionary.
        """
        # 1. Validation and Safety Checks
        if self.prototype_manager.prototypes is None:
            print(f"[Client {self.client_id}] Warning: No local prototypes found. Skipping alignment.")
            return {}

        local_protos = self.prototype_manager.prototypes
        
        # 2. Device Management
        # Global prototypes likely arrived from CPU (server broadcast). 
        # We must move them to the client's device (GPU) for matrix multiplication.
        global_protos_device = global_prototypes.to(self.device)
        global_conf_device = global_confidence.to(self.device)
        
        # Verify dimensions
        if local_protos.shape[1] != global_protos_device.shape[1]:
            raise ValueError(
                f"Dimension mismatch: Local {local_protos.shape} vs Global {global_protos_device.shape}"
            )

        # 3. Compute Similarity Matrix (Cosine Similarity)
        # Formula: (A . B^T) / (|A| * |B|)
        # Since we normalize first, this becomes just matrix multiplication.
        local_norm = F.normalize(local_protos, dim=1)
        global_norm = F.normalize(global_protos_device, dim=1)
        
        # Matrix Multiplication: [K_local, Dim] @ [Dim, K_global] -> [K_local, K_global]
        similarity_matrix = torch.mm(local_norm, global_norm.T)
        
        # 4. Find Best Matches (Nearest Neighbor)
        # For each local prototype (row), find column with max similarity
        best_similarity, best_global_idx = similarity_matrix.max(dim=1)
        
        # 5. Store Alignment Metadata
        # We store this in the client state for use in the Loss Function (Step 5)
        self.prototype_alignment = {
            'local_to_global_idx': best_global_idx,  # [K_local] (Indices)
            'similarity_scores': best_similarity,    # [K_local] (Values 0.0 to 1.0)
            'global_confidence': global_conf_device, # [K_global] (Reference)
            'global_prototypes': global_protos_device # Store for distillation loss
        }
        
        # Logging statistics
        avg_sim = best_similarity.mean().item()
        min_sim = best_similarity.min().item()
        
        print(f"[Client {self.client_id}] Alignment Complete:")
        print(f"  - Mapped {len(local_protos)} local -> {len(global_protos_device)} global prototypes")
        print(f"  - Similarity stats: Mean={avg_sim:.3f}, Min={min_sim:.3f}")
        
        return self.prototype_alignment

    
    def train_round_1(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes STEP 1: Round-1 Local Training (Cold Start).
        
        Workflow:
        1. Feature Extraction: Pass all data through the frozen backbone to get embeddings.
        2. Prototype Initialization: Use K-Means on these embeddings to create local prototypes.
        3. Adapter Training: Train only the adapters using MAE Reconstruction Loss.
        (Note: No prototype-based loss is used in Round 1 as global prototypes don't exist yet).
        
        Returns:
            local_prototypes (torch.Tensor): [K, Dim]
            prototype_counts (torch.Tensor): [K]
            model_update (Dict[str, torch.Tensor]): State dict of updated adapter weights.
        """
        print(f"[Client {self.client_id}] Starting Round 1 training...")
        
        # --- Phase 1: Feature Extraction & Prototype Initialization ---
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(self.local_loader):
                images = images.to(self.device)
                
                # Forward pass - expected to return (output, embeddings)
                # Adjust 'return_features=True' based on your specific model wrapper
                _, embeddings = self.model(images, return_features=True)
                
                # Move to CPU immediately to prevent GPU OOM on large datasets
                all_embeddings.append(embeddings.cpu())
        
        if not all_embeddings:
            raise ValueError(f"Client {self.client_id} has no data!")

        # Concatenate all extracted features
        all_embeddings_tensor = torch.cat(all_embeddings, dim=0)  # [Total_Samples, 768]
        
        # Compute Prototypes (K-Means)
        # We use the manager to handle the clustering logic
        local_protos, proto_counts = self.prototype_manager.compute_local_prototypes_kmeans(
            all_embeddings_tensor, 
            K=self.config.get('K', 20)
        )
        
        # --- Phase 2: Train Adapters (Reconstruction Only) ---
        self.model.train()
        num_epochs = self.config.get('local_epochs', 1)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (images, _) in enumerate(self.local_loader):
                images = images.to(self.device)
                
                # 1. Forward Pass
                # Note: Depending on your specific MAE implementation, 
                # you might need to call forward_decoder explicitly or just model(images).
                # Here we follow the logic provided in the prompt.
                
                # We assume model.vit.forward_decoder exists or model returns reconstruction
                if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'forward_decoder'):
                    # Custom implementation logic
                    # Pass through encoder (with adapters)
                    latent, _ = self.model(images, return_features=True)
                    # Pass through decoder
                    reconstructed = self.model.vit.forward_decoder(latent) # Shape check needed here
                else:
                    # Fallback standard forward
                    reconstructed, _ = self.model(images)

                # 2. Reconstruction Loss (MSE)
                # Flatten images: [B, C, H, W] -> [B, N_Pixels]
                original = images.view(images.size(0), -1)
                
                # Ensure shapes match for MSE
                if reconstructed.shape != original.shape:
                    # Handle cases where reconstruction might be patched
                    reconstructed = reconstructed.view(original.shape)

                loss = F.mse_loss(reconstructed, original)
                
                # 3. Optimization
                self.optimizer.zero_grad(set_to_none=True) # Slightly more efficient
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.local_loader)
            # Optional: Print progress
            # print(f"  Ep {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        # --- Phase 3: Extract Updates ---
        # We only send back the parameters that were actually trained (Adapters)
        # to save bandwidth and ensure the server doesn't overwrite the frozen backbone.
        model_update = {
            name: param.cpu().clone() 
            for name, param in self.model.named_parameters()
            if 'adapter' in name and param.requires_grad
        }
        
        print(f"[Client {self.client_id}] Round 1 complete. Protos: {local_protos.shape}")
        
        return local_protos, proto_counts, model_update


def federated_round_1(
    clients: List[ClientNode], 
    config: Dict[str, Any]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Orchestrates STEP 1 (Round 1) across all participating clients.
    
    This is the 'Cold Start' phase:
    1. Clients extract features from their local data.
    2. Clients cluster features to form initial local prototypes.
    3. Clients train adapters on reconstruction task.
    
    Args:
        clients (List[ClientNode]): List of participating client objects.
        config (Dict): Global configuration.
        
    Returns:
        all_local_protos (List[torch.Tensor]): List of prototype tensors per client.
        all_proto_counts (List[torch.Tensor]): List of count tensors per client.
        all_updates (List[Dict]): List of state_dicts (adapters only) per client.
    """
    print(f"\n{'='*40}")
    print(f"FEDERATED ROUND 1 (Initialization)")
    print(f"{'='*40}")
    
    all_local_protos = []
    all_proto_counts = []
    all_updates = []
    
    # In a real system, this loop would likely be parallelized using Ray or multiprocessing
    for i, client in enumerate(clients):
        try:
            local_protos, proto_counts, model_update = client.train_round_1()
            
            all_local_protos.append(local_protos)
            all_proto_counts.append(proto_counts)
            all_updates.append(model_update)
            
        except Exception as e:
            print(f"[Error] Client {client.client_id} failed during Round 1: {e}")
            # In production, you might want to exclude this client's results
            continue

    return all_local_protos, all_proto_counts, all_updates



def client_alignment_phase(
    client: ClientNode, 
    global_prototypes: torch.Tensor, 
    global_confidence: torch.Tensor
) -> Dict[str, Any]:
    """
    Wrapper function to execute Step 4 for a specific client.
    """
    try:
        return client.align_prototypes_to_global(global_prototypes, global_confidence)
    except Exception as e:
        print(f"[Error] Alignment failed for Client {client.client_id}: {e}")
        return {}