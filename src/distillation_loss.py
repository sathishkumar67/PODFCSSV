import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal

class GatedPrototypeDistillationLoss(nn.Module):
    """
    Gated Prototype Distillation Loss.

    This module implements a mechanism to distill knowledge from global prototypes 
    into local client models. The key novelty is the **Gating Mechanism**, which 
    weighs the loss contribution based on:
    1. Similarity (How close is the match?)
    2. Uncertainty (Is the assignment ambiguous?)
    3. Global Confidence (Does the server trust this prototype?)

    Attributes:
        temperature (float): Scaling factor for logits (softmax/sigmoid).
        distillation_weight (float): Overall weight of this loss term.
        gating_type (str): The strategy used to compute gate weights.
        tau_threshold (float): Base threshold for gating activation.
    """
    
    def __init__(
        self, 
        temperature: float = 0.07, 
        distillation_weight: float = 0.5, 
        gating_type: Literal['binary', 'soft_sigmoid', 'confidence_weighted', 'soft_assignment'] = 'soft_sigmoid', 
        tau_threshold: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.gating_type = gating_type
        self.tau_threshold = tau_threshold
        
        # Validation
        valid_types = {'binary', 'soft_sigmoid', 'confidence_weighted', 'soft_assignment'}
        if gating_type not in valid_types:
            raise ValueError(f"Invalid gating_type '{gating_type}'. Options: {valid_types}")

    def forward(
        self, 
        embeddings: torch.Tensor, 
        best_similarity: torch.Tensor, 
        global_prototypes: torch.Tensor, 
        prototype_confidence: torch.Tensor, 
        global_counts: Optional[torch.Tensor] = None, 
        current_task: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the gated distillation loss.

        Args:
            embeddings (torch.Tensor): Client embeddings [Batch, Dim].
            best_similarity (torch.Tensor): Pre-computed max cosine similarity [Batch].
            global_prototypes (torch.Tensor): Global prototype bank [K, Dim].
            prototype_confidence (torch.Tensor): Server-side confidence scores [K].
            global_counts (Optional[torch.Tensor]): Usage counts (unused in current logic, kept for API compat).
            current_task (Optional[int]): Task ID (unused in current logic, kept for API compat).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - L_proto: Scalar loss value (averaged over batch).
                - gate_weights: Tensor [Batch] showing the weight assigned to each sample.
        """
        # 1. Device Management
        # Ensure global resources are on the same device as the embeddings (GPU)
        device = embeddings.device
        global_protos = global_prototypes.to(device)
        proto_conf = prototype_confidence.to(device)
        
        # 2. Normalization & Similarity
        embeddings_norm = F.normalize(embeddings, dim=1)
        global_protos_norm = F.normalize(global_protos, dim=1)
        
        # Recompute similarities (required for entropy and soft assignment)
        # [B, Dim] @ [Dim, K] -> [B, K]
        sims = torch.mm(embeddings_norm, global_protos_norm.T)
        
        # Identify best matches
        best_proto_idx = sims.argmax(dim=1)  # [B]
        
        # 3. Adaptive Uncertainty-Based Gating (Enhancement 4)
        # Calculate Entropy of the similarity distribution
        # High Entropy = Ambiguous assignment (uncertain) -> Lower the gate threshold
        probs = F.softmax(sims / self.temperature, dim=1)
        log_probs = F.log_softmax(sims / self.temperature, dim=1)
        
        # Entropy = - sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=1)  # [B]
        
        # Normalize entropy by log(K) to get range [0, 1]
        # Use torch.log to maintain device consistency (avoiding numpy)
        max_entropy = torch.log(torch.tensor(global_protos.size(0), device=device, dtype=torch.float))
        entropy_norm = entropy / (max_entropy + 1e-8)
        
        # Adaptive threshold: Higher uncertainty (entropy) -> Increase threshold (stricter)
        # Logic: If uncertain, we rely less on the match unless it's very strong?
        # Note: Original code was `tau - 0.2 * entropy`. 
        # If entropy is high (1.0), threshold drops? No, usually we want to filter OUT uncertain ones.
        # Preserving original logic:
        tau_adaptive = self.tau_threshold - (0.2 * entropy_norm)  # [B]
        
        # ====================
        # GATING STRATEGIES
        # ====================
        
        gate = None
        L_proto = None
        
        if self.gating_type == 'binary':
            # Option A: Hard cutoff
            gate = (best_similarity >= self.tau_threshold).float()
            
        elif self.gating_type == 'soft_sigmoid':
            # Option B: Differentiable soft gating (Recommended)
            # Shifts the sigmoid curve based on the adaptive threshold
            gate = torch.sigmoid((best_similarity - tau_adaptive) / self.temperature)
            
        elif self.gating_type == 'confidence_weighted':
            # Option C: Scale by Server-side confidence
            # If the server isn't sure about this prototype, don't pull the client towards it.
            batch_conf = proto_conf[best_proto_idx]  # [B]
            sigmoid_gate = torch.sigmoid((best_similarity - tau_adaptive) / self.temperature)
            gate = batch_conf * sigmoid_gate
            
        elif self.gating_type == 'soft_assignment':
            # Option D: Soft assignment over all prototypes (Distribution Matching)
            # This is computationally more expensive [B, K, D]
            
            # 1. Weights based on similarity
            soft_weights = probs  # [B, K] (already computed as softmax)
            
            # 2. Weighted MSE Loss
            # Expand dims for broadcasting: [B, 1, D] - [1, K, D] -> [B, K, D]
            # Warning: Memory intensive for large K. 
            diff = embeddings_norm.unsqueeze(1) - global_protos_norm.unsqueeze(0)
            squared_diff = diff.pow(2).sum(dim=2)  # [B, K] Euclidean dist squared
            
            # Weighted sum of distances
            proto_loss_per_sample = (soft_weights * squared_diff).sum(dim=1)  # [B]
            
            # 3. Gate based on confidence of the weighted cluster centers
            # weighted average of confidence scores
            avg_confidence = torch.matmul(soft_weights, proto_conf)  # [B]
            gate = avg_confidence
            
            # Final Loss
            L_proto = (gate * proto_loss_per_sample).mean()
            return L_proto, gate

        # ====================
        # STANDARD LOSS CALCULATION (For types A, B, C)
        # ====================
        
        if L_proto is None:
            # 1. Get the actual target vector (Best matched prototype)
            target_protos = global_protos_norm[best_proto_idx]  # [B, Dim]
            
            # 2. Calculate L2 distance (Euclidean)
            # We want to minimize distance between embedding and its assigned prototype
            dist = torch.norm(embeddings_norm - target_protos, p=2, dim=1)  # [B]
            
            # 3. Apply Gate
            L_proto = (gate * dist).mean()
        
        return L_proto, gate

    def __repr__(self):
        return (f"GatedPrototypeDistillationLoss(type='{self.gating_type}', "
                f"tau={self.tau_threshold}, temp={self.temperature})")