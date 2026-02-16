from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GPADLoss(nn.Module):
    """
    Gated Prototype Anchored Distillation (GPAD) Loss.

    This loss function is designed for Federated Continual Learning scenarios where
    local models need to learn from global prototypes without catastrophic forgetting.
    It encourages the local model's embeddings to be close to the best-matching
    global prototype, but only if the match is sufficiently "confident".

    Key Mechanisms:
    1.  **Adaptive Thresholding**: The matching threshold is dynamic. It increases
        when the model is uncertain (high entropy in assignment), preventing
        noisy or ambiguous anchors from distorting the feature space.
    2.  **Gating**: A soft gating mechanism (sigmoid) scales the loss. If the
        similarity is marginally above the threshold, the weight is low. If it strongly
        exceeds the threshold, the weight approaches 1.0.
    3.  **Anchoring**: For confident samples, the loss minimizes the distance between the
        embedding and its matched prototype.
    """
    
    # Constants for numerical stability and configuration
    EPSILON: float = 1e-8
    SOFT_ASSIGNMENT_TEMP: float = 0.1

    def __init__(self, 
                base_tau: float = 0.5, 
                temp_gate: float = 0.1, 
                lambda_entropy: float = 0.1):
        """
        Initialize the GPAD Loss.

        Args:
            base_tau (float): The base similarity threshold (0 to 1). Matches with
                            cosine similarity below this value are considered weak.
                            Default: 0.5.
            temp_gate (float): Temperature parameter for the sigmoid gating function.
                            Controls the sharpness of the transition from 0 to 1
                            around the threshold. Lower values make it sharper.
                            Default: 0.1.
            lambda_entropy (float): Coefficient for the entropy term in the
                                    adaptive threshold calculation. Higher values
                                    make the threshold stricter for uncertain samples.
                                    Default: 0.1.
        """
        super().__init__()
        self.base_tau = base_tau
        self.temp_gate = temp_gate
        self.lambda_entropy = lambda_entropy

    def forward(self, 
                embeddings: torch.Tensor, 
                global_prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute the GPAD loss for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): The feature embeddings from the local model.
                                        Shape: [Batch_Size (B), Embedding_Dim (D)]
            global_prototypes (torch.Tensor): The global prototypes (centroids) from the global model.
                                            Shape: [Num_Prototypes (M), Embedding_Dim (D)]

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss (mean over batch).
                        Returns 0.0 if no prototypes are provided.
        """ 
        # 1. Normalization & Similarity
        sims = self._compute_similarity_matrix(embeddings, global_prototypes)
        
        # 2. Uncertainty & Thresholding
        tau_adaptive = self._compute_adaptive_threshold(sims)
        
        # 3. Gating & Anchoring
        max_sim, gate = self._compute_gating(sims, tau_adaptive)
        
        # 4. Loss Computation
        loss = self._compute_anchored_loss(max_sim, gate)
        
        return loss

    def _compute_similarity_matrix(self, embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity matrix between embeddings and prototypes.
        Both inputs are normalized to the unit hypersphere.
        """
        z = F.normalize(embeddings, p=2, dim=1)
        p = F.normalize(prototypes, p=2, dim=1)
        return torch.mm(z, p.t())

    def _compute_adaptive_threshold(self, sims: torch.Tensor) -> torch.Tensor:
        """
        Calculates the adaptive threshold based on assignment entropy.
        Higher entropy (uncertainty) leads to a stricter (higher) threshold.
        """
        B, M = sims.shape
        # Soft assignment distribution
        softmax_all = F.softmax(sims / self.SOFT_ASSIGNMENT_TEMP, dim=1)
        
        # Entropy calculation: -sum(p * log(p))
        entropy = -torch.sum(softmax_all * torch.log(softmax_all + self.EPSILON), dim=1)
        
        # Normalize entropy relative to max possible entropy: log(M)
        max_ent = torch.log(torch.tensor(float(M), device=sims.device))
        ent_norm = entropy / (max_ent + self.EPSILON)
        
        # Compute threshold
        return self.base_tau + self.lambda_entropy * ent_norm

    def _compute_gating(self, sims: torch.Tensor, tau_adaptive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines the gating weights for each sample.
        Returns:
            max_sim (torch.Tensor): Best similarity score for each sample.
            gate (torch.Tensor): Computed gating weight [0, 1].
        """
        max_sim, _ = sims.max(dim=1)
        
        # Binary Anchor Decision: Is similarity strictly above threshold?
        is_anchored = (max_sim > tau_adaptive).float()
        
        # Soft Gating: Sigmoid of the margin
        gate_logit = (max_sim - tau_adaptive)
        gate_sigmoid = torch.sigmoid(gate_logit / self.temp_gate)
        
        # Final Gate
        gate = gate_sigmoid * is_anchored
        return max_sim, gate

    def _compute_anchored_loss(self, max_sim: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted Euclidean distance loss.
        User Formula: L_proto = g * ||z_b - v_j*||
        
        Using cosine relation: ||u - v|| = sqrt(2 * (1 - cos_sim))
        """
        # Distance squared: ||u - v||^2 = 2(1 - cos_cos)
        dist_sq = 2 * (1 - max_sim)
        
        # Linear Distance: ||u - v||
        # Add epsilon for numerical stability of sqrt near 0
        dist = torch.sqrt(dist_sq + self.EPSILON)
        
        loss_per_sample = gate * dist
        return loss_per_sample.mean()