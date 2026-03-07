"""
Gated Prototype Anchored Distillation (GPAD) formulation.

This module provides the core metric-learning regularization objective 
to mitigate catastrophic forgetting in Federated Continual Learning. 
Rather than relying on computationally heavy generative replay or rigid 
weight-consolidation methods (e.g., EWC), GPAD anchors client representations 
directly to a dynamic, non-parametric global prototype bank living on the 
hypersphere.

Theoretical Framework
---------------------
Given an $L_2$-normalized feature vector $z_i \in \mathbb{R}^D$ and a globally 
aggregated prototype bank $V \in \mathbb{R}^{M \\times D}$, GPAD computes a 
confidence-gated distillation loss:

    \mathcal{L}_{GPAD}(z_i) = g(z_i) \cdot \mathbb{D}(z_i, v^*)

Where:
- $\mathbb{D}(z_i, v^*) = 2(1 - \cos(z_i, v^*))$ is the squared Euclidean distance.
- $v^* = \arg\max_{v \in V} \cos(z_i, v)$ is the nearest neighbor prototype.
- $g(z_i) \in [0, 1]$ is a routing gate combining a structural Heaviside step 
  function with a Sigmoid smoothing transition, activated only when the similarity 
  to $v^*$ exceeds an entropy-adaptive threshold $\tau(z_i)$.

Adaptive Entropy Thresholding
-----------------------------
To prevent degenerate feature collapse (forcing $z_i$ into ambiguous or incorrect 
prototypes), $\tau$ is dynamically penalized by the normalized Shannon Entropy 
of the assignment distribution:

    \tau(z_i) = \tau_{base} + \lambda_{H} \frac{\mathcal{H}(\text{softmax}(S / T))}{\log M}

High entropy $\mathcal{H}$ indicates categorical ambiguity, raising $\tau$ to 
reject the anchoring. Low entropy indicates a high-confidence semantic match, 
lowering $\tau$ to allow gradient flow.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GPADLoss(nn.Module):
    """
    Gated Prototype Anchored Distillation Objective.
    
    Operates explicitly on $L_2$-normalized dense embeddings.
    """

    def __init__(
        self,
        base_tau: float = 0.5,
        temp_gate: float = 0.1,
        lambda_entropy: float = 0.1,
        soft_assign_temp: float = 0.1,
        epsilon: float = 1e-8,
    ):
        """
        Hyperparameter configuration.
        
        Args:
            base_tau: $\tau_{base}$, the absolute minimum cosine similarity 
                      required to trigger anchoring at zero entropy.
            temp_gate: Sigmoid temperature scalar controlling the Lipschitz 
                       constant of the gating gradient near the threshold plane.
            lambda_entropy: $\lambda_H$, scaling coefficient for the entropy penalty.
            soft_assign_temp: $T$, the temperature defining the sharpness of the 
                              prototype assignment distribution.
            epsilon: Numerical stability dampener.
        """
        super().__init__()
        self.base_tau = base_tau
        self.temp_gate = temp_gate
        self.lambda_entropy = lambda_entropy
        self.soft_assign_temp = soft_assign_temp
        self.epsilon = epsilon

    def forward(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean mini-batch GPAD objective.
        
        Args:
            embeddings: $Z \in \mathbb{R}^{B \\times D}$
            global_prototypes: $V \in \mathbb{R}^{M \\times D}$
            
        Returns:
            Scalar objective $\mathcal{L}$.
        """
        # Edge case: Federated round 1 has an empty global bank |V| = 0
        if global_prototypes.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # 1. Cosine similarity mapping
        sims = self._compute_similarity_matrix(embeddings, global_prototypes)

        # 2. Entropy-adaptive threshold generation
        tau_adaptive = self._compute_adaptive_threshold(sims)

        # 3. Heaviside/Sigmoid routing modulation
        max_sim, gate = self._compute_gating(sims, tau_adaptive)

        # 4. Squared Euclidean projection
        loss = self._compute_anchored_loss(max_sim, gate)
        return loss

    def _compute_similarity_matrix(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        $S = \frac{Z}{\|Z\|_2} \left(\frac{V}{\|V\|_2}\\right)^T$
        """
        z = F.normalize(embeddings, p=2, dim=1)
        p = F.normalize(prototypes, p=2, dim=1)
        return torch.mm(z, p.t())

    def _compute_adaptive_threshold(self, sims: torch.Tensor) -> torch.Tensor:
        """
        Computes $\tau(z_i)$ via Maximum Entropy penalization.
        """
        B, M = sims.shape

        # Parametric softmax distribution
        softmax_all = F.softmax(sims / self.soft_assign_temp, dim=1)

        # Shannon Entropy vector $H \\in \mathbb{R}^B$
        entropy = -torch.sum(
            softmax_all * torch.log(softmax_all.clamp(min=self.epsilon)), dim=1
        )

        # Min-max normalize against the uniform continuous distribution log(M)
        max_ent = torch.log(torch.tensor(float(M), device=sims.device))
        ent_norm = entropy / (max_ent + self.epsilon)

        return self.base_tau + self.lambda_entropy * ent_norm

    def _compute_gating(
        self,
        sims: torch.Tensor,
        tau_adaptive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the continuous routing gate $g \in [0, 1]$.
        """
        max_sim, _ = sims.max(dim=1)

        # Step function prevents arbitrary gradient injection from noise
        is_anchored = (max_sim > tau_adaptive).to(max_sim.dtype)

        # Sigmoid relaxation enforces $C^1$ continuity near the decision boundary
        gate_logit = max_sim - tau_adaptive
        gate_sigmoid = torch.sigmoid(gate_logit / self.temp_gate)

        gate = gate_sigmoid * is_anchored
        return max_sim, gate

    def _compute_anchored_loss(
        self,
        max_sim: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gated distance objective.
        Because $\|z\|=\|v^*\|=1$, distance simplifies algebraically to cosine derivation.
        """
        # Squared Euclidean proxy: $2(1 - S_{max})$
        dist_sq = 2.0 * (1.0 - max_sim)
        loss_per_sample = gate * dist_sq
        return loss_per_sample.mean()

    @torch.no_grad()
    def compute_anchor_mask(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-mode boolean extraction of the gating step function.
        Used by the local Client routing layer to direct non-anchored features 
        towards the stochastic Novelty Buffer.
        """
        if global_prototypes.size(0) == 0:
            return torch.zeros(
                embeddings.size(0), dtype=torch.bool, device=embeddings.device
            )

        sims = self._compute_similarity_matrix(embeddings, global_prototypes)
        tau_adaptive = self._compute_adaptive_threshold(sims)
        max_sim, _ = sims.max(dim=1)
        
        return max_sim > tau_adaptive

