"""Implement the GPAD loss used in the federated training path.

The loss follows the same sequence each time it is called:
1. Normalize the batch embeddings and the current global prototypes.
2. Measure cosine similarity between every embedding and every prototype.
3. Turn the full similarity distribution into an adaptive anchor threshold.
4. Decide which samples are anchored strongly enough to trust the bank.
5. Smooth that hard decision with a sigmoid gate.
6. Penalize anchored embeddings when they drift away from their best match.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPADLoss(nn.Module):
    """Compute prototype anchoring for one mini-batch of embeddings.

    The constructor stores the few knobs that control GPAD:
    1. ``base_tau`` is the minimum similarity needed before any entropy term.
    2. ``temp_gate`` controls how sharp the soft gate becomes near the threshold.
    3. ``lambda_entropy`` decides how much uncertain assignments raise the bar.
    4. ``soft_assign_temp`` shapes the prototype assignment distribution.
    5. ``epsilon`` keeps the entropy math numerically stable.
    """

    def __init__(
        self,
        base_tau: float = 0.5,
        temp_gate: float = 0.1,
        lambda_entropy: float = 0.1,
        soft_assign_temp: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
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
        """Return the mean GPAD loss for one batch.

        The forward path is intentionally short:
        1. Exit with a graph-connected zero if there is no prototype bank yet.
        2. Build the full embedding-to-prototype similarity matrix.
        3. Convert that matrix into one adaptive threshold per sample.
        4. Compute the final soft anchor gate.
        5. Apply the gated distance penalty and average it across the batch.
        """
        if global_prototypes.size(0) == 0 or embeddings.size(0) == 0:
            return embeddings.sum() * 0.0

        similarities = self._compute_similarity_matrix(
            embeddings=embeddings,
            prototypes=global_prototypes,
        )
        tau_adaptive = self._compute_adaptive_threshold(similarities)
        max_similarity, gate = self._compute_gating(similarities, tau_adaptive)
        return self._compute_anchored_loss(max_similarity, gate)

    def _compute_similarity_matrix(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Build the cosine-similarity matrix with shape ``[batch, prototypes]``."""
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_prototypes = F.normalize(prototypes, p=2, dim=1)
        return torch.mm(normalized_embeddings, normalized_prototypes.t())

    def _compute_adaptive_threshold(self, similarities: torch.Tensor) -> torch.Tensor:
        """Turn prototype-assignment entropy into one threshold per sample.

        A confident sample keeps a threshold close to ``base_tau``. A sample
        that spreads probability mass across many prototypes gets a higher
        threshold, which makes anchoring more conservative.
        """
        batch_size, num_prototypes = similarities.shape
        if num_prototypes <= 1:
            return similarities.new_full((batch_size,), self.base_tau)

        assignment = F.softmax(similarities / self.soft_assign_temp, dim=1)
        entropy = -torch.sum(
            assignment * torch.log(assignment.clamp(min=self.epsilon)),
            dim=1,
        )
        max_entropy = torch.log(
            similarities.new_tensor(float(num_prototypes)).clamp(min=1.0)
        )
        normalized_entropy = entropy / max_entropy.clamp(min=self.epsilon)
        return self.base_tau + self.lambda_entropy * normalized_entropy

    def _compute_gating(
        self,
        similarities: torch.Tensor,
        tau_adaptive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the best similarity and the final gate for each sample.

        The gate combines:
        1. The paper's hard ``>=`` anchor rule.
        2. A smooth sigmoid transition around the same threshold.
        This keeps the routing rule crisp while still letting gradients change
        smoothly near the decision boundary.
        """
        max_similarity, _ = similarities.max(dim=1)
        hard_anchor = (max_similarity >= tau_adaptive).to(max_similarity.dtype)
        soft_gate = torch.sigmoid((max_similarity - tau_adaptive) / self.temp_gate)
        return max_similarity, hard_anchor * soft_gate

    def _compute_anchored_loss(
        self,
        max_similarity: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the gated squared distance between anchored embeddings and prototypes."""
        squared_distance = 2.0 * (1.0 - max_similarity)
        return (gate * squared_distance).mean()

    @torch.no_grad()
    def compute_anchor_mask(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Return a boolean mask that marks which samples are anchored.

        The training loop uses this mask only to decide how to route each
        embedding, so the helper runs under ``torch.no_grad()`` and mirrors the
        same threshold logic used in the differentiable forward pass.
        """
        if global_prototypes.size(0) == 0 or embeddings.size(0) == 0:
            return torch.zeros(
                embeddings.size(0),
                dtype=torch.bool,
                device=embeddings.device,
            )

        similarities = self._compute_similarity_matrix(embeddings, global_prototypes)
        tau_adaptive = self._compute_adaptive_threshold(similarities)
        max_similarity, _ = similarities.max(dim=1)
        return max_similarity >= tau_adaptive
