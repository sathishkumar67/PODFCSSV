"""
Gated Prototype Anchored Distillation (GPAD) Loss Module.

This module implements a knowledge-distillation-style regularization loss
specifically designed for Federated Continual Self-Supervised Learning. The
core objective is to prevent catastrophic forgetting by anchoring the local
client's evolving feature representations to a set of globally consensus
prototypes maintained by the central server.

Unlike conventional distillation losses (e.g., KL divergence between logits),
GPAD operates directly in the feature embedding space and uses a gated
anchoring mechanism to selectively enforce alignment only for confident
matches, avoiding negative transfer from ambiguous or noisy prototype
assignments.

Mathematical Formulation
------------------------
For a single L2-normalized embedding z ∈ R^D produced by the client's
encoder, the per-sample GPAD loss is:

    L_GPAD(z) = g(z) · ||z - v*(z)||₂

where:
    v*(z) = argmax_{v ∈ V_global} cos(z, v)
        → the nearest global prototype by cosine similarity.

    g(z) = σ( (s*(z) - τ(z)) / T_gate ) · 𝟙[s*(z) > τ(z)]
        → a soft gate ∈ [0,1] that activates only when the best-match
          similarity s*(z) exceeds an adaptive threshold τ(z).

    τ(z) = τ_base + λ_ent · H_norm(z)
        → the adaptive threshold, which rises with the normalized entropy
          H_norm of the similarity distribution, penalizing ambiguous
          assignments where z is roughly equidistant to many prototypes.

    H_norm(z) = H( softmax(sim(z, V) / T_soft) ) / log(M)
        → Shannon entropy of the soft assignment distribution, normalized
          by the maximum possible entropy log(M) so that the penalty is
          invariant to the number of prototypes M.

The batch loss is simply the mean over all samples:
    L = (1/B) Σ_i L_GPAD(z_i)

Design Rationale
----------------
1. **Adaptive Threshold**: A fixed threshold would either over-anchor
   (forcing alignment to poor matches) or under-anchor (ignoring valid
   matches) depending on the prototype bank's state. The entropy-adaptive
   threshold automatically becomes stricter when assignments are ambiguous,
   and more permissive when a clear best-match prototype exists.

2. **Hard + Soft Gating**: The hard binary mask 𝟙[s* > τ] ensures that
   clearly non-matching samples contribute exactly zero loss (no noisy
   gradients), while the sigmoid provides a smooth gradient landscape near
   the decision boundary, avoiding discontinuities during backpropagation.

3. **Euclidean Distance on the Unit Sphere**: Since both z and v* are
   L2-normalized, the squared Euclidean distance simplifies to 2(1 - cos(z, v*)),
   making the loss a monotonic function of cosine similarity. Using the
   linear (not squared) distance avoids gradient magnitudes that vanish as
   the embedding approaches its target.

References
----------
[1] Tian et al., "Contrastive Representation Distillation", ICLR 2020.
[2] Li \u0026 Hoiem, "Learning without Forgetting", IEEE TPAMI 2018.
[3] McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GPADLoss(nn.Module):
    """
    Gated Prototype Anchored Distillation (GPAD) Loss.

    This loss regularizes client feature spaces against a global prototype bank
    to prevent catastrophic forgetting in a federated continual learning setting.
    It uses a confidence-gated mechanism that selectively anchors embeddings only
    when the prototype assignment is unambiguous, avoiding negative transfer from
    noisy or uncertain matches.

    The loss consists of four sequential stages computed per forward pass:
        1. Cosine similarity computation between embeddings and all prototypes.
        2. Adaptive threshold derivation from the entropy of each embedding's
           similarity distribution.
        3. Gated anchor selection via hard mask + soft sigmoid weighting.
        4. Weighted Euclidean distance loss for anchored samples only.

    Attributes
    ----------
    base_tau : float
        Minimum similarity threshold for anchoring in the zero-entropy
        (maximally confident) case. Higher values make anchoring stricter.
    temp_gate : float
        Temperature controlling the steepness of the sigmoid gate. Lower
        values produce a sharper, near-binary gate; higher values yield a
        smoother transition and more gradient flow near the boundary.
    lambda_entropy : float
        Scaling factor that controls how strongly assignment uncertainty
        (entropy) raises the anchoring threshold. Higher values make the
        gate more conservative under ambiguous prototype assignments.
    soft_assign_temp : float
        Temperature for the softmax used to convert raw cosine similarities
        into a probability distribution before computing entropy. Lower
        values sharpen the distribution (making entropy more discriminative).
    epsilon : float
        Small numerical constant added to prevent division by zero in
        entropy normalization and to stabilize the sqrt in distance computation.
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
        Initialize the GPAD loss module.

        Parameters
        ----------
        base_tau : float
            Base similarity threshold for confident anchoring. An embedding is
            only anchored to its best-match global prototype if the best cosine
            similarity exceeds (base_tau + entropy_penalty). Intuitively, this
            sets the "floor" confidence required even when the assignment is
            perfectly unambiguous. Range: 0.3–0.7. Default: 0.5.
        temp_gate : float
            Temperature parameter for the sigmoid gating function. The gate is
            computed as σ((sim - threshold) / temp_gate). Smaller values create
            a sharp transition (approaching a step function), while larger values
            produce a smoother gate with more gradient signal near the decision
            boundary. Range: 0.05–0.5. Default: 0.1.
        lambda_entropy : float
            Weight that scales the normalized entropy penalty added to base_tau.
            The effective threshold is: τ = base_tau + lambda_entropy * H_norm.
            Higher values make the gate more conservative when the embedding's
            similarity distribution is spread across many prototypes (high
            uncertainty). Range: 0.1–0.5. Default: 0.1.
        soft_assign_temp : float
            Temperature dividing the raw cosine similarities before the softmax
            that produces the probability distribution for entropy calculation:
            p = softmax(sim / soft_assign_temp). Lower temperatures make the
            distribution peakier (low entropy for clear matches, high entropy
            for ambiguous ones). Range: 0.05–0.5. Default: 0.1.
        epsilon : float
            Numerical stability constant. Used in three places:
            (a) inside log() during entropy computation to prevent log(0),
            (b) in the entropy normalization denominator,
            (c) inside sqrt() when computing Euclidean distance from cosine
            similarity. Default: 1e-8.
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
        Compute the batch-averaged GPAD distillation loss.

        This method implements the full four-stage GPAD pipeline:

        Stage 1 — Similarity: L2-normalize both inputs and compute the full
            pairwise cosine similarity matrix S ∈ R^{B×M} via matrix
            multiplication, where B = batch size and M = number of global
            prototypes.

        Stage 2 — Adaptive Threshold: For each embedding, derive a per-sample
            threshold τ_i that adapts to the uncertainty (entropy) of its
            similarity distribution across prototypes. Ambiguous assignments
            get a stricter (higher) threshold.

        Stage 3 — Gating: For each embedding, extract the best-match similarity
            s*_i and evaluate a two-part gate: (a) a hard binary check ensuring
            s*_i > τ_i, and (b) a soft sigmoid weight providing smooth gradients.
            The product of these two yields the final gate g_i ∈ [0, 1].

        Stage 4 — Loss: Convert the best-match cosine similarity to Euclidean
            distance on the unit sphere, weight it by the gate, and average
            across the batch.

        Parameters
        ----------
        embeddings : torch.Tensor
            Feature embeddings from the client's local encoder, shape [B, D].
            These do NOT need to be pre-normalized; normalization is applied
            internally.
        global_prototypes : torch.Tensor
            The current global prototype bank from the server, shape [M, D].
            Also normalized internally before similarity computation.

        Returns
        -------
        torch.Tensor
            Scalar loss (0-dim tensor) representing the mean gated distance
            across the batch. Returns a differentiable zero if the prototype
            bank is empty (M = 0).
        """
        # Edge case: empty prototype bank → no anchoring possible, return
        # a differentiable zero so that backward() does not break.
        if global_prototypes.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Stage 1: Compute the full pairwise cosine similarity matrix [B, M].
        sims = self._compute_similarity_matrix(embeddings, global_prototypes)

        # Stage 2: Derive per-sample adaptive thresholds [B] from entropy.
        tau_adaptive = self._compute_adaptive_threshold(sims)

        # Stage 3: Compute per-sample gate values [B] and best similarities [B].
        max_sim, gate = self._compute_gating(sims, tau_adaptive)

        # Stage 4: Gate-weighted Euclidean distance loss → scalar.
        loss = self._compute_anchored_loss(max_sim, gate)

        return loss

    def _compute_similarity_matrix(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the cosine similarity matrix between embeddings and prototypes.

        Both inputs are first L2-normalized along the feature dimension, after
        which cosine similarity reduces to a simple matrix multiplication
        (dot product). This normalization ensures that all similarity values
        lie in [-1, 1], making threshold comparisons scale-invariant.

        Parameters
        ----------
        embeddings : torch.Tensor
            Client feature embeddings, shape [B, D].
        prototypes : torch.Tensor
            Global prototype bank, shape [M, D].

        Returns
        -------
        torch.Tensor
            Cosine similarity matrix of shape [B, M], where entry (i, j) is
            the cosine similarity between the i-th embedding and the j-th
            global prototype.
        """
        # Project both sets of vectors onto the unit hypersphere so that
        # dot product == cosine similarity.
        z = F.normalize(embeddings, p=2, dim=1)  # [B, D] → unit norm per row
        p = F.normalize(prototypes, p=2, dim=1)  # [M, D] → unit norm per row

        # Matrix multiplication yields the full similarity matrix: [B, M].
        return torch.mm(z, p.t())

    def _compute_adaptive_threshold(self, sims: torch.Tensor) -> torch.Tensor:
        """
        Derive a per-sample adaptive anchoring threshold from assignment entropy.

        The idea is that when an embedding is roughly equidistant to many
        prototypes (high entropy in the soft assignment distribution), we should
        be *more conservative* about anchoring — the best match may just be
        noise. Conversely, when the similarity distribution is sharply peaked
        at one prototype (low entropy), the match is trustworthy and we can
        use a lower threshold.

        Computation:
            1. Convert cosine similarities to a probability distribution via
               temperature-scaled softmax: p_j = softmax(s_j / T_soft).
            2. Compute Shannon entropy: H = -Σ_j p_j log(p_j).
            3. Normalize to [0, 1] by dividing by the maximum possible entropy
               H_max = log(M), where M is the number of prototypes.
            4. Final threshold: τ_i = base_tau + lambda_entropy * H_norm_i.

        Parameters
        ----------
        sims : torch.Tensor
            Cosine similarity matrix of shape [B, M].

        Returns
        -------
        torch.Tensor
            Per-sample adaptive threshold of shape [B]. Each value is in the
            range [base_tau, base_tau + lambda_entropy].
        """
        B, M = sims.shape

        # Temperature-scaled softmax converts raw cosine similarities into a
        # probability distribution over prototypes. Lower temperature makes
        # the distribution peakier (low entropy for clear matches).
        softmax_all = F.softmax(sims / self.soft_assign_temp, dim=1)  # [B, M]

        # Shannon entropy of the soft assignment distribution. The epsilon
        # inside log prevents log(0) for any near-zero probabilities.
        entropy = -torch.sum(
            softmax_all * torch.log(softmax_all + self.epsilon), dim=1
        )  # [B]

        # Normalize entropy to [0, 1] by dividing by the theoretical maximum
        # entropy log(M) (achieved by a uniform distribution over M prototypes).
        # This ensures the penalty magnitude is invariant to bank size M.
        max_ent = torch.log(torch.tensor(float(M), device=sims.device))
        ent_norm = entropy / (max_ent + self.epsilon)  # [B], values in [0, 1]

        # Final adaptive threshold: base floor + entropy-scaled penalty.
        # When entropy is 0 (perfect match), threshold = base_tau.
        # When entropy is maximal (uniform), threshold = base_tau + lambda_entropy.
        return self.base_tau + self.lambda_entropy * ent_norm

    def _compute_gating(
        self,
        sims: torch.Tensor,
        tau_adaptive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the confidence gate that determines how strongly each sample's
        loss is enforced.

        The gate combines two mechanisms:
            (a) A **hard binary mask** that strictly zeros out any sample whose
                best-match similarity does not exceed the adaptive threshold.
                This prevents any gradient flow from clearly non-matching
                embeddings (important for avoiding noisy updates in FL).
            (b) A **soft sigmoid weight** that provides a smooth, differentiable
                transition near the decision boundary. This is crucial for stable
                training because a pure hard gate would produce zero gradients
                everywhere except at the exact threshold point.

        The final gate is the product: g = sigmoid_weight * hard_mask.

        Parameters
        ----------
        sims : torch.Tensor
            Cosine similarity matrix, shape [B, M].
        tau_adaptive : torch.Tensor
            Per-sample adaptive thresholds, shape [B].

        Returns
        -------
        max_sim : torch.Tensor
            Best-match cosine similarity for each sample, shape [B].
        gate : torch.Tensor
            Final gating weight for each sample, shape [B]. Values are in
            [0, 1], with 0 meaning "do not anchor" and 1 meaning "fully anchor".
        """
        # Extract the highest similarity score across all prototypes for each
        # embedding. This identifies the best candidate anchor prototype.
        max_sim, _ = sims.max(dim=1)  # [B]

        # Hard mask: strict binary decision. True (1.0) if the best similarity
        # exceeds the adaptive threshold, False (0.0) otherwise. This ensures
        # zero gradient contribution from clearly non-matching samples.
        is_anchored = (max_sim > tau_adaptive).float()  # [B]

        # Soft sigmoid gate: provides smooth gradient near the decision boundary.
        # The logit is the margin (sim - threshold), so the sigmoid is centered
        # exactly at the threshold. temp_gate controls the steepness:
        #   - Small temp → sharp transition (near step function)
        #   - Large temp → gentle slope (more gradient signal but less selective)
        gate_logit = max_sim - tau_adaptive
        gate_sigmoid = torch.sigmoid(gate_logit / self.temp_gate)  # [B]

        # Final gate: product of hard mask and soft weight. The hard mask
        # ensures exact zeros for non-matches (no gradient leakage), while
        # the sigmoid modulates the strength for accepted matches.
        gate = gate_sigmoid * is_anchored  # [B]

        return max_sim, gate

    def _compute_anchored_loss(
        self,
        max_sim: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gate-weighted Euclidean distance loss for anchored samples.

        For L2-normalized vectors z and v on the unit sphere, the squared
        Euclidean distance has a closed-form relationship with cosine similarity:

            ||z - v||² = ||z||² - 2·z·v + ||v||² = 1 - 2·cos(z,v) + 1 = 2(1 - S)
            ||z - v||  = √(2(1 - S))

        Using the linear distance (not squared) ensures that the gradient signal
        does not vanish as the embedding approaches its target prototype
        (unlike squared distance, which has zero gradient at the optimum).

        The gate modulates the loss per sample:
            - gate ≈ 1.0: High-confidence anchor → full distance penalty.
            - gate ≈ 0.0: Low-confidence / non-anchor → zero penalty.

        Parameters
        ----------
        max_sim : torch.Tensor
            Best-match cosine similarity for each sample, shape [B].
        gate : torch.Tensor
            Per-sample gating weight in [0, 1], shape [B].

        Returns
        -------
        torch.Tensor
            Scalar mean loss across the batch (0-dim tensor).
        """
        # Convert cosine similarity to squared Euclidean distance on the
        # unit sphere. This is exact for unit-norm vectors.
        dist_sq = 2 * (1 - max_sim)  # [B], values in [0, 4]

        # Take the square root to get linear Euclidean distance. We add
        # epsilon inside the sqrt for numerical stability when dist_sq ≈ 0
        # (i.e., when z and v* are nearly identical).
        dist = torch.sqrt(dist_sq + self.epsilon)  # [B], values in [0, 2]

        # Weight each sample's distance by its gate value. Non-anchored
        # samples (gate = 0) contribute exactly zero to the loss.
        loss_per_sample = gate * dist  # [B]

        # Return the batch mean as the final scalar loss.
        return loss_per_sample.mean()

    # ======================================================================
    # Anchor Mask — used by FederatedClient for per-embedding routing
    # ======================================================================
    @torch.no_grad()
    def compute_anchor_mask(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce a boolean mask classifying each embedding as "anchored" or "novel".

        This method implements the same similarity + adaptive threshold logic as
        the forward pass but returns a binary decision rather than a loss value.
        It is called by FederatedClient.train_epoch() to route each embedding in
        the batch through the per-embedding decision flow:

            1. Anchored (True)     → GPAD loss is applied to pull this embedding
                                     toward its best-match global prototype.
            2. Non-anchored (False) → The embedding is routed to local prototype
                                     matching or the novelty buffer instead.

        This separation ensures that GPAD regularization is only applied to
        embeddings that confidently match an existing global concept, while
        genuinely novel data is allowed to form new local prototypes without
        interference from the global bank.

        Parameters
        ----------
        embeddings : torch.Tensor
            Batch of feature embeddings from the client encoder, shape [B, D].
        global_prototypes : torch.Tensor
            Current global prototype bank from the server, shape [M, D].

        Returns
        -------
        torch.Tensor
            Boolean mask of shape [B]. True = anchored (known concept),
            False = non-anchored (potentially novel concept).
        """
        # Edge case: if no global prototypes exist (e.g., Round 1), nothing
        # can be anchored — return all-False mask.
        if global_prototypes.size(0) == 0:
            return torch.zeros(
                embeddings.size(0), dtype=torch.bool, device=embeddings.device
            )

        # Compute full similarity matrix and per-sample adaptive thresholds.
        sims = self._compute_similarity_matrix(embeddings, global_prototypes)
        tau_adaptive = self._compute_adaptive_threshold(sims)

        # An embedding is "anchored" iff its best match exceeds its adaptive
        # threshold → it confidently belongs to an existing global concept.
        max_sim, _ = sims.max(dim=1)
        return max_sim > tau_adaptive