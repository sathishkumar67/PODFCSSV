"""
Client-Side Components for Federated Continual Self-Supervised Learning.

Two classes compose the client subsystem:

1. ``FederatedClient``
   Represents a single participant (edge device) in the federation.  Manages
   local model training, feature extraction, prototype generation, online
   prototype refinement via EMA, and the novelty buffer for concept discovery.

2. ``ClientManager``
   Factory and dispatch layer that instantiates N clients, assigns them to
   hardware devices, and coordinates training rounds — either sequentially
   (CPU) or in parallel (multi-GPU via ThreadPoolExecutor).

Training Flow Per Round
-----------------------
Round 1 — Initialisation (no global knowledge):
    Loss = MAE reconstruction loss only.
    After training, Spherical K-Means (K = ``k_init_prototypes``) is run on
    all local embeddings to produce the initial local prototypes.

Round > 1 — Continual Learning (global prototypes available):
    Loss = MAE + λ · GPAD (for anchored embeddings only).
    Per-embedding routing classifies each embedding as:
      (a) Anchored   → GPAD distillation loss (alignment with global bank).
      (b) Non-anchored → local prototype EMA update or novelty buffer.

Per-Embedding Routing Decision Flow
------------------------------------
For each embedding z from the encoder:

    1. Compute cosine similarity to ALL global prototypes.  Derive an
       adaptive threshold from the entropy of the similarity distribution.

    2. If max_sim(z, global_bank) > adaptive_threshold:
       → z is **anchored** → GPAD loss applied.
       → Local prototypes are NOT updated.

    3. If max_sim < adaptive_threshold (non-anchored):
       a. Compare z against all local prototypes.
       b. If max_sim(z, local_protos) > local_update_threshold:
          → **EMA-update** the closest local prototype.
       c. Else:
          → z is **truly novel** → append to the novelty buffer.

    4. When the novelty buffer reaches ``novelty_buffer_size``:
       → K-Means (K = ``novelty_k``) on the buffer.
       → Merge resulting centroids into local prototypes (Merge-or-Add).
       → Clear the buffer.

Prototype Types
---------------
- **Global prototypes** — maintained by the server; broadcast to clients.
  Used for GPAD anchoring (preventing forgetting of global concepts).
- **Local prototypes** — maintained per-client; uploaded to the server each
  round.  Summarise the client's private data distribution.  Refined online
  via EMA and periodically via buffer K-Means.

References
----------
[1] He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
[2] McMahan et al., "Communication-Efficient Learning of Deep Networks from
    Decentralized Data", AISTATS 2017.
[3] Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

from __future__ import annotations
import copy
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# FEDERATED CLIENT
# ══════════════════════════════════════════════════════════════════════════

class FederatedClient:
    """Simulates a single edge device (client) in the federation.

    Each client owns:
    - An independent deep copy of the global model (weights are never shared
      directly — only prototypes and final state dicts are communicated).
    - A local prototype bank of L2-normalised vectors that summarise its
      private data distribution.
    - A novelty buffer that accumulates truly novel embeddings until a fresh
      K-Means clustering is triggered to discover new concepts.
    - A local AdamW optimiser for gradient-based training.

    Primary operations per round:
        ``train_epoch()``          — local SGD with per-embedding routing.
        ``generate_prototypes()``  — Round-1 K-Means prototype init.
        ``get_local_prototypes()`` — retrieve current protos (Round > 1).

    Attributes
    ----------
    client_id              : int                    – unique numeric id.
    device                 : torch.device           – hardware device.
    dtype                  : torch.dtype            – input tensor precision.
    model                  : nn.Module              – local model copy.
    optimizer              : torch.optim.Optimizer  – local optimiser.
    local_prototypes       : Optional[Tensor]       – ``[K, D]`` or None.
    local_update_threshold : float                  – local merge threshold.
    local_ema_alpha        : float                  – EMA interpolation factor.
    lambda_proto           : float                  – GPAD loss weight.
    novelty_buffer         : List[Tensor]           – accumulated novel embeddings.
    novelty_buffer_size    : int                    – buffer capacity.
    novelty_k              : int                    – K for buffer K-Means.
    kmeans_max_iters       : int                    – max K-Means iterations.
    kmeans_tol             : float                  – K-Means convergence tol.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        optimizer_cls: type = optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        local_update_threshold: float = 0.7,
        local_ema_alpha: float = 0.1,
        lambda_proto: float = 1.0,
        novelty_buffer_size: int = 500,
        novelty_k: int = 20,
        kmeans_max_iters: int = 100,
        kmeans_tol: float = 1e-4,
    ) -> None:
        """Initialise the federated client.

        Parameters
        ----------
        client_id : int
            Unique identifier (used for logging and device assignment).
        model : nn.Module
            Global model template.  ``copy.deepcopy()`` is called internally
            so each client trains independently.
        device : torch.device
            Hardware device for this client's model and data.
        dtype : torch.dtype
            Floating-point dtype for casting input tensors.
        optimizer_cls : type
            Optimiser class.  Default: AdamW.
        optimizer_kwargs : dict, optional
            Kwargs forwarded to the optimiser (e.g. ``{"lr": 1e-4}``).
            Default: ``{"lr": 1e-3}``.
        local_update_threshold : float
            Min cosine similarity for EMA-updating a local prototype.
            Embeddings below this go to the novelty buffer.
            Range: 0.4–0.8.  Default: 0.7.
        local_ema_alpha : float
            EMA factor for local prototype refinement:
            ``P = normalise((1−α)·P_old + α·z)``.
            Range: 0.05–0.3.  Default: 0.1.
        lambda_proto : float
            GPAD loss weight: total = mae + λ·gpad.
            Range: 0.001–0.1.  Default: 1.0.
        novelty_buffer_size : int
            Buffer capacity before triggering K-Means.  Default: 500.
        novelty_k : int
            K for buffer K-Means clustering.  Default: 20.
        kmeans_max_iters : int
            Max iterations for Spherical K-Means.  Default: 100.
        kmeans_tol : float
            Convergence tolerance for K-Means.  Default: 1e-4.
        """
        self.client_id = client_id
        self.device = device
        self.dtype = dtype
        self.local_update_threshold = local_update_threshold
        self.local_ema_alpha = local_ema_alpha
        self.lambda_proto = lambda_proto
        self.novelty_buffer_size = novelty_buffer_size
        self.novelty_k = novelty_k
        self.kmeans_max_iters = kmeans_max_iters
        self.kmeans_tol = kmeans_tol

        # Local prototype bank [K, D], L2-normalised.  Populated by
        # generate_prototypes() and maintained via EMA + buffer clustering.
        self.local_prototypes: Optional[torch.Tensor] = None

        # Novelty buffer: L2-normalised embeddings that failed both global
        # anchoring and local proto matching.  When full → K-Means.
        self.novelty_buffer: List[torch.Tensor] = []

        # Deep-copy the global model so this client trains independently.
        self.model = copy.deepcopy(model).to(self.device)
        logger.info(f"[Client {self.client_id}] Model copied to {self.device}")

        # Only track TRAINABLE parameters in the optimiser (critical for
        # PEFT: backbone is frozen, only adapter params have requires_grad).
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.optimizer = optimizer_cls(trainable_params, **opt_kwargs)

        logger.info(
            f"[Client {self.client_id}] Initialised | device={self.device} | "
            f"dtype={self.dtype} | opt={optimizer_cls.__name__} | "
            f"ema_α={self.local_ema_alpha} | threshold={self.local_update_threshold} | "
            f"λ_proto={self.lambda_proto} | buf={self.novelty_buffer_size} | "
            f"novelty_k={self.novelty_k}"
        )

    # ==================================================================
    # Feature Extraction (used only by generate_prototypes in Round 1)
    # ==================================================================
    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature embeddings from the ViT encoder backbone.

        Accesses ``model.vit`` (the ViTMAEModel encoder inside
        ViTMAEForPreTraining), takes the last hidden state, and applies global
        average pooling across the sequence dimension to produce one
        D-dimensional feature vector per input sample.

        Note
        ----
        During ``train_epoch()`` this method is NOT called — embeddings are
        extracted directly from the single MAE forward pass's hidden_states
        to ensure mask consistency.  This method is used only during
        ``generate_prototypes()`` (Round 1, under ``torch.inference_mode``).

        Parameters
        ----------
        inputs : ``[B, 3, H, W]``  – images, already on device and dtype.

        Returns
        -------
        torch.Tensor – ``[B, D]`` pooled embeddings.
        """
        encoder_output = self.model.vit(inputs)
        # Global average pool over the sequence dim: [B, L, D] → [B, D].
        return encoder_output.last_hidden_state.mean(dim=1)

    # ==================================================================
    # Local Training
    # ==================================================================
    def train_epoch(
        self,
        dataloader: DataLoader,
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> float:
        """Execute one epoch of local training with per-embedding routing.

        Steps per batch:
          1. **MAE forward pass** with ``output_hidden_states=True``.
          2. **Feature extraction** from the *same* forward pass (encoder
             hidden states, patch tokens only — CLS excluded).
          3. **Per-embedding routing** (Round > 1): anchored → GPAD loss;
             non-anchored → local proto EMA / novelty buffer.
          4. **Backward pass** and optimiser step.

        Loss composition:
          - Round 1:   L = L_MAE
          - Round > 1: L = L_MAE + λ · L_GPAD (anchored only)

        Parameters
        ----------
        dataloader        : DataLoader           – client's private data.
        global_prototypes : ``[M, D]``, optional – server's global bank.
                            None in Round 1.
        gpad_loss_fn      : nn.Module, optional  – GPAD loss module.
                            None in Round 1.

        Returns
        -------
        float – Average total loss over the epoch (detached scalar).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Check whether GPAD regularisation is active this round.
        has_gpad = (global_prototypes is not None) and (gpad_loss_fn is not None)
        if has_gpad:
            logger.info(
                f"[Client {self.client_id}] MAE + GPAD "
                f"(λ={self.lambda_proto}, M={global_prototypes.shape[0]})"
            )
        else:
            logger.info(
                f"[Client {self.client_id}] MAE only (no global prototypes)"
            )

        for batch_idx, batch in enumerate(dataloader):
            # ── Data preparation ─────────────────────────────────────
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            # ── Step 1: Single MAE forward pass ──────────────────────
            # output_hidden_states=True gives us access to the encoder's
            # intermediate representations so we can extract embeddings
            # without a second forward call.
            #
            # Why this matters: ViT-MAE re-samples a random mask on every
            # call.  A separate _extract_features() call would operate on
            # a DIFFERENT mask, making the MAE loss and the prototype
            # embeddings inconsistent.  By extracting from the SAME call
            # both the loss and the hidden states share the same mask.
            outputs = self.model(inputs, output_hidden_states=True)
            mae_loss = getattr(outputs, "loss", None)
            if mae_loss is None:
                # Fallback: differentiable zero keeps the graph intact.
                mae_loss = torch.tensor(
                    0.0, dtype=self.dtype, device=self.device, requires_grad=True
                )
            final_loss = mae_loss

            # ── Step 2: Extract patch embeddings from the same pass ──
            # outputs.hidden_states is a tuple of (1 embed + N layers)
            # tensors, all from the ENCODER.  Shape per tensor:
            #   [B, N_visible + 1, D]
            # where N_visible ≈ 49 (for 75% mask ratio) and index 0 is
            # the CLS token.
            #
            # We take the last encoder layer, slice off the CLS token
            # ([:, 1:, :]) to keep only visual patch features, and
            # average-pool → [B, D].
            #
            # detach() is applied because these embeddings are used ONLY
            # for routing logic — no gradient should flow back through the
            # embedding extraction path.  The model is updated exclusively
            # via the mae_loss (and gpad_loss) backward pass.
            last_hidden = outputs.hidden_states[-1]              # [B, N+1, D]
            embeddings = last_hidden[:, 1:, :].mean(dim=1).detach()  # [B, D]

            # ── Step 3: Per-embedding routing (Round > 1 only) ───────
            if has_gpad and embeddings is not None:
                protos_on_device = global_prototypes.to(self.device)

                # 3a. Classify each embedding as anchored or non-anchored
                #     using the GPAD module's adaptive-threshold logic.
                anchor_mask = gpad_loss_fn.compute_anchor_mask(
                    embeddings, protos_on_device
                )  # [B] boolean

                # 3b. Anchored → GPAD distillation loss (alignment with
                #     the global bank to prevent catastrophic forgetting).
                anchored_embs = embeddings[anchor_mask]
                if anchored_embs.shape[0] > 0:
                    gpad_loss = gpad_loss_fn(anchored_embs, protos_on_device)
                    final_loss = final_loss + self.lambda_proto * gpad_loss

                # 3c. Non-anchored → local prototype EMA or novelty buffer.
                non_anchored_embs = embeddings[~anchor_mask]
                if non_anchored_embs.shape[0] > 0:
                    self._route_non_anchored(non_anchored_embs)

            # ── Step 4: Backward pass & optimiser step ───────────────
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            total_loss += final_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(
            f"[Client {self.client_id}] Epoch done | loss={avg_loss:.6f} | "
            f"batches={num_batches} | buffer={len(self.novelty_buffer)}"
        )
        return avg_loss

    # ==================================================================
    # Per-Embedding Routing: Local Prototype Update & Novelty Buffer
    # ==================================================================
    @torch.no_grad()
    def _route_non_anchored(self, embeddings: torch.Tensor) -> None:
        """Route non-anchored embeddings through local prototype matching.

        For each embedding that did NOT match any global prototype:

        Case A — no local prototypes exist yet:
            Append everything to the novelty buffer.

        Case B — local prototypes exist:
            If cos(z, nearest_local) > ``local_update_threshold``:
                → EMA-update the nearest local prototype.
            Else:
                → append z to the novelty buffer.

        After processing, check if the buffer has reached capacity and
        trigger K-Means clustering if needed.

        Parameters
        ----------
        embeddings : ``[B', D]``  – non-anchored embeddings (B' ≤ B).
        """
        # L2-normalise (runs under @no_grad, so detach is redundant but
        # semantically clear).
        z_norm = F.normalize(embeddings.detach(), p=2, dim=1)  # [B', D]

        # ── Case A: no local protos → buffer everything ──────────────
        if self.local_prototypes is None or self.local_prototypes.shape[0] == 0:
            for i in range(z_norm.shape[0]):
                self.novelty_buffer.append(z_norm[i].clone())
            self._maybe_cluster_buffer()
            return

        # ── Case B: compare against existing local prototypes ────────
        if self.local_prototypes.device != self.device:
            self.local_prototypes = self.local_prototypes.to(self.device)

        # Normalise local protos for valid cosine similarity.
        p_norm = F.normalize(self.local_prototypes, p=2, dim=1)  # [K, D]

        # Similarity matrix: [B', K].
        sims = torch.mm(z_norm, p_norm.t())
        max_sim, best_idx = sims.max(dim=1)  # both [B']

        for i in range(z_norm.shape[0]):
            if max_sim[i] > self.local_update_threshold:
                # ── LOCAL MATCH → EMA update ─────────────────────────
                # Read old_proto from p_norm (the normalised copy used for
                # similarity) to avoid numerical drift from repeated
                # in-place writes to self.local_prototypes.
                proto_idx = best_idx[i]
                old_proto = p_norm[proto_idx]                  # unit-norm
                blended = (
                    (1 - self.local_ema_alpha) * old_proto
                    + self.local_ema_alpha * z_norm[i]         # also unit-norm
                )
                # Re-normalise: the EMA blend of two unit vectors has
                # norm < 1 and must be projected back to the sphere.
                self.local_prototypes[proto_idx] = F.normalize(
                    blended, p=2, dim=0
                )
            else:
                # ── TRULY NOVEL → novelty buffer ─────────────────────
                self.novelty_buffer.append(z_norm[i].clone())

        self._maybe_cluster_buffer()

    def _maybe_cluster_buffer(self) -> None:
        """Trigger K-Means on the novelty buffer once it reaches capacity."""
        if len(self.novelty_buffer) >= self.novelty_buffer_size:
            logger.info(
                f"[Client {self.client_id}] Buffer full "
                f"({len(self.novelty_buffer)} ≥ {self.novelty_buffer_size}) "
                f"→ K-Means clustering"
            )
            self._cluster_novelty_buffer()

    @torch.no_grad()
    def _cluster_novelty_buffer(self) -> None:
        """K-Means on the novelty buffer + Merge-or-Add into local protos.

        Steps:
          1. Stack buffered embeddings → ``[N, D]``, L2-normalise.
          2. Spherical K-Means with K = min(novelty_k, N).
          3. For each centroid:
             - cos(centroid, local_proto) > threshold → **merge** via EMA.
             - else → **add** as a new local prototype.
          4. Clear the buffer.
        """
        if len(self.novelty_buffer) == 0:
            return

        # Stack and normalise.
        buffer_tensor = torch.stack(self.novelty_buffer, dim=0).to(self.device)
        buffer_tensor = F.normalize(buffer_tensor, p=2, dim=1)  # [N, D]

        K = min(self.novelty_k, buffer_tensor.shape[0])
        logger.info(
            f"[Client {self.client_id}] Clustering buffer: "
            f"{buffer_tensor.shape[0]} samples → K={K}"
        )

        new_centroids = self._kmeans(buffer_tensor, K=K)  # [K, D]

        # ── Merge-or-Add into local prototypes ───────────────────────
        if self.local_prototypes is None or self.local_prototypes.shape[0] == 0:
            # First time: all centroids become the initial local protos.
            self.local_prototypes = new_centroids
            merged_count = 0
            added_count = K
        else:
            if self.local_prototypes.device != self.device:
                self.local_prototypes = self.local_prototypes.to(self.device)

            # Normalise both sets for cosine similarity.
            p_norm = F.normalize(self.local_prototypes, p=2, dim=1)
            c_norm = F.normalize(new_centroids, p=2, dim=1)

            # Similarity matrix: [K_new, K_existing].
            sims = torch.mm(c_norm, p_norm.t())
            max_sim, best_idx = sims.max(dim=1)

            # Clone from the NORMALISED copy (p_norm) for correct EMA math.
            # Using self.local_prototypes directly could introduce drift
            # because in-place EMA writes may have shifted values slightly
            # away from the unit sphere.
            updated_protos = p_norm.clone()
            protos_to_add = []
            merged_count = 0
            added_count = 0

            for i in range(new_centroids.shape[0]):
                if max_sim[i] > self.local_update_threshold:
                    # ── MERGE: similar to existing → EMA update ──────
                    idx = best_idx[i]
                    old_proto = updated_protos[idx]            # unit-norm
                    blended = (
                        (1 - self.local_ema_alpha) * old_proto
                        + self.local_ema_alpha * c_norm[i]     # unit-norm
                    )
                    updated_protos[idx] = F.normalize(blended, p=2, dim=0)
                    merged_count += 1
                else:
                    # ── ADD: genuinely new concept ────────────────────
                    protos_to_add.append(new_centroids[i])
                    added_count += 1

            # Combine updated existing protos with newly added ones.
            if len(protos_to_add) > 0:
                new_stack = torch.stack(protos_to_add, dim=0)
                self.local_prototypes = torch.cat(
                    [updated_protos, new_stack], dim=0
                )
            else:
                self.local_prototypes = updated_protos

        # Clear the buffer — all information captured in local protos.
        self.novelty_buffer.clear()

        logger.info(
            f"[Client {self.client_id}] Buffer clustered → "
            f"Merged {merged_count}, Added {added_count} → "
            f"Total {self.local_prototypes.shape[0]} local protos"
        )

    def get_local_prototypes(self) -> Optional[torch.Tensor]:
        """Return current local prototypes ``[K, D]`` (or None if unset)."""
        return self.local_prototypes

    # ==================================================================
    # Round-1 Prototype Generation (Post-Training K-Means)
    # ==================================================================
    @torch.no_grad()
    def generate_prototypes(
        self, dataloader: DataLoader, K_init: int = 10
    ) -> torch.Tensor:
        """Generate initial local prototypes via Spherical K-Means.

        Called ONLY in Round 1, after the initial MAE-only training epoch.
        Subsequent rounds maintain prototypes online via EMA + buffer.

        Pipeline:
          1. Feature extraction — run encoder on the full local dataset.
          2. L2 normalisation — project onto the unit hypersphere.
          3. Spherical K-Means — cluster into K_init centroids.
          4. Store — save a detached clone for online EMA in later rounds.

        Parameters
        ----------
        dataloader : DataLoader – client's private data.
        K_init     : int        – number of prototype clusters.
                                  Range: 5–50.  Default: 10.

        Returns
        -------
        torch.Tensor – ``[K_init, D]``, L2-normalised.
        """
        self.model.eval()
        all_features = []

        logger.info(
            f"[Client {self.client_id}] Extracting features for prototypes..."
        )

        # Step 1: extract features from the full local dataset.
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            with torch.inference_mode():
                features = self._extract_features(inputs)
            all_features.append(features)

        embeddings = torch.cat(all_features, dim=0)  # [N, D]
        logger.info(
            f"[Client {self.client_id}] {embeddings.shape[0]} embeddings "
            f"(dim={embeddings.shape[1]})"
        )

        # Step 2: L2-normalise onto the unit hypersphere.
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Step 3: Spherical K-Means.
        centroids = self._kmeans(embeddings, K=K_init)
        logger.info(
            f"[Client {self.client_id}] K-Means done | K={K_init} | "
            f"shape={centroids.shape}"
        )

        # Step 4: store for online EMA updates in later rounds.
        self.local_prototypes = centroids.detach().clone()
        return centroids

    # ==================================================================
    # Spherical K-Means Clustering (Pure PyTorch)
    # ==================================================================
    def _kmeans(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """Spherical K-Means on L2-normalised embeddings.

        Since all points lie on the unit hypersphere, cosine similarity
        (= dot product for unit vectors) is the natural distance metric.

        Algorithm:
          1. **Init** — Forgy: randomly select K data points as centroids.
          2. **Assign** — cosine similarity via matmul; each point → best centroid.
          3. **Update** — recompute centroids as cluster means, then L2-normalise.
          4. **Converge** — stop when total centroid shift < ``kmeans_tol``
             (both old and new centroids are normalised, so the shift is a
             true displacement on the unit sphere).
          5. **Empty cluster** — re-seed with a random data point.

        Parameters
        ----------
        X : ``[N, D]``  – L2-normalised input data.
        K : int          – number of clusters (clamped to N if N < K).

        Returns
        -------
        torch.Tensor – ``[K, D]``, L2-normalised centroids.
        """
        N, D = X.shape
        K = min(K, N)  # safety clamp

        # Step 1: Forgy initialisation.
        indices = torch.randperm(N, device=X.device)[:K]
        centroids = X[indices].clone()  # [K, D], already unit-norm

        for iteration in range(self.kmeans_max_iters):
            # Normalise centroids (redundant at iter 0 but required after
            # the arithmetic-mean update in later iterations).
            centroids = F.normalize(centroids, p=2, dim=1)

            # Step 2: assignment via dot product (= cosine sim for unit vecs).
            sims = torch.mm(X, centroids.t())  # [N, K]
            _, labels = sims.max(dim=1)         # [N]

            # Step 3: recompute centroids as arithmetic means of clusters.
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    # Empty cluster → re-seed with a random data point.
                    new_centroids[k] = X[torch.randint(0, N, (1,), device=X.device).item()]

            # Step 4: convergence check on the unit sphere.
            # Normalise BEFORE computing shift so that both tensors are
            # unit-norm and the ‖Δ‖ is a true spherical displacement.
            new_centroids = F.normalize(new_centroids, p=2, dim=1)
            center_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids

            if center_shift < self.kmeans_tol:
                logger.info(
                    f"[Client {self.client_id}] K-Means converged at "
                    f"iter {iteration + 1} | shift={center_shift:.6f}"
                )
                break

        # Final normalisation (already done inside loop, but defensive).
        return F.normalize(centroids, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════════════
# CLIENT MANAGER (Factory + Dispatch)
# ══════════════════════════════════════════════════════════════════════════

class ClientManager:
    """Factory and dispatch manager for federated client instances.

    Simulates N edge devices by:
      1. Instantiating N ``FederatedClient`` objects (each with its own
         deep-copied model and device assignment).
      2. Dispatching training commands per round — in parallel (multi-GPU)
         or sequentially (CPU).

    Execution Modes
    ---------------
    **Parallel (GPU)**: Client i → ``cuda:i`` (strict 1:1 mapping).
    Training threads are spawned via ``ThreadPoolExecutor``.  PyTorch
    releases the GIL during CUDA kernels, enabling true parallelism.

    **Sequential (CPU)**: All clients share the CPU and train one after
    another.  Threading is avoided because the GIL would negate any benefit.

    Attributes
    ----------
    clients     : List[FederatedClient]
    num_clients : int
    gpu_count   : int
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_clients: int,
        gpu_count: int = 0,
        dtype: torch.dtype = torch.float32,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        local_update_threshold: float = 0.7,
        local_ema_alpha: float = 0.1,
        lambda_proto: float = 1.0,
        novelty_buffer_size: int = 500,
        novelty_k: int = 20,
        kmeans_max_iters: int = 100,
        kmeans_tol: float = 1e-4,
    ) -> None:
        """Spawn all federated clients.

        Parameters
        ----------
        base_model             – Global model template (deep-copied per client).
        num_clients            – Number of clients to simulate.
        gpu_count              – Available GPUs (0 = CPU mode).
        dtype                  – Input tensor dtype.
        optimizer_kwargs       – Forwarded to each client's optimiser.
        local_update_threshold – Local merge threshold.
        local_ema_alpha        – EMA factor for local protos.
        lambda_proto           – GPAD loss weight.
        novelty_buffer_size    – Buffer capacity.
        novelty_k              – K for buffer K-Means.
        kmeans_max_iters       – Max K-Means iterations.
        kmeans_tol             – K-Means convergence tolerance.
        """
        self.clients: List[FederatedClient] = []
        self.num_clients = num_clients
        self.gpu_count = gpu_count
        self.dtype = dtype
        self.optimizer_kwargs = optimizer_kwargs
        self.local_update_threshold = local_update_threshold
        self.local_ema_alpha = local_ema_alpha
        self.lambda_proto = lambda_proto
        self.novelty_buffer_size = novelty_buffer_size
        self.novelty_k = novelty_k
        self.kmeans_max_iters = kmeans_max_iters
        self.kmeans_tol = kmeans_tol

        self._initialize_clients(base_model)

    # ------------------------------------------------------------------
    def _initialize_clients(self, base_model: nn.Module) -> None:
        """Instantiate all clients with the correct device assignments.

        GPU mode: Client i → cuda:i  (strict 1:1 mapping).
        CPU mode: all clients → cpu.
        """
        logger.info(
            f"[ClientManager] Initialising {self.num_clients} clients..."
        )

        if self.gpu_count > 0:
            if self.num_clients != self.gpu_count:
                raise ValueError(
                    f"1:1 Client-GPU mapping required: {self.num_clients} clients "
                    f"but {self.gpu_count} GPUs."
                )
            logger.info(
                f"[ClientManager] Parallel mode: {self.num_clients} clients "
                f"→ {self.gpu_count} GPUs"
            )
        else:
            logger.info(
                f"[ClientManager] Sequential mode: {self.num_clients} "
                f"clients on CPU"
            )

        for i in range(self.num_clients):
            device = (
                torch.device(f"cuda:{i}")
                if self.gpu_count > 0
                else torch.device("cpu")
            )

            client = FederatedClient(
                client_id=i,
                model=base_model,
                device=device,
                dtype=self.dtype,
                optimizer_kwargs=self.optimizer_kwargs,
                local_update_threshold=self.local_update_threshold,
                local_ema_alpha=self.local_ema_alpha,
                lambda_proto=self.lambda_proto,
                novelty_buffer_size=self.novelty_buffer_size,
                novelty_k=self.novelty_k,
                kmeans_max_iters=self.kmeans_max_iters,
                kmeans_tol=self.kmeans_tol,
            )
            self.clients.append(client)

        logger.info(
            f"[ClientManager] {self.num_clients} clients ready."
        )

    # ------------------------------------------------------------------
    def train_round(
        self,
        dataloaders: List[DataLoader],
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> List[float]:
        """Dispatch one round of local training to ALL clients.

        GPU mode → parallel via ``ThreadPoolExecutor``.
        CPU mode → sequential loop.

        Parameters
        ----------
        dataloaders       : List[DataLoader] – one per client (len must match).
        global_prototypes : ``[M, D]``, optional – for GPAD.  None in Round 1.
        gpad_loss_fn      : nn.Module, optional  – GPAD module.  None in Round 1.

        Returns
        -------
        List[float] – average loss per client.  ``nan`` on failure.

        Raises
        ------
        ValueError – if ``len(dataloaders) != num_clients``.
        """
        if len(dataloaders) != self.num_clients:
            raise ValueError(
                f"Dataloader count ({len(dataloaders)}) ≠ "
                f"client count ({self.num_clients})"
            )

        round_losses = [0.0] * self.num_clients

        if self.gpu_count > 0:
            # ── Parallel: one thread per GPU ─────────────────────────
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
                logger.info(
                    f"[ClientManager] Spawning {self.num_clients} threads "
                    f"(1 per GPU)..."
                )
                futures = {}
                for i, client in enumerate(self.clients):
                    futures[
                        executor.submit(
                            client.train_epoch,
                            dataloaders[i],
                            global_prototypes=global_prototypes,
                            gpad_loss_fn=gpad_loss_fn,
                        )
                    ] = i

                for future in futures:
                    idx = futures[future]
                    try:
                        loss = future.result()
                        round_losses[idx] = loss
                        logger.info(
                            f"[ClientManager] Client {idx} "
                            f"(GPU {idx}) done | loss={loss:.4f}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[ClientManager] Client {idx} FAILED: {e}"
                        )
                        round_losses[idx] = float("nan")
        else:
            # ── Sequential: CPU ──────────────────────────────────────
            logger.info(
                f"[ClientManager] Sequential training on CPU "
                f"({self.num_clients} clients)..."
            )
            for i, client in enumerate(self.clients):
                try:
                    loss = client.train_epoch(
                        dataloaders[i],
                        global_prototypes=global_prototypes,
                        gpad_loss_fn=gpad_loss_fn,
                    )
                    round_losses[i] = loss
                    logger.info(
                        f"[ClientManager] Client {i} (CPU) done | "
                        f"loss={loss:.4f}"
                    )
                except Exception as e:
                    logger.error(
                        f"[ClientManager] Client {i} FAILED: {e}"
                    )
                    round_losses[i] = float("nan")

        return round_losses