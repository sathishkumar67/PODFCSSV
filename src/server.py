"""
Server-Side Components for Federated Continual Self-Supervised Learning.

Two core server responsibilities live here:

1. ``GlobalPrototypeBank``
   A dynamically growing set of L2-normalised prototype vectors on the unit
   hypersphere.  Each vector represents a visual concept discovered across all
   clients.  New local prototypes from each round are integrated via a
   "Merge-or-Add" strategy with Exponential Moving Average (EMA) updates.

2. ``FederatedModelServer``
   Standard Federated Averaging (FedAvg) for model weight aggregation:
   element-wise arithmetic mean of all participating clients' state dicts.

3. ``run_server_round``
   Convenience function that executes both prototype merging and weight
   aggregation for one communication round.

4. ``GlobalModel``
   Wrapper around the server-side model instance.  Handles pre-trained
   checkpoint loading and weight updates from FedAvg.

Merge-or-Add Strategy
---------------------
- **Merge**: Incoming prototype with cosine sim ≥ ``merge_threshold`` to an
  existing global prototype → update in-place via EMA:
  ``G_new = normalise((1 − α) · G_old + α · P_local)``.
- **Add**: Cosine sim below threshold → the prototype encodes a genuinely novel
  concept and is appended to the bank (subject to ``max_prototypes`` cap).

References
----------
[1] McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.
[2] Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# 1. GLOBAL PROTOTYPE BANK
# ══════════════════════════════════════════════════════════════════════════

class GlobalPrototypeBank:
    """Server-side global prototype bank.

    Manages a dynamically sized collection of L2-normalised prototype vectors.
    Each vector is a cluster centroid representing a distinct visual concept
    discovered across the federation.

    The bank implements an online Merge-or-Add strategy:
      1. Compute cosine similarity of the incoming prototype vs ALL globals.
      2. If best match ≥ ``merge_threshold`` → **merge** via EMA.
      3. Else → **add** as a new global prototype (if capacity allows).

    All vectors are kept L2-normalised so that cosine similarity = dot product.

    Attributes
    ----------
    embedding_dim    : int            – Feature dimensionality D.
    merge_threshold  : float          – Server-side merge threshold.
    ema_alpha        : float          – EMA interpolation factor.
    device           : torch.device   – Computation device.
    max_prototypes   : int            – Bank capacity limit.
    prototypes       : torch.Tensor   – ``[M, D]`` prototype matrix.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        merge_threshold: float = 0.8,
        ema_alpha: float = 0.1,
        device: str = "cpu",
        max_prototypes: int = 50,
    ) -> None:
        """Initialise the global prototype bank.

        Parameters
        ----------
        embedding_dim   : int   – Must match encoder output dim (e.g. 768 for
                                  ViT-Base).  Default: 768.
        merge_threshold : float – If cos(incoming, best_global) ≥ this value
                                  the global proto is updated via EMA; otherwise
                                  a new entry is added.  Range: 0.5–0.85.
                                  Default: 0.8.
        ema_alpha       : float – EMA weight for prototype update:
                                  ``G = norm((1−α)·G_old + α·P_new)``.
                                  Range: 0.01–0.2.  Default: 0.1.
        device          : str   – 'cpu', 'cuda', or 'cuda:N'.  Default: 'cpu'.
        max_prototypes  : int   – Maximum bank capacity.  Once full, only
                                  merges are allowed.  Range: 20–200.  Default: 50.
        """
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)
        self.max_prototypes = max_prototypes

        # Start with an empty bank [0, D].
        self.prototypes = torch.zeros(0, embedding_dim, device=self.device)

    # ------------------------------------------------------------------
    # Core aggregation: merge incoming local prototypes into the bank
    # ------------------------------------------------------------------
    @torch.no_grad()
    def merge_local_prototypes(
        self,
        local_protos_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Integrate local prototypes from all clients into the global bank.

        All incoming prototypes are concatenated, L2-normalised, and processed
        **sequentially** against the current bank.  Sequential processing is
        necessary because each prototype can modify the bank (merge or add),
        and subsequent prototypes must see the updated state — e.g. two clients
        that independently discover the same concept should not both add it.

        Parameters
        ----------
        local_protos_list : List[torch.Tensor]
            One tensor per client, each ``[K_i, D]``.  They do not need to
            be pre-normalised.

        Returns
        -------
        torch.Tensor
            Updated bank ``[M_new, D]``.  M_new ≥ M_old (bank never shrinks).
        """
        if not local_protos_list:
            return self.prototypes

        # Concatenate all client prototypes into one batch and normalise.
        incoming = torch.cat(local_protos_list, dim=0).to(self.device)
        incoming = F.normalize(incoming, p=2, dim=1)

        # --- Round 1: bank is empty → seed with all incoming protos ---
        if self.prototypes.size(0) == 0:
            if (
                self.max_prototypes is not None
                and incoming.size(0) > self.max_prototypes
            ):
                logger.warning(
                    f"Round 1 incoming ({incoming.size(0)}) exceeds max capacity "
                    f"({self.max_prototypes}).  Truncating."
                )
                self.prototypes = incoming[: self.max_prototypes]
            else:
                self.prototypes = incoming
            return self.prototypes

        # --- Round > 1: process each incoming prototype sequentially ---
        for i in range(incoming.size(0)):
            p_new = incoming[i]  # single vector [D]

            # Defensive re-normalisation: EMA can cause slight norm drift.
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

            # Cosine similarity vs every global prototype (dot product).
            sims = torch.mv(self.prototypes, p_new)  # [M]
            max_sim, best_idx = sims.max(dim=0)

            if max_sim >= self.merge_threshold:
                # ── MERGE ──
                # The incoming prototype matches an existing concept.
                # Blend via EMA and re-normalise to the unit sphere.
                old_vec = self.prototypes[best_idx]
                blended = (1 - self.ema_alpha) * old_vec + self.ema_alpha * p_new
                self.prototypes[best_idx] = F.normalize(blended, p=2, dim=0)
            else:
                # ── ADD ──
                # Novel concept.  Append if capacity allows.
                if (
                    self.max_prototypes is None
                    or self.prototypes.size(0) < self.max_prototypes
                ):
                    self.prototypes = torch.cat(
                        [self.prototypes, p_new.unsqueeze(0)], dim=0
                    )
                else:
                    logger.info(
                        f"Bank at capacity ({self.max_prototypes}).  "
                        f"Skipping novel prototype."
                    )

        return self.prototypes

    def get_prototypes(self) -> torch.Tensor:
        """Return the current bank ``[M, D]`` (empty ``[0, D]`` if uninitialised)."""
        return self.prototypes


# ══════════════════════════════════════════════════════════════════════════
# 2. FEDERATED MODEL SERVER (FedAvg)
# ══════════════════════════════════════════════════════════════════════════

class FederatedModelServer:
    """Federated Averaging (FedAvg) weight aggregation server.

    Computes the element-wise arithmetic mean of all client state dicts to
    produce a single global consensus model each round.  Assumes uniform
    weighting (equal data contribution per client).

    Notes
    -----
    - Tensors keep their original dtype during aggregation (no implicit
      conversion to float32).
    - If a parameter key is missing from any client, that key is skipped
      entirely (with a warning) to avoid biased averaging over a partial set.
    """

    @torch.no_grad()
    def aggregate_weights(
        self,
        client_weights_map: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client weights via FedAvg (uniform mean).

        Parameters
        ----------
        client_weights_map : Dict[str, Dict[str, torch.Tensor]]
            ``{client_id: state_dict}``.  All clients must share the same
            model architecture.

        Returns
        -------
        Dict[str, torch.Tensor]
            Global state dict.  Empty dict if no weights received.
        """
        if not client_weights_map:
            logger.warning("No client weights received for aggregation.")
            return {}

        client_ids = list(client_weights_map.keys())
        num_clients = len(client_ids)

        # Single client → no averaging needed.
        if num_clients == 1:
            return client_weights_map[client_ids[0]]

        # Use the first client as a structural template for the expected keys.
        reference_keys = client_weights_map[client_ids[0]].keys()

        global_state: Dict[str, torch.Tensor] = {}

        for key in reference_keys:
            # Collect this parameter from every client.
            tensors = [
                client_weights_map[cid][key]
                for cid in client_ids
                if key in client_weights_map[cid]
            ]

            if len(tensors) == num_clients:
                # All clients contributed → stack along dim 0 and average.
                global_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                # Some clients missing this key → skip to avoid bias.
                logger.warning(
                    f"'{key}' missing in {num_clients - len(tensors)}/{num_clients} "
                    f"clients — skipping aggregation for this parameter."
                )

        return global_state


# ══════════════════════════════════════════════════════════════════════════
# 3. ROUND ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════

def run_server_round(
    proto_manager: GlobalPrototypeBank,
    model_server: FederatedModelServer,
    client_payloads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute the complete server-side logic for one communication round.

    1. **Prototype aggregation**: extract local prototypes from all client
       payloads and merge them into the global bank (Merge-or-Add + EMA).
    2. **Weight aggregation**: extract local state dicts and compute the
       FedAvg global weights (element-wise mean).

    Parameters
    ----------
    proto_manager   : GlobalPrototypeBank     – Modified in-place by merge.
    model_server    : FederatedModelServer     – Stateless aggregator.
    client_payloads : List[Dict[str, Any]]
        Each dict should contain:
          - ``'client_id'`` (str)
          - ``'protos'`` (torch.Tensor, optional) – ``[K_i, D]``
          - ``'weights'`` (Dict[str, torch.Tensor])

    Returns
    -------
    Dict[str, Any]
        ``{'global_prototypes': Tensor, 'global_weights': state_dict}``.
        Empty dict if no payloads.
    """
    if not client_payloads:
        return {}

    # Step 1 — Prototype aggregation.
    protos = [p["protos"] for p in client_payloads if "protos" in p]
    global_protos = proto_manager.merge_local_prototypes(protos)

    # Step 2 — Weight aggregation (FedAvg).
    client_weights_map = {}
    for i, p in enumerate(client_payloads):
        if "weights" in p:
            cid = p.get("client_id", f"unknown_client_{i}")
            client_weights_map[cid] = p["weights"]

    global_weights = model_server.aggregate_weights(client_weights_map)

    return {
        "global_prototypes": global_protos,
        "global_weights": global_weights,
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. GLOBAL MODEL WRAPPER
# ══════════════════════════════════════════════════════════════════════════

class GlobalModel:
    """Server-side global model wrapper.

    Handles two jobs:
      1. **Initialisation** – Loads a ``ViTMAEForPreTraining`` checkpoint,
         injects IBA adapters, and freezes the backbone.
      2. **Weight updates** – After each round's FedAvg aggregation, loads
         the averaged weights into the model (``strict=False`` to handle
         partial adapter-only state dicts).

    Attributes
    ----------
    device : torch.device  – Computation device.
    model  : nn.Module     – The actual PyTorch model.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Load a pre-trained ViT-MAE and inject adapters.

        Falls back to a lightweight ``nn.Linear`` stub if the required
        libraries are unavailable, allowing pipeline testing without
        HuggingFace dependencies.

        Parameters
        ----------
        device : str – 'cpu', 'cuda', or 'cuda:N'.  Default: 'cpu'.
        """
        self.device = torch.device(device)
        logger.info(f"Initialising Global Model on {self.device}...")

        try:
            from transformers import ViTMAEForPreTraining
            from src.mae_with_adapter import inject_adapters

            self.model = ViTMAEForPreTraining.from_pretrained(
                "facebook/vit-mae-base"
            )
            self.model = inject_adapters(self.model)
            self.model.to(self.device)
            self.model.eval()  # server model is never trained directly

            logger.info("Global Model initialised (pre-trained + adapters).")

        except ImportError as e:
            logger.warning(f"Import failed ({e}) — falling back to stub model.")
            self.model = torch.nn.Linear(10, 10).to(self.device)

        except Exception as e:
            logger.error(f"Model init error: {e}")
            raise

    def update_model_weights(
        self,
        aggregated_weights: Dict[str, torch.Tensor],
    ) -> None:
        """Load FedAvg-aggregated weights into the global model.

        Uses ``strict=False`` because clients typically send back only the
        trainable adapter parameters, not the frozen backbone weights.

        Parameters
        ----------
        aggregated_weights : Dict[str, torch.Tensor]
            Global state dict from ``FederatedModelServer.aggregate_weights()``.
        """
        if not aggregated_weights:
            logger.warning("Empty weight update — skipping.")
            return

        try:
            keys = self.model.load_state_dict(aggregated_weights, strict=False)
            logger.info(
                f"Global model updated.  "
                f"Missing keys: {len(keys.missing_keys)}, "
                f"Unexpected keys: {len(keys.unexpected_keys)}"
            )
        except Exception as e:
            logger.error(f"Failed to update global model weights: {e}")
            raise