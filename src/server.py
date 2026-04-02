"""Implement the server-side synchronization step of the federated pipeline.

The server is responsible for the global state that is shared across clients.
After every communication round it performs two coordinated updates:

1. merge the client-local prototype payloads into the shared global prototype
   bank, and
2. aggregate the uploaded adapter weights into the next global adapter state.

Those merged prototypes and aggregated adapter weights are the only pieces of
state broadcast back to the clients for the next round.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GlobalPrototypeBank:
    """Store the shared prototype memory used by all federated clients.

    The bank follows a deterministic merge-or-add update rule:
    1. normalize every incoming local prototype,
    2. bootstrap directly from the first non-empty round if the bank is empty,
    3. compare each later prototype against the current bank,
    4. EMA-merge close matches, and
    5. append genuinely new prototypes while capacity remains.

    This class therefore acts as the compact global memory of the federated
    method.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        merge_threshold: float = 0.85,
        ema_alpha: float = 0.1,
        device: str = "cpu",
        max_prototypes: Optional[int] = 50,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.max_prototypes = max_prototypes
        self.prototypes = torch.zeros(
            0,
            embedding_dim,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def merge_local_prototypes(
        self,
        local_protos_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Merge one round of client prototypes into the shared global bank.

        The merge path is:
        1. Discard empty payloads.
        2. Normalize and concatenate the remaining local prototypes.
        3. Bootstrap directly if this is the first non-empty round.
        4. Compare each incoming prototype against the current bank.
        5. EMA-merge close matches.
        6. Append only prototypes that are still novel and still fit within the
           bank capacity.
        """
        valid_prototypes = [
            prototypes.to(device=self.device, dtype=self.dtype)
            for prototypes in local_protos_list
            if prototypes is not None and prototypes.numel() > 0
        ]
        if not valid_prototypes:
            return self.prototypes

        incoming = torch.cat(valid_prototypes, dim=0)
        incoming = F.normalize(incoming, p=2, dim=1)

        if self.prototypes.size(0) == 0:
            if self.max_prototypes is not None:
                incoming = incoming[: self.max_prototypes]
            self.prototypes = incoming
            return self.prototypes

        for prototype in incoming:
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
            similarities = torch.mv(self.prototypes, prototype)
            max_similarity, best_index = similarities.max(dim=0)

            if max_similarity >= self.merge_threshold:
                updated = (
                    (1.0 - self.ema_alpha) * self.prototypes[best_index]
                    + self.ema_alpha * prototype
                )
                self.prototypes[best_index] = F.normalize(updated, p=2, dim=0)
                continue

            if self.max_prototypes is not None and self.prototypes.size(0) >= self.max_prototypes:
                logger.info(
                    "Skipping a novel prototype because the global bank is at capacity (%s).",
                    self.max_prototypes,
                )
                continue

            self.prototypes = torch.cat(
                [self.prototypes, prototype.unsqueeze(0)],
                dim=0,
            )

        return self.prototypes

    def get_prototypes(self) -> torch.Tensor:
        """Return the current prototype matrix without changing server state."""
        return self.prototypes


class FederatedModelServer:
    """Aggregate the trainable adapter weights submitted by the clients.

    Only the trainable adapter subset is aggregated. That keeps communication
    small, leaves the frozen MAE backbone untouched, and makes the server state
    align exactly with the parameter-efficient training design.
    """

    @torch.no_grad()
    def aggregate_weights(
        self,
        client_weights_map: Dict[str, Dict[str, torch.Tensor]],
        current_global_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return one averaged adapter state dict for the next global round.

        The aggregation path is intentionally defensive:
        1. Build the union of parameter names across all client payloads.
        2. Gather the corresponding tensor from each client when it exists.
        3. Fall back to the current global tensor if a client omitted a key.
        4. Average everything in a common dtype on a common device.
        5. Return the aggregated trainable state for the next broadcast.
        """
        if not client_weights_map:
            logger.warning("No client weights were submitted for aggregation.")
            return {}

        all_keys = sorted(
            {
                parameter_name
                for client_state in client_weights_map.values()
                for parameter_name in client_state.keys()
            }
        )

        aggregated_state: Dict[str, torch.Tensor] = {}

        for key in all_keys:
            tensors: List[torch.Tensor] = []

            for client_id, client_state in client_weights_map.items():
                if key in client_state:
                    tensors.append(client_state[key])
                    continue

                if current_global_weights is not None and key in current_global_weights:
                    logger.warning(
                        "Client %s did not send parameter %s; using the current global value instead.",
                        client_id,
                        key,
                    )
                    tensors.append(current_global_weights[key])
                    continue

                logger.warning(
                    "Skipping missing parameter %s from client %s because no global fallback exists.",
                    key,
                    client_id,
                )

            if not tensors:
                continue

            reference_tensor = tensors[0]
            accumulated = torch.zeros_like(reference_tensor)
            for tensor in tensors:
                accumulated = accumulated + tensor.to(
                    device=reference_tensor.device,
                    dtype=reference_tensor.dtype,
                )

            aggregated_state[key] = accumulated / len(tensors)

        return aggregated_state


def run_server_round(
    proto_manager: GlobalPrototypeBank,
    model_server: FederatedModelServer,
    client_payloads: List[Dict[str, Any]],
    current_global_weights: Dict[str, torch.Tensor],
    round_idx: int = 1,
    server_model_ema_alpha: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run the full server update for one communication round.

    This helper keeps the entire server-side round logic in one place:
    1. merge the local prototype banks,
    2. average the uploaded client adapter weights,
    3. smooth the resulting adapter state with server-side EMA, and
    4. return both the updated global prototypes and the updated adapter state
       that should be broadcast next.
    """
    if not client_payloads:
        return torch.empty(0, 0, dtype=torch.float32), {}

    local_prototypes = [
        payload["protos"]
        for payload in client_payloads
        if "protos" in payload and payload["protos"] is not None
    ]
    global_prototypes = proto_manager.merge_local_prototypes(local_prototypes)

    client_weights_map: Dict[str, Dict[str, torch.Tensor]] = {}
    for payload_index, payload in enumerate(client_payloads):
        if "weights" not in payload:
            continue
        client_id = payload.get("client_id", f"unknown_client_{payload_index}")
        client_weights_map[client_id] = payload["weights"]

    aggregated_state = model_server.aggregate_weights(
        client_weights_map=client_weights_map,
        current_global_weights=current_global_weights,
    )

    if round_idx > 1 and current_global_weights:
        for key, new_tensor in aggregated_state.items():
            if key not in current_global_weights:
                continue
            old_tensor = current_global_weights[key].to(
                device=new_tensor.device,
                dtype=new_tensor.dtype,
            )
            aggregated_state[key] = (
                (1.0 - server_model_ema_alpha) * old_tensor
                + server_model_ema_alpha * new_tensor
            )

    return global_prototypes, aggregated_state


class GlobalModel:
    """Own a server-side MAE model instance for checkpoint-oriented workflows.

    The class is kept mostly for compatibility with older utilities. It loads
    the same adapter-injected MAE backbone used everywhere else in the repo.
    """

    def __init__(
        self,
        device: str = "cpu",
        pretrained_model_name: str = "facebook/vit-mae-base",
        adapter_bottleneck_dim: int = 64,
    ) -> None:
        self.device = torch.device(device)
        logger.info("Initializing the global model on %s.", self.device)

        try:
            from transformers import ViTMAEForPreTraining

            from src.mae_with_adapter import inject_adapters

            self.model = ViTMAEForPreTraining.from_pretrained(pretrained_model_name)
            self.model = inject_adapters(
                self.model,
                bottleneck_dim=adapter_bottleneck_dim,
            )
            self.model.to(self.device)
            self.model.eval()
        except ImportError as exc:
            logger.warning(
                "Transformers is unavailable (%s). Falling back to a stub linear model.",
                exc,
            )
            self.model = torch.nn.Linear(10, 10).to(self.device)

    def update_model_weights(self, aggregated_weights: Dict[str, torch.Tensor]) -> None:
        """Load the latest aggregated adapter weights into the stored model."""
        if not aggregated_weights:
            logger.warning("Received an empty aggregated state dict.")
            return

        load_result = self.model.load_state_dict(aggregated_weights, strict=False)
        logger.info(
            "Updated the global model | missing=%s | unexpected=%s",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )
