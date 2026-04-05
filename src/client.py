"""Implement the full client-side training and local-memory logic.

This module contains the client behavior that makes the federated method
continual rather than stage-local. Between two server synchronizations each
client:
1. receives the latest shared adapter weights,
2. trains on its current dataset with MAE reconstruction loss,
3. applies GPAD only to embeddings that confidently match the shared prototype
   bank,
4. routes the remaining embeddings through its own persistent local memory,
5. clusters novel evidence when enough of it accumulates, and
6. returns the updated adapter payload and local prototypes to the server.

The key design choice is persistence: the optimizer state, local prototypes,
and novelty buffer are preserved when the dataset changes.
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _empty_routing_stats() -> Dict[str, int]:
    """Create one fresh routing-statistics container.

    The same keys are reused at the batch, epoch, and round levels so every
    training summary reports local matches, novel samples, clustering events,
    and prototype-growth events in a consistent format.
    """
    return {
        "local_matches": 0,
        "novel_samples": 0,
        "cluster_events": 0,
        "merged_prototypes": 0,
        "added_prototypes": 0,
    }


class FederatedClient:
    """Represent one federated participant and its persistent local state.

    Each client owns:
    1. its own adapter-injected MAE copy,
    2. its own optimizer state,
    3. a persistent local prototype bank, and
    4. a novelty buffer for embeddings that do not fit existing memory well
       enough yet.

    The class therefore handles both local optimization and local continual
    memory maintenance.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        local_update_threshold: float = 0.7,
        local_ema_alpha: float = 0.1,
        lambda_proto: float = 1.0,
        novelty_buffer_size: int = 500,
        novelty_k: int = 20,
        kmeans_max_iters: int = 100,
        kmeans_tol: float = 1e-4,
    ) -> None:
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

        self.model = copy.deepcopy(model).to(self.device)
        self.local_prototypes: Optional[torch.Tensor] = None
        self.novelty_buffer: List[torch.Tensor] = []

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        optimizer_settings = optimizer_kwargs or {"lr": 1e-4}
        self.optimizer = optim.AdamW(trainable_parameters, **optimizer_settings)

        logger.info(
            "Client %s ready on %s | threshold=%.3f | ema=%.3f | buffer=%s | novelty_k=%s",
            self.client_id,
            self.device,
            self.local_update_threshold,
            self.local_ema_alpha,
            self.novelty_buffer_size,
            self.novelty_k,
        )

    def sync_trainable_weights(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Load the latest server adapter weights into the local model copy.

        Only the trainable adapter parameters are exchanged, so ``strict=False``
        is the correct behavior and keeps the frozen MAE backbone untouched.
        """
        if not global_weights:
            return
        self.model.load_state_dict(global_weights, strict=False)

    def get_trainable_state(self) -> Dict[str, torch.Tensor]:
        """Return the exact trainable upload payload for one client round.

        Only the trainable adapter parameters are copied out, which keeps the
        communication budget focused on the parameter-efficient part of the
        model instead of the full MAE backbone.
        """
        trainable_state: Dict[str, torch.Tensor] = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                trainable_state[name] = parameter.detach().cpu().clone()
        return trainable_state

    def _pool_encoder_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert the last encoder tokens into one embedding per image.

        The pooling rule is shared everywhere in the repository:
        1. Drop the CLS token when it exists.
        2. Average only the patch tokens.
        3. Use that pooled vector for routing, clustering, and later probing.
        """
        if hidden_states.dim() != 3:
            raise ValueError("Expected hidden states with shape [batch, tokens, dim].")

        if hidden_states.size(1) > 1:
            patch_tokens = hidden_states[:, 1:, :]
        else:
            patch_tokens = hidden_states
        return patch_tokens.mean(dim=1)

    def _forward_embeddings(
        self,
        inputs: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        """Run one MAE forward pass and return both outputs and pooled embeddings.

        GPAD, prototype routing, and reconstruction loss all rely on the same
        encoder pass, so this helper exposes both the raw model outputs and the
        pooled encoder features from one call.
        """
        outputs = self.model(inputs, output_hidden_states=True)
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError(
                "The MAE model did not return hidden states. "
                "GPAD requires output_hidden_states=True support."
            )

        final_hidden_states = outputs.hidden_states[-1]
        embeddings = self._pool_encoder_tokens(final_hidden_states)
        return outputs, embeddings

    def train_epoch(
        self,
        dataloader: DataLoader,
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Train one full local epoch and return flat logging metrics.

        Each batch follows the same path:
        1. run one MAE forward pass and get both reconstruction loss and pooled
           embeddings,
        2. determine which samples are globally anchored,
        3. apply GPAD only to the anchored subset,
        4. route the remaining embeddings through the local-memory logic,
        5. step the optimizer on the combined objective, and
        6. accumulate JSON-friendly metrics for later analysis and plotting.
        """
        self.model.train()

        total_loss = 0.0
        total_mae_loss = 0.0
        total_gpad_loss = 0.0
        total_samples = 0
        total_batches = 0
        total_anchored = 0
        routing_stats = _empty_routing_stats()

        has_gpad = global_prototypes is not None and gpad_loss_fn is not None
        prototype_bank = None
        if has_gpad:
            prototype_bank = global_prototypes.detach().to(
                device=self.device,
                dtype=self.dtype,
            )

        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device=self.device, dtype=self.dtype)

            outputs, embeddings = self._forward_embeddings(inputs)
            mae_loss = getattr(outputs, "loss", None)
            if mae_loss is None:
                mae_loss = embeddings.sum() * 0.0

            final_loss = mae_loss
            batch_anchor_mask = None
            batch_gpad_loss_value = 0.0

            if has_gpad and prototype_bank is not None:
                detached_embeddings = embeddings.detach()
                batch_anchor_mask = gpad_loss_fn.compute_anchor_mask(
                    detached_embeddings,
                    prototype_bank,
                )

                anchored_embeddings = embeddings[batch_anchor_mask]
                if anchored_embeddings.size(0) > 0:
                    gpad_loss = gpad_loss_fn(anchored_embeddings, prototype_bank)
                    final_loss = final_loss + self.lambda_proto * gpad_loss
                    batch_gpad_loss_value = float(gpad_loss.detach().item())

                non_anchored_embeddings = detached_embeddings[~batch_anchor_mask]
                batch_routing = self._route_non_anchored(non_anchored_embeddings)
                for key, value in batch_routing.items():
                    routing_stats[key] += value
            else:
                batch_anchor_mask = torch.zeros(
                    embeddings.size(0),
                    dtype=torch.bool,
                    device=embeddings.device,
                )

            self.optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)
            total_loss += float(final_loss.detach().item())
            total_mae_loss += float(mae_loss.detach().item())
            total_gpad_loss += batch_gpad_loss_value
            total_samples += batch_size
            total_batches += 1
            total_anchored += int(batch_anchor_mask.sum().item())

        average_loss = total_loss / total_batches if total_batches > 0 else 0.0
        average_mae_loss = total_mae_loss / total_batches if total_batches > 0 else 0.0
        average_gpad_loss = (
            total_gpad_loss / total_batches if total_batches > 0 else 0.0
        )

        prototype_count = (
            int(self.local_prototypes.size(0))
            if self.local_prototypes is not None
            else 0
        )

        return {
            "loss": average_loss,
            "mae_loss": average_mae_loss,
            "gpad_loss": average_gpad_loss,
            "num_batches": float(total_batches),
            "num_samples": float(total_samples),
            "anchored_fraction": (
                total_anchored / total_samples if total_samples > 0 else 0.0
            ),
            "local_match_fraction": (
                routing_stats["local_matches"] / total_samples if total_samples > 0 else 0.0
            ),
            "novel_fraction": (
                routing_stats["novel_samples"] / total_samples if total_samples > 0 else 0.0
            ),
            "buffer_cluster_events": float(routing_stats["cluster_events"]),
            "merged_prototypes": float(routing_stats["merged_prototypes"]),
            "added_prototypes": float(routing_stats["added_prototypes"]),
            "prototype_count": float(prototype_count),
            "novelty_buffer_size": float(len(self.novelty_buffer)),
        }

    @torch.no_grad()
    def _route_non_anchored(self, embeddings: torch.Tensor) -> Dict[str, int]:
        """Route embeddings that were not anchored to the global bank.

        This method handles the local-memory side of the federated algorithm:
        1. normalize the incoming embeddings,
        2. if no local bank exists yet, send everything to the novelty buffer,
        3. otherwise compare each embedding with the local prototype bank,
        4. EMA-update the best local prototype when the match is strong enough,
           and
        5. send the rest into the novelty buffer for later clustering.

        The method therefore decides whether a sample should reinforce existing
        local knowledge or become evidence for a new local concept.
        """
        stats = _empty_routing_stats()
        if embeddings.size(0) == 0:
            return stats

        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        if self.local_prototypes is None or self.local_prototypes.size(0) == 0:
            for embedding in normalized_embeddings:
                self.novelty_buffer.append(embedding.detach().cpu().clone())
                stats["novel_samples"] += 1

            cluster_stats = self._maybe_cluster_buffer()
            for key, value in cluster_stats.items():
                stats[key] += value
            return stats

        working_prototypes = F.normalize(
            self.local_prototypes.to(self.device),
            p=2,
            dim=1,
        )

        for embedding in normalized_embeddings:
            similarities = torch.mv(working_prototypes, embedding)
            max_similarity, best_index = similarities.max(dim=0)

            if max_similarity >= self.local_update_threshold:
                updated_prototype = (
                    (1.0 - self.local_ema_alpha) * working_prototypes[best_index]
                    + self.local_ema_alpha * embedding
                )
                working_prototypes[best_index] = F.normalize(
                    updated_prototype,
                    p=2,
                    dim=0,
                )
                stats["local_matches"] += 1
            else:
                self.novelty_buffer.append(embedding.detach().cpu().clone())
                stats["novel_samples"] += 1

        self.local_prototypes = working_prototypes

        cluster_stats = self._maybe_cluster_buffer()
        for key, value in cluster_stats.items():
            stats[key] += value
        return stats

    def _maybe_cluster_buffer(self) -> Dict[str, int]:
        """Trigger novelty-buffer clustering only when enough novel evidence exists."""
        if len(self.novelty_buffer) < self.novelty_buffer_size:
            return _empty_routing_stats()
        return self._cluster_novelty_buffer()

    @torch.no_grad()
    def _cluster_novelty_buffer(self) -> Dict[str, int]:
        """Cluster the novelty buffer and merge or append the discovered centroids.

        This is the local memory-growth step:
        1. stack and normalize the buffered embeddings,
        2. run spherical K-means to form candidate centroids,
        3. merge centroids into similar local prototypes with EMA,
        4. append truly new centroids as new local prototypes, and
        5. clear the buffer once the update completes.
        """
        stats = _empty_routing_stats()
        if len(self.novelty_buffer) == 0:
            return stats

        stats["cluster_events"] = 1

        buffer_tensor = torch.stack(self.novelty_buffer, dim=0).to(self.device)
        buffer_tensor = F.normalize(buffer_tensor, p=2, dim=1)
        num_clusters = min(self.novelty_k, buffer_tensor.size(0))
        new_centroids = self._kmeans(buffer_tensor, num_clusters)

        if self.local_prototypes is None or self.local_prototypes.size(0) == 0:
            self.local_prototypes = new_centroids
            stats["added_prototypes"] = int(new_centroids.size(0))
            self.novelty_buffer.clear()
            return stats

        working_prototypes = F.normalize(self.local_prototypes.to(self.device), p=2, dim=1)
        centroids = F.normalize(new_centroids, p=2, dim=1)
        centroids_to_add: List[torch.Tensor] = []

        for centroid in centroids:
            similarities = torch.mv(working_prototypes, centroid)
            max_similarity, best_index = similarities.max(dim=0)

            if max_similarity >= self.local_update_threshold:
                updated_prototype = (
                    (1.0 - self.local_ema_alpha) * working_prototypes[best_index]
                    + self.local_ema_alpha * centroid
                )
                working_prototypes[best_index] = F.normalize(
                    updated_prototype,
                    p=2,
                    dim=0,
                )
                stats["merged_prototypes"] += 1
            else:
                centroids_to_add.append(centroid)
                stats["added_prototypes"] += 1

        if centroids_to_add:
            stacked_new_centroids = torch.stack(centroids_to_add, dim=0)
            working_prototypes = torch.cat(
                [working_prototypes, stacked_new_centroids],
                dim=0,
            )

        self.local_prototypes = working_prototypes
        self.novelty_buffer.clear()
        return stats

    def get_local_prototypes(self) -> Optional[torch.Tensor]:
        """Return a detached copy of the current local prototype bank."""
        if self.local_prototypes is None:
            return None
        return self.local_prototypes.detach().clone()

    @torch.no_grad()
    def generate_prototypes(
        self,
        dataloader: DataLoader,
        K_init: int = 10,
    ) -> torch.Tensor:
        """Extract current-dataset centroids and merge them into local memory.

        The first round of each new stage uses this helper so the client's
        stored prototype memory is enriched with fresh concepts from the new
        dataset instead of being replaced. The update path is:
        1. extract embeddings for the full current dataset,
        2. normalize them on the unit sphere,
        3. run spherical K-means to obtain stage centroids, and
        4. merge or append those centroids into the persistent local bank.
        """
        self.model.eval()
        feature_batches: List[torch.Tensor] = []

        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device=self.device, dtype=self.dtype)
            _, embeddings = self._forward_embeddings(inputs)
            feature_batches.append(embeddings.detach())

        if not feature_batches:
            if self.local_prototypes is None:
                self.local_prototypes = torch.empty(
                    0,
                    0,
                    device=self.device,
                    dtype=self.dtype,
                )
            return self.local_prototypes

        embeddings = torch.cat(feature_batches, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centroids = self._kmeans(embeddings, K=K_init)

        if self.local_prototypes is None or self.local_prototypes.size(0) == 0:
            self.local_prototypes = centroids.detach().clone()
            return self.local_prototypes

        working_prototypes = F.normalize(self.local_prototypes.to(self.device), p=2, dim=1)
        centroids = F.normalize(centroids, p=2, dim=1)

        for centroid in centroids:
            similarities = torch.mv(working_prototypes, centroid)
            max_similarity, best_index = similarities.max(dim=0)

            if max_similarity >= self.local_update_threshold:
                updated_prototype = (
                    (1.0 - self.local_ema_alpha) * working_prototypes[best_index]
                    + self.local_ema_alpha * centroid
                )
                working_prototypes[best_index] = F.normalize(
                    updated_prototype,
                    p=2,
                    dim=0,
                )
            else:
                working_prototypes = torch.cat(
                    [working_prototypes, centroid.unsqueeze(0)],
                    dim=0,
                )

        self.local_prototypes = F.normalize(working_prototypes, p=2, dim=1).detach().clone()
        return self.local_prototypes

    def _kmeans(self, features: torch.Tensor, K: int) -> torch.Tensor:
        """Run spherical K-means on unit-normalized feature vectors.

        The implementation repeatedly assigns each feature to its closest
        centroid, recomputes centroids from the assignments, renormalizes them,
        and stops once the centroid movement falls below the configured
        tolerance or the iteration limit is reached.
        """
        num_samples, feature_dim = features.shape
        if num_samples == 0:
            return torch.empty(0, feature_dim, device=features.device, dtype=features.dtype)

        K = min(K, num_samples)
        initial_indices = torch.randperm(num_samples, device=features.device)[:K]
        centroids = features[initial_indices].clone()

        for _ in range(self.kmeans_max_iters):
            centroids = F.normalize(centroids, p=2, dim=1)
            similarities = torch.mm(features, centroids.t())
            labels = similarities.argmax(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for cluster_index in range(K):
                mask = labels == cluster_index
                if mask.any():
                    new_centroids[cluster_index] = features[mask].mean(dim=0)
                else:
                    random_index = torch.randint(
                        low=0,
                        high=num_samples,
                        size=(1,),
                        device=features.device,
                    ).item()
                    new_centroids[cluster_index] = features[random_index]

            new_centroids = F.normalize(new_centroids, p=2, dim=1)
            center_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids

            if center_shift < self.kmeans_tol:
                break

        return F.normalize(centroids, p=2, dim=1)


class ClientManager:
    """Own the client collection and orchestrate one federated round at a time.

    The main training loop delegates round-level coordination here so
    ``main.py`` can stay focused on the stage plan and reporting flow. The
    manager handles:
    1. client creation,
    2. adapter-weight synchronization,
    3. local-epoch repetition,
    4. per-round parallelism when the GPU topology matches the client count,
       and
    5. sequential fallback on CPU.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_clients: int,
        gpu_count: int = 0,
        dtype: torch.dtype = torch.float32,
        local_epochs: int = 1,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        local_update_threshold: float = 0.7,
        local_ema_alpha: float = 0.1,
        lambda_proto: float = 1.0,
        novelty_buffer_size: int = 500,
        novelty_k: int = 20,
        kmeans_max_iters: int = 100,
        kmeans_tol: float = 1e-4,
    ) -> None:
        self.num_clients = num_clients
        self.gpu_count = gpu_count
        self.dtype = dtype
        self.local_epochs = local_epochs
        self.optimizer_kwargs = optimizer_kwargs
        self.local_update_threshold = local_update_threshold
        self.local_ema_alpha = local_ema_alpha
        self.lambda_proto = lambda_proto
        self.novelty_buffer_size = novelty_buffer_size
        self.novelty_k = novelty_k
        self.kmeans_max_iters = kmeans_max_iters
        self.kmeans_tol = kmeans_tol
        self.clients: List[FederatedClient] = []

        self._initialize_clients(base_model)

    def _train_client_for_round(
        self,
        client: FederatedClient,
        dataloader: DataLoader,
        global_prototypes: Optional[torch.Tensor],
        gpad_loss_fn: Optional[nn.Module],
    ) -> Dict[str, float]:
        """Train one client for the configured local-epoch budget.

        The helper keeps the round interface simple:
        1. Call ``train_epoch`` the requested number of times.
        2. Average metrics that represent rates or losses.
        3. Sum counters that represent events or sample totals.
        4. Keep last-value metrics for state-like quantities.
        """
        epoch_results = [
            client.train_epoch(
                dataloader,
                global_prototypes=global_prototypes,
                gpad_loss_fn=gpad_loss_fn,
            )
            for _ in range(self.local_epochs)
        ]

        if not epoch_results:
            return {}

        average_keys = {
            "loss",
            "mae_loss",
            "gpad_loss",
            "anchored_fraction",
            "local_match_fraction",
            "novel_fraction",
        }
        sum_keys = {
            "num_batches",
            "num_samples",
            "buffer_cluster_events",
            "merged_prototypes",
            "added_prototypes",
        }
        last_value_keys = {
            "prototype_count",
            "novelty_buffer_size",
        }

        aggregated_result: Dict[str, float] = {}
        metric_keys = epoch_results[0].keys()
        for key in metric_keys:
            if key in average_keys:
                aggregated_result[key] = float(
                    sum(epoch_result.get(key, 0.0) for epoch_result in epoch_results)
                    / len(epoch_results)
                )
            elif key in sum_keys:
                aggregated_result[key] = float(
                    sum(epoch_result.get(key, 0.0) for epoch_result in epoch_results)
                )
            elif key in last_value_keys:
                aggregated_result[key] = float(epoch_results[-1].get(key, 0.0))
            else:
                aggregated_result[key] = float(epoch_results[-1].get(key, 0.0))

        return aggregated_result

    def _initialize_clients(self, base_model: nn.Module) -> None:
        """Create each client object and assign one explicit device per client.

        The current publishable setup expects either CPU execution or one GPU
        per client so the schedule remains easy to interpret and reproduce.
        """
        if self.gpu_count > 0 and self.num_clients != self.gpu_count:
            raise ValueError(
                "When GPUs are used, the code expects one GPU per client "
                "to keep the scheduling explicit and reproducible."
            )

        for client_index in range(self.num_clients):
            if self.gpu_count > 0:
                device = torch.device(f"cuda:{client_index}")
            else:
                device = torch.device("cpu")

            client = FederatedClient(
                client_id=client_index,
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

    def sync_clients(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Broadcast the latest global adapter weights to every client copy."""
        for client in self.clients:
            client.sync_trainable_weights(global_weights)

    def train_round(
        self,
        dataloaders: List[DataLoader],
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> List[Dict[str, float]]:
        """Run one full communication round across all clients.

        The manager uses the same round structure in every stage:
        1. Verify that one dataloader exists for each client.
        2. Launch local training on each client.
        3. Use parallel execution only when the GPU topology matches the client
           count.
        4. Fall back to sequential CPU execution when needed.
        """
        if len(dataloaders) != self.num_clients:
            raise ValueError(
                f"Received {len(dataloaders)} dataloaders for {self.num_clients} clients."
            )

        round_results: List[Dict[str, float]] = [{} for _ in range(self.num_clients)]

        if self.gpu_count > 0:
            with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
                future_to_index = {
                    executor.submit(
                        self._train_client_for_round,
                        client,
                        dataloaders[index],
                        global_prototypes=global_prototypes,
                        gpad_loss_fn=gpad_loss_fn,
                    ): index
                    for index, client in enumerate(self.clients)
                }

                for future, index in future_to_index.items():
                    round_results[index] = future.result()
        else:
            for index, client in enumerate(self.clients):
                round_results[index] = self._train_client_for_round(
                    client,
                    dataloaders[index],
                    global_prototypes=global_prototypes,
                    gpad_loss_fn=gpad_loss_fn,
                )

        return round_results
