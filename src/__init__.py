"""Expose the small set of reusable modules that power the whole pipeline.

The repository is organized so the top-level training script can read almost
like an experiment checklist:
1. ``src.mae_with_adapter`` builds the frozen MAE backbone plus trainable
   residual adapters.
2. ``src.loss`` defines GPAD, the prototype-anchored regularizer used only in
   the federated mode.
3. ``src.client`` owns client-side optimization, routing, and local-memory
   maintenance.
4. ``src.server`` owns server-side prototype merging and adapter aggregation.

Importing from ``src`` gives the training entrypoint one compact place to
collect those building blocks.
"""

from src.client import ClientManager, FederatedClient
from src.loss import GPADLoss
from src.mae_with_adapter import IBA_Adapter, ViTBlockWithAdapter, inject_adapters
from src.server import (
    FederatedModelServer,
    GlobalModel,
    GlobalPrototypeBank,
    run_server_round,
)

__all__ = [
    "ClientManager",
    "FederatedClient",
    "FederatedModelServer",
    "GlobalModel",
    "GlobalPrototypeBank",
    "GPADLoss",
    "IBA_Adapter",
    "ViTBlockWithAdapter",
    "inject_adapters",
    "run_server_round",
]
