"""Expose the reusable components behind the current experiment pipeline.

The active repository workflow is split deliberately between one orchestration
file and a small set of reusable modules:

1. ``main.py`` owns the executable experiment flow, dataset plan, stage-wise
   evaluation, plotting, and final checkpoint export.
2. ``src.mae_with_adapter`` builds the frozen ViT-MAE backbone and injects the
   trainable adapters used by both run modes.
3. ``src.loss`` defines GPAD, the prototype-aware regularizer used only in the
   federated path.
4. ``src.client`` contains the client-side continual state, including local
   training, local prototypes, and novelty buffers.
5. ``src.server`` contains the server-side global state, including prototype
   merging and adapter-weight aggregation.

Importing from ``src`` gives the rest of the codebase one stable place to fetch
the active algorithmic building blocks without duplicating implementation
details.
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
