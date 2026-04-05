"""Expose the reusable building blocks behind the active PODFCSSV pipeline.

The repository keeps the executable orchestration in ``main.py`` and places the
reusable algorithmic pieces under ``src``:
1. ``src.mae_with_adapter`` builds the frozen ViT-MAE backbone and injects the
   trainable adapters,
2. ``src.loss`` defines GPAD, the prototype-aware regularizer used only in the
   federated run,
3. ``src.client`` owns client-side continual state, local optimization, and
   local-memory updates, and
4. ``src.server`` owns the shared global prototype bank and adapter
   aggregation.

Importing from ``src`` keeps the rest of the codebase simple and avoids
duplicating the active training logic across files.
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
