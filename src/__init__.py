"""
PODFCSSV — Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

Public API Exports
------------------
Loss:
    GPADLoss            — Gated Prototype Anchored Distillation loss

Server:
    GlobalPrototypeBank — EMA-based global prototype management
    FederatedModelServer — FedAvg weight aggregation
    GlobalModel         — Server-side model wrapper
    run_server_round    — Single-round server orchestrator

Client:
    FederatedClient     — Single edge-device simulation
    ClientManager       — Multi-GPU client orchestrator

Adapters:
    IBA_Adapter         — Information-Bottlenecked Adapter module
    inject_adapters     — Adapter injection into frozen ViT-MAE
"""

from src.loss import GPADLoss
from src.server import (
    GlobalPrototypeBank,
    FederatedModelServer,
    GlobalModel,
    run_server_round,
)
from src.client import FederatedClient, ClientManager
from src.mae_with_adapter import IBA_Adapter, inject_adapters

__all__ = [
    "GPADLoss",
    "GlobalPrototypeBank",
    "FederatedModelServer",
    "GlobalModel",
    "run_server_round",
    "FederatedClient",
    "ClientManager",
    "IBA_Adapter",
    "inject_adapters",
]
