
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.server import ServerPrototypeManager, FederatedModelServer, run_server_round, GlobalModel
from src.client import ClientManager
from src.loss import GPADLoss

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simulation(num_clients=2, gpu_count=0): # Default to CPU heavily for testing logic, set gpu_count > 0 if available
    logger.info("="*60)
    logger.info("STARTING FEDERATED CONTINUAL LEARNING SIMULATION")
    logger.info("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Using {gpu_count} GPUs.")
    else:
        gpu_count = 0
        logger.info("Using CPU.")

    # 1. Initialize Global Model (Server Side)
    # We use a mocked model for speed if transformers not available, but let's try real one first
    # Or create a tiny dummy model that mimics ViTMAE interface
    
    class MockViTMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(32, 32)
            self.hidden_dim = 32
            
        def forward(self, x, output_hidden_states=False, **kwargs):
            # x: (B, L, D) -> here just (B, 32)
            # Simulate MAE output
            class Output:
                pass
            out = Output()
            out.loss = torch.tensor(0.5, requires_grad=True)
            out.hidden_states = [x.unsqueeze(1)] # (B, 1, 32)
            return out

    # Start with a simple linear model for fast test, or try loading real if possible
    # Given the environment, let's use the tiny mock to test logic flow quickly.
    # The important part is the dimensions and return types.
    base_model = MockViTMAE()
    
    # 2. Initialize Clients
    cm = ClientManager(base_model, num_clients, gpu_count)
    
    # Create Dummy Data
    # (B, 32)
    data = torch.randn(10, 32)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=2)
    dataloaders = [dataloader] * num_clients
    
    # 3. Initialize Server Components
    pm = ServerPrototypeManager(embedding_dim=32, device="cpu") # Prototypes on CPU for aggregation
    fms = FederatedModelServer()
    
    # -------------------------------------------------------------------------
    # ROUND 1: Initialization Phase (MAE Only)
    # -------------------------------------------------------------------------
    logger.info("\n>>> ROUND 1: MAE Only + Prototype Initialization <<<")
    
    # Train Clients (Parallel)
    # global_prototypes is None
    losses = cm.train_round(dataloaders, global_prototypes=None)
    logger.info(f"Round 1 Client Losses: {losses}")
    
    # Generate Prototypes & Weights
    client_payloads = []
    for i, client in enumerate(cm.clients):
        # Generate Prototypes (K-Means)
        protos = client.generate_prototypes(dataloader, K_init=5)
        # Get Weights (send deep copy to simulate network)
        weights = {k: v.cpu().clone() for k, v in client.model.state_dict().items()}
        
        payload = {
            'client_id': f"client_{i}",
            'protos': protos.cpu(),
            'weights': weights
        }
        client_payloads.append(payload)
        
    # Server Aggregation
    server_updates = run_server_round(
        proto_manager=pm,
        model_server=fms,
        client_payloads=client_payloads
    )
    
    global_protos = server_updates['global_prototypes']
    global_weights = server_updates['global_weights']
    
    logger.info(f"Global Prototypes aggregated: {global_protos.shape}") # Should be roughly (N*K_init, D) or merged
    
    # Broadcast (Update Client Models)
    # Ideally we load `global_weights` back into clients for sync, 
    # but for this test we focus on prototype propagation for Round 2.
    
    # -------------------------------------------------------------------------
    # ROUND 2: Continual Learning Phase (MAE + GPAD)
    # -------------------------------------------------------------------------
    logger.info("\n>>> ROUND 2: MAE + GPAD <<<")
    
    gpad_criterion = GPADLoss()
    
    # Train Clients (Parallel) with Global Prototypes
    # Note: global_prototypes might need movement to device inside train_epoch, 
    # which we handled in client.py
    losses_r2 = cm.train_round(
        dataloaders, 
        global_prototypes=global_protos, 
        gpad_loss_fn=gpad_criterion
    )
    logger.info(f"Round 2 Client Losses: {losses_r2}")
    
    logger.info("="*60)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    run_simulation()
