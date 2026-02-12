
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from transformers import ViTMAEForPreTraining

from src.data import FederatedDataManager
from src.mae_with_adapter import inject_adapters, count_trainable_params
from src.prototypes import LocalPrototypeManager, ServerPrototypeManager
from src.loss import GPADLoss
from src.federated import setup_distributed, cleanup_distributed, average_models, aggregate_prototypes, broadcast_global_prototypes

def train_federated(rank, world_size, config):
    setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[Rank {rank}] Started on {device}")
    
    # Data
    dm = FederatedDataManager(num_clients=world_size, num_tasks=config['num_tasks'], seed=42)
    
    # Model
    model = ViTMAEForPreTraining.from_pretrained(config['model_name'])
    model = inject_adapters(model, bottleneck_dim=config['adapter_dim'])
    model.to(device)
    
    # Optimizer (Adapters only)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=config['lr'])
    
    # GPAD Loss
    gpad_criterion = GPADLoss(base_tau=0.5).to(device)
    
    # Prototype Managers
    local_pm = LocalPrototypeManager(device=device)
    
    # Server PM (Only used by Rank 0 effectively, but instantiated)
    server_pm = ServerPrototypeManager(device=device) if rank == 0 else None
    
    # Initial Globals
    global_prototypes = torch.zeros(0, 768, device=device)
    global_confidences = torch.zeros(0, device=device)
    
    for task_id in range(config['num_tasks']):
        if rank == 0:
             print(f"\n[System] Starting Task {task_id}")
        
        train_loader = dm.get_dataloader(client_id=rank, task_id=task_id, mode='train')
        
        for round_idx in range(config['rounds_per_task']):
            model.train()
            total_loss_avg = 0
            steps = 0
            
            # --- Local Training Round ---
            # "For every mini-batch... applies random masking..."
            for batch in train_loader:
                # 1. Input & Masking
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                
                optimizer.zero_grad()
                
                # 2. Forward Pass
                # ViTMAE forward returns: loss (recon), logits (recon), hidden_states
                
                outputs = model(images, output_hidden_states=True)
                recon_loss = outputs.loss
                
                # Extract Embeddings (CLS token of Encoder last layer)
                # outputs.hidden_states index -1 is final encoder layer
                final_encoder_layer = outputs.hidden_states[-1]
                embeddings = final_encoder_layer[:, 0, :] # [B, D]
                
                # 3. GPAD Loss
                gpad_loss = gpad_criterion(embeddings, global_prototypes, global_confidences)
                
                # 4. Total Loss
                total_loss = recon_loss + config['lambda_proto'] * gpad_loss
                total_loss.backward()
                optimizer.step()
                
                # 5. Online Local Prototype Update
                with torch.no_grad():
                    local_pm.update_batch(embeddings.detach())
                
                total_loss_avg += total_loss.item()
                steps += 1
                
            # --- End of Local Round ---
            
            # Communication
            
            # 1. Adapters (Avg)
            average_models(model, world_size)
            
            # 2. Prototypes
            # Get local set
            local_protos, local_counts = local_pm.get_prototypes()
            
            # Gather at Rank 0
            gathered_p, gathered_c = aggregate_prototypes(local_protos, local_counts, rank, world_size, device=device)
            
            # Server Merge
            if rank == 0:
                global_prototypes, global_confidences = server_pm.merge_batch(gathered_p, gathered_c)
                print(f"[Round {round_idx}] Task {task_id} - Avg Loss: {total_loss_avg/steps if steps>0 else 0:.4f} - Global Protos: {global_prototypes.size(0)}")
                
            # Broadcast back
            # We need to broadcast both Protos AND Confidences
            global_prototypes = broadcast_global_prototypes(global_prototypes, rank, device=device)
            global_confidences = broadcast_global_prototypes(global_confidences, rank, device=device)
            
    cleanup_distributed()

if __name__ == "__main__":
    config = {
        'num_tasks': 2,
        'num_clients': 2,
        'rounds_per_task': 2,
        'local_epochs': 1, 
        'lr': 1e-3,
        'lambda_proto': 1.0,
        'model_name': 'facebook/vit-mae-base',
        'adapter_dim': 64
    }
    
    # Use spawn for Windows compatibility with CUDA/PyTorch
    mp.spawn(train_federated, args=(config['num_clients'], config), nprocs=config['num_clients'], join=True)
