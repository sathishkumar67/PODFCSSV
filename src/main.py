
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from transformers import ViTMAEForPreTraining

from src.data import FederatedDataManager
from src.mae_with_adapter import inject_adapters, count_trainable_params
from src.prototypes import PrototypeManager
from src.loss import TotalLoss
from src.federated import setup_distributed, cleanup_distributed, average_models, aggregate_prototypes, broadcast_global_prototypes

def train_federated(rank, world_size, config):
    setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[Rank {rank}] Started on {device}")
    
    # 1. Initialize Data
    # Each rank acts as a client.
    # We instantiate the DataManager. 
    # Note: In real simulation, data split creation should be deterministic across ranks or pre-computed.
    # Our `FederatedDataManager` uses `seed=42`, so `partition_data` will produce consistent splits
    # if called on every process.
    dm = FederatedDataManager(num_clients=world_size, num_tasks=config['num_tasks'], seed=42)
    
    # 2. Initialize Model (ViT-MAE + Adapters)
    # Load Pretrained MAE
    model = ViTMAEForPreTraining.from_pretrained(config['model_name'])
    model = inject_adapters(model, bottleneck_dim=config['adapter_dim'])
    model.to(device)
    
    # Optimizer (Only optimize adapters)
    # Filter parameters that require grad
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=config['lr'])
    
    # Loss
    criterion = TotalLoss(lambda_proto=config['lambda_proto']).to(device)
    
    # Prototype Manager
    pm = PrototypeManager(
        num_prototypes=config['num_local_prototypes'],
        merge_threshold=config['merge_threshold'],
        embedding_dim=768, # ViT-Base dim
        device=device
    )
    
    # Global Prototypes Cache
    global_prototypes = None
    
    # 3. Continual Learning Loop (Tasks)
    for task_id in range(config['num_tasks']):
        if rank == 0:
            print(f"\n{'='*40}")
            print(f"STARTING TASK {task_id}")
            print(f"{'='*40}")
            
        # Get Task Data
        train_loader = dm.get_dataloader(client_id=rank, task_id=task_id, mode='train')
        
        # Federated Rounds
        for round_idx in range(config['rounds_per_task']):
            model.train()
            
            embeddings_list = []
            epoch_loss = 0
            
            # Local Training (Epochs)
            for epoch in range(config['local_epochs']):
                batch_loss = 0
                for batch in train_loader:
                    # Input
                    if isinstance(batch, list) or isinstance(batch, tuple):
                        images, targets = batch
                    else:
                        images = batch
                        targets = None # MAE is self-supervised usually, but we might use labels for CE if hybrid
                        
                    images = images.to(device)
                    if targets is not None:
                        targets = targets.to(device)
                        
                    optimizer.zero_grad()
                    
                    # Forward
                    # We need access to embeddings for prototype loss.
                    # ViTMAEForPreTraining returns (loss, logits, hidden_states, attentions)
                    # We need to peek into the encoder output.
                    # Our `ViTBlockWithAdapter` facilitates this but standard `model()` call wraps it.
                    # `model.vit` is the backbone. 
                    
                    outputs = model.vit(images, output_hidden_states=True)
                    # Use CLS token or Mean Pooling of last hidden state
                    # sequence_output = outputs.last_hidden_state [Batch, Seq, D]
                    # MAE encoder output.
                    
                    # For prototype loss, we need feature embeddings.
                    # MAE uses [CLS] token usually for downstream.
                    # Let's use Mean Pooling or CLS.
                    features = outputs.last_hidden_state[:, 0, :] # CLS token
                    
                    # Store for prototype computation (only last epoch/step?)
                    if epoch == config['local_epochs'] - 1:
                         embeddings_list.append(features.detach())
                    
                    # Compute Loss
                    # If unlabeled, pure MAE loss?
                    # The user prompt uses CE Loss in example.
                    # The Guide says "Gated Distillation".
                    # If we use `ViTMAEForPreTraining`, it computes reconstruction loss internally if `labels` provided 
                    # (labels here means masked patches, not class labels).
                    # Wait, `ViTMAE` forward computes pixel reconstruction loss.
                    
                    # If we strictly follow "Federated *Self-Supervised*", we should use Reconstruction Loss.
                    # If we follow user's "fed_ddp_manual.py", it uses CE.
                    # I will assume we use CE for the Classification Task (Fine-tuning) + Prototype Distillation.
                    # BUT `images` here are (N, C, H, W).
                    
                    # To get logits for CE, we need a classifier head?
                    # `ViTMAEForPreTraining` has a Decoder, but for pixel reconstruction.
                    # If this is "Layer 2: Parameter Efficient Fine-Tuning", we usually attach a Classifier Head.
                    # I SHOULD ADD A CLASSIFIER HEAD.
                    # But the user didn't give me one. `ViTMAEForPreTraining` output is reconstruction.
                    
                    # I will assume `fed_ddp_manual` implies we want classification capability.
                    # I'll add a linear probe / classifier on top of features?
                    # Or maybe the "MAE" part implies we use the reconstruction loss + prototype loss on latent?
                    # Prototype Loss on latent is valid.
                    # CE Loss requires labels.
                    
                    # Let's implement Prototype Loss + MAE Reconstruction Loss (if supported) OR CE if we add head.
                    # For simplicity and "Self-Supervised" naming:
                    # Loss = Recon Loss (internal) + Lambda * Proto Loss.
                    # But the guide mentions "Accuracy" and "Forgetting".
                    # Accuracy requires a Classifier.
                    # I will simply add a dynamic classifier head or assume we measure accuracy via K-NN on prototypes?
                    # "Layer 5: Evaluation... Per-task accuracy".
                    
                    # Let's use `ViTMAEForPreTraining` loss (Recon) as base, and add Proto Loss.
                    # And for Evaluation, we can use K-NN (matches Prototypes!).
                    # So we don't need a trained classifier head if we do Proto-based classification. This fits the theme!
                    
                    # Loss = model(images).loss (Recon) + ProtoLoss(features, global_protos)
                    
                    mae_outputs = model(images) 
                    recon_loss = mae_outputs.loss
                    
                    # Proto Loss
                    proto_loss = criterion.proto_criterion(features, global_prototypes)
                    
                    total_loss = recon_loss + config['lambda_proto'] * proto_loss
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    batch_loss += total_loss.item()
                
                # End Epoch
            
            # End Local Rounds
            # 1. Compute Local Prototypes
            all_embeddings = torch.cat(embeddings_list, dim=0)
            local_protos, local_counts = pm.compute_local_prototypes(all_embeddings)
            
            # 2. Communication: Exchange Params
            average_models(model, world_size)
            
            # 3. Communication: Exchange Prototypes
            # Gather at Rank 0
            gathered_p, gathered_c = aggregate_prototypes(local_protos, local_counts, rank, world_size, device=device)
            
            # Rank 0 Merges
            global_protos_tensor = None
            if rank == 0:
                global_protos_tensor = pm.merge_global_prototypes(gathered_p, gathered_c)
            
            # Broadcast back
            global_prototypes = broadcast_global_prototypes(global_protos_tensor, rank, device=device)
            
            if rank == 0:
                print(f"[Round {round_idx}] Task {task_id} - Loss: {batch_loss:.4f} - Global Protos: {global_prototypes.size(0)}")

    cleanup_distributed()

if __name__ == "__main__":
    # Configuration
    config = {
        'num_tasks': 2, # Small for test
        'num_clients': 2,
        'rounds_per_task': 2,
        'local_epochs': 1,
        'lr': 1e-3,
        'lambda_proto': 1.0,
        'model_name': 'facebook/vit-mae-base',
        'adapter_dim': 64,
        'num_local_prototypes': 10,
        'merge_threshold': 0.85
    }
    
    # Run
    world_size = config['num_clients']
    # Use spawn
    mp.spawn(train_federated, args=(world_size, config), nprocs=world_size, join=True)
