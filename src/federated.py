
import torch
import torch.distributed as dist
import os

def setup_distributed(rank, world_size):
    """Initialize process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize Process Group
    backend = "gloo" if os.name == 'nt' else "nccl"
    try:
        # Try NCCL if available (even on Windows with newer torch)
        if torch.cuda.is_available():
             dist.init_process_group("nccl", rank=rank, world_size=world_size)
        else:
             dist.init_process_group("gloo", rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Failed to init {backend} or nccl: {e}. Fallback to gloo.")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def average_models(model, world_size):
    """
    All-reduce aggregation for model parameters.
    """
    for param in model.parameters():
        if param.requires_grad:
             dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
             param.data /= world_size

def aggregate_prototypes(local_protos, local_counts, rank, world_size, embedding_dim=768, device='cuda'):
    """
    Gather all local prototypes at Rank 0.
    
    Args:
        local_protos (Tensor): [K, D]
        local_counts (Tensor): [K]
        
    Returns:
        gathered_protos_list: List of [K_i, D] tensors (Only on Rank 0)
        gathered_counts_list: List of [K_i] tensors (Only on Rank 0)
    """
    # 1. Exchange sizes
    k_local = torch.tensor([local_protos.size(0)], device=device, dtype=torch.long)
    
    # Gather sizes
    # We can use gather if we pad, but let's use explicit send/recv for variable sizes to be safe/simple
    
    gathered_protos = []
    gathered_counts = []
    
    if rank == 0:
        gathered_protos.append(local_protos)
        gathered_counts.append(local_counts)
        
        for src in range(1, world_size):
            # Recv size
            size_tensor = torch.tensor([0], device=device, dtype=torch.long)
            dist.recv(size_tensor, src=src)
            k = size_tensor.item()
            
            if k > 0:
                 # Recv protos
                 p_tensor = torch.zeros(k, embedding_dim, device=device)
                 dist.recv(p_tensor, src=src)
                 gathered_protos.append(p_tensor)
                 
                 # Recv counts
                 c_tensor = torch.zeros(k, device=device)
                 dist.recv(c_tensor, src=src)
                 gathered_counts.append(c_tensor)
            else:
                 # Handle empty
                 pass
            
    else:
        # Send size
        dist.send(k_local, dst=0)
        if k_local.item() > 0:
             # Send protos
             dist.send(local_protos, dst=0)
             # Send counts
             dist.send(local_counts, dst=0)
             
    return gathered_protos, gathered_counts

def broadcast_global_prototypes(global_protos, rank, embedding_dim=768, device='cuda'):
    """
    Broadcast [M, D] tensor from Rank 0 to all.
    """
    # 1. Broadcast validity/size
    if rank == 0:
        if global_protos is None:
             size = torch.tensor([0], device=device, dtype=torch.long)
        else:
             size = torch.tensor([global_protos.size(0)], device=device, dtype=torch.long)
    else:
        size = torch.tensor([0], device=device, dtype=torch.long)
        
    dist.broadcast(size, src=0)
    
    M = size.item()
    if M == 0:
         return torch.zeros(0, embedding_dim, device=device)

    # 2. Broadcast data
    if rank != 0:
        global_protos = torch.zeros(M, embedding_dim, device=device)
        
    dist.broadcast(global_protos, src=0)
    return global_protos
