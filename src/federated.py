
import torch
import torch.distributed as dist
import os

def setup_distributed(rank, world_size):
    """Initialize process group"""
    import tempfile
    import platform
    from pathlib import Path
    
    # Force GLOO to use loopback on Windows to avoid 'unsupported device' error
    if platform.system() == 'Windows':
        # Try to find loopback interface name if possible, or usually it's Loopback Pseudo-Interface 1
        # Or just let Gloo default but safer with file init
        pass 

    # Shared file for initialization (Robust on Windows/CI)
    init_file = Path(tempfile.gettempdir()) / "gloo_shared_init"
    # Ensure no old file if we are the first to run (though concurrency makes this racy)
    # But init_process_group handles it mostly.
    
    # URL format: file:///C:/path/to/file
    # On Windows: file:///C:/...
    init_method = f"file:///{init_file.absolute().as_posix()}"
    
    backend = "gloo" 
    # Use gloo for compatibility unless user strongly prefers nccl.
    # On Windows, gloo is safest for CPU/GPU hybrid logic.
    
    try:
        if rank == 0 and os.path.exists(init_file):
             try:
                 os.remove(init_file)
             except:
                 pass
        
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    except Exception as e:
        print(f"File init failed: {e}. Trying env:// with localhost...")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
        
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

def broadcast_global_prototypes(tensor, rank, device='cuda'):
    """
    Generic broadcast for prototypes or confidences.
    """
    # 1. Broadcast validity/size/shape
    if rank == 0:
        if tensor is None:
             shape_tensor = torch.tensor([0], device=device, dtype=torch.long)
        else:
             # Send dimensionality and shape
             # We assume < 0 means None/Empty?
             # Let's just send number of dimensions, then shape.
             dim_count = torch.tensor([tensor.dim()], device=device, dtype=torch.long)
             shape_tensor = torch.tensor(tensor.shape, device=device, dtype=torch.long)
    else:
        dim_count = torch.tensor([0], device=device, dtype=torch.long)
        # We don't know shape size yet
        
    # Broadcast dim count first
    dist.broadcast(dim_count, src=0)
    ndim = dim_count.item()
    
    if ndim == 0:
         return None # Or empty tensor? 
         
    if rank != 0:
         shape_tensor = torch.zeros(ndim, device=device, dtype=torch.long)
         
    dist.broadcast(shape_tensor, src=0)
    
    # 2. Broadcast data
    shape = list(shape_tensor.cpu().numpy())
    
    if rank != 0:
        tensor = torch.zeros(*shape, device=device)
        
    dist.broadcast(tensor, src=0)
    return tensor
