def forward_with_prototype_assignment(client, batch_images, global_prototypes):
    """
    STEP 5.1-5.2: Forward Pass & Global Prototype Assignment
    
    For each embedding:
    1. Compute similarity to all global prototypes
    2. Assign to best match (or use soft assignment)
    """
    
    # Forward pass through adapted model
    logits, embeddings = client.model(batch_images, return_features=True)
    # embeddings: [B, 768]
    
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, dim=1)
    global_protos_norm = F.normalize(global_prototypes, dim=1)
    
    # Compute similarity to all global prototypes
    similarity_scores = torch.mm(embeddings_norm, global_protos_norm.T)
    # [B, K_global]
    
    # Option A: Hard assignment (argmax)
    best_proto_idx = similarity_scores.argmax(dim=1)  # [B]
    best_similarity = similarity_scores.gather(1, best_proto_idx.unsqueeze(1)).squeeze()  # [B]
    
    # Option B: Soft assignment (recommended - prevents mode collapse)
    temperature = 0.07  # Temperature for softmax
    soft_weights = F.softmax(similarity_scores / temperature, dim=1)  # [B, K_global]
    
    return {
        'embeddings': embeddings,
        'logits': logits,
        'similarity_scores': similarity_scores,
        'best_proto_idx': best_proto_idx,
        'best_similarity': best_similarity,
        'soft_weights': soft_weights,
    }
    
def client_send_updates(client, round_id, task_id):
    """
    STEP 7: Client sends updates to server
    
    What to send:
    - Updated local prototypes
    - Prototype counts (sample contribution)
    - Prototype confidence scores
    - Prototype task tags
    - Adapter weights (updated)
    - Novelty flags (which prototypes are new)
    
    What NOT to send:
    - Full embeddings
    - Raw data
    - Encoder weights (frozen)
    
    Total size: ~510 KB per client per round
    """
    
    client_message = {
        'client_id': client.client_id,
        'round_id': round_id,
        'task_id': task_id,
        
        # Prototypes
        'local_prototypes': client.prototype_manager.prototypes,  # [K_local, 768]
        'prototype_counts': client.prototype_manager.prototype_counts,  # [K_local]
        'prototype_confidence': compute_client_confidence(client),  # [K_local]
        'prototype_task_id': torch.full((len(client.prototype_manager.prototypes),), 
                                    task_id, dtype=torch.long),
        
        # Model updates
        'adapter_params': {
            name: param.cpu().clone() for name, param in 
            client.model.named_parameters() if 'adapter' in name
        },
        
        # Metadata
        'data_size': len(client.replay_buffer),
        'novelty_count': len(client.prototype_manager.novelty_buffer),
    }
    
    return client_message


def compute_client_confidence(client):
    """Confidence score for each local prototype"""
    # Count-based + variance-based confidence
    total_count = client.prototype_manager.prototype_counts.sum()
    count_conf = client.prototype_manager.prototype_counts / total_count
    
    # Variance-based (placeholder - would need actual variance)
    variance_conf = torch.ones_like(count_conf)
    
    return count_conf * variance_conf


def server_update_prototypes(server, client_messages, round_id, task_id):
    """
    STEP 8: Server Update Global Prototypes
    
    Process:
    1. Receive local prototypes from all clients
    2. Match each local → best global prototype
    3. Merge if match is good, else create new
    4. Update confidence scores and metadata
    5. Optionally consolidate to prevent explosion
    """
    
    print(f"\n[Server] Round {round_id} Task {task_id} - Updating global prototypes")
    
    # Collect all client prototypes
    all_client_protos = []
    all_client_counts = []
    all_client_conf = []
    
    for msg in client_messages:
        all_client_protos.append(msg['local_prototypes'])
        all_client_counts.append(msg['prototype_counts'])
        all_client_conf.append(msg['prototype_confidence'])
    
    # ====================
    # MATCHING STRATEGY
    # ====================
    
    # OPTION A: Simple Cosine Matching (Fast, recommended for K < 100)
    
    similarity_threshold = 0.80
    
    for client_idx, client_protos in enumerate(all_client_protos):
        for local_idx, local_proto in enumerate(client_protos):
            local_norm = F.normalize(local_proto.unsqueeze(0), dim=1)
            global_norm = F.normalize(server.global_prototypes, dim=1)
            
            # Cosine similarity
            sims = torch.mm(local_norm, global_norm.T)[0]  # [K_global]
            best_match_idx = sims.argmax()
            best_sim = sims[best_match_idx]
            
            if best_sim > similarity_threshold:
                # Merge: weighted average
                count_local = all_client_counts[client_idx][local_idx]
                count_global = server.prototype_counts[best_match_idx]
                conf_local = all_client_conf[client_idx][local_idx]
                conf_global = server.prototype_confidence[best_match_idx]
                
                # Weight by confidence and count
                total_weight = count_local * conf_local + count_global * conf_global
                merged = (count_local * conf_local * local_proto +
                         count_global * conf_global * server.global_prototypes[best_match_idx]
                         ) / (total_weight + 1e-8)
                merged = F.normalize(merged, dim=0)
                
                server.global_prototypes[best_match_idx] = merged
                server.prototype_counts[best_match_idx] = (count_local + count_global) / 2
                server.prototype_last_used[best_match_idx] = round_id
                
            else:
                # Create new global prototype
                server.global_prototypes = torch.cat([
                    server.global_prototypes,
                    local_proto.unsqueeze(0)
                ])
                server.prototype_counts = torch.cat([
                    server.prototype_counts,
                    torch.tensor([all_client_counts[client_idx][local_idx]])
                ])
                server.prototype_confidence = torch.cat([
                    server.prototype_confidence,
                    all_client_conf[client_idx][local_idx].unsqueeze(0)
                ])
                server.prototype_created_round = torch.cat([
                    server.prototype_created_round,
                    torch.tensor([round_id])
                ])
                server.prototype_last_used = torch.cat([
                    server.prototype_last_used,
                    torch.tensor([round_id])
                ])
                server.task_per_prototype = torch.cat([
                    server.task_per_prototype,
                    torch.tensor([task_id])
                ])
    
    # ====================
    # CONSOLIDATION
    # ====================
    
    # ENHANCEMENT 2: Periodically consolidate to prevent explosion
    if round_id % 10 == 0:
        server.consolidate_prototypes()
    
    # ====================
    # UPDATE MODEL WEIGHTS
    # ====================
    
    # Aggregate adapter weights from all clients (FedAvg)
    aggregated_adapters = {}
    for param_name in client_messages[0]['adapter_params'].keys():
        param_values = [msg['adapter_params'][param_name] 
                       for msg in client_messages]
        aggregated = torch.stack(param_values).mean(dim=0)
        aggregated_adapters[param_name] = aggregated
    
    # Load into server model
    server.global_model.load_state_dict(aggregated_adapters, strict=False)
    
    print(f"[Server] Global prototypes: {len(server.global_prototypes)} "
          f"(added {len(all_client_protos)} new this round)")
    
    return server.global_prototypes, aggregated_adapters

def consolidate_prototypes(server):
    """
    ENHANCEMENT 2: Remove dead/redundant prototypes
    """
    # Remove unused prototypes
    min_usage_ratio = 0.01
    total_usage = server.prototype_counts.sum()
    
    active_mask = server.prototype_counts > (min_usage_ratio * total_usage)
    
    # Merge very similar ones
    sims = torch.mm(server.global_prototypes, server.global_prototypes.T)
    merge_threshold = 0.95
    
    for i in range(len(server.global_prototypes)):
        for j in range(i+1, len(server.global_prototypes)):
            if active_mask[i] and active_mask[j]:
                if sims[i, j] > merge_threshold:
                    # Merge j into i
                    total_count = server.prototype_counts[i] + server.prototype_counts[j]
                    merged = (server.prototype_counts[i] * server.global_prototypes[i] +
                             server.prototype_counts[j] * server.global_prototypes[j]
                             ) / total_count
                    server.global_prototypes[i] = F.normalize(merged, dim=0)
                    server.prototype_counts[i] = total_count
                    active_mask[j] = False
    
    # Keep only active
    if active_mask.sum() < len(server.global_prototypes):
        server.global_prototypes = server.global_prototypes[active_mask]
        server.prototype_counts = server.prototype_counts[active_mask]
        server.prototype_confidence = server.prototype_confidence[active_mask]
        server.prototype_created_round = server.prototype_created_round[active_mask]
        server.prototype_last_used = server.prototype_last_used[active_mask]
        server.task_per_prototype = server.task_per_prototype[active_mask]
        
        print(f"[Server] Consolidated prototypes: "
              f"{len(server.global_prototypes)} → {active_mask.sum()}")