from sklearn.cluster import KMeans

class ClientNode:
    """Represents a federated client"""
    
    def __init__(self, client_id, local_loader, vit_model, config):
        self.client_id = client_id
        self.local_loader = local_loader
        self.model = copy.deepcopy(vit_model)
        self.config = config
        
        # Components
        self.prototype_manager = LocalPrototypeManager(
            K=config['K'],
            alpha_ema=config['alpha_ema'],
            novelty_threshold=config['novelty_threshold']
        )
        self.replay_buffer = ExperienceReplayBuffer(max_size=1000)
        
        # Optimizer (only for adapters)
        adapter_params = [p for name, p in self.model.named_parameters() 
                        if 'adapter' in name]
        self.optimizer = torch.optim.Adam(adapter_params, lr=0.001)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_round_1(self, global_model=None):
        """
        STEP 1: Round-1 Local Training
        
        Process:
        1. Train MAE reconstruction (if first phase)
        2. Extract embeddings
        3. Compute local prototypes
        4. NO prototype loss yet (global prototypes don't exist)
        
        Returns:
            local_prototypes: Client's prototype set
            prototype_counts: How many samples contributed to each
            model_weights: Updated adapter weights (to send to server)
        """
        
        all_embeddings = []
        
        # Phase 1: Forward pass on all local data to collect embeddings
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.local_loader):
                images = images.to(self.device)
                
                # Forward pass - get embeddings
                _, embeddings = self.model(images, return_features=True)
                all_embeddings.append(embeddings.cpu())
                
                # Initialize prototype manager on first batch
                if batch_idx == 0:
                    self.prototype_manager.initialize_from_first_batch(embeddings)
        
        # Phase 2: Compute local prototypes from all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 768]
        local_protos, proto_counts = self.prototype_manager.compute_local_prototypes_kmeans(
            all_embeddings, K=self.config['K']
        )
        
        # Phase 3: Train adapters with reconstruction loss (MAE-style)
        self.model.train()
        num_epochs = self.config['local_epochs']
        
        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(self.local_loader):
                images = images.to(self.device)
                
                # MAE reconstruction loss
                reconstructed = self.model.vit.forward_decoder(images)
                original = images.view(images.size(0), -1)
                L_recon = F.mse_loss(reconstructed, original)
                
                # No prototype loss in round 1
                loss = L_recon
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Extract adapter weights to send
        model_update = {
            name: param.clone() for name, param in self.model.named_parameters()
            if 'adapter' in name and param.requires_grad
        }
        
        return local_protos, proto_counts, model_update

def federated_round_1(clients, config):
    """
    Execute STEP 1 across all clients
    """
    all_local_protos = []
    all_proto_counts = []
    all_updates = []
    
    for client in clients:
        local_protos, proto_counts, model_update = client.train_round_1()
        all_local_protos.append(local_protos)
        all_proto_counts.append(proto_counts)
        all_updates.append(model_update)
    
    return all_local_protos, all_proto_counts, all_updates