
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDistillationLoss(nn.Module):
    def __init__(self, temperature=0.07, tau_base=0.5, lambda_entropy=0.1):
        super().__init__()
        self.temperature = temperature
        self.tau_base = tau_base
        self.lambda_entropy = lambda_entropy

    def forward(self, embeddings, prototypes):
        """
        Args:
            embeddings: [Batch, D]
            prototypes: [M, D] global prototypes
            
        Returns:
            loss: scalar
        """
        if prototypes is None or prototypes.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        # 1. Normalize
        z_norm = F.normalize(embeddings, p=2, dim=1)
        p_norm = F.normalize(prototypes, p=2, dim=1)
        
        # 2. Similarity [Batch, M]
        sims = torch.mm(z_norm, p_norm.t())
        
        # 3. Best matches
        max_sim, best_idx = sims.max(dim=1)
        
        # 4. Entropy calculation for adaptive gating
        # P(z belongs to k) ~ exp(sim)
        probs = F.softmax(sims / self.temperature, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # 5. Adaptive Threshold
        tau_adaptive = self.tau_base - self.lambda_entropy * entropy
        
        # 6. Gate
        # gate = sigmoid((sim - tau) / T)
        gate_logits = (max_sim - tau_adaptive) / self.temperature
        gate = torch.sigmoid(gate_logits)
        
        # 7. Prototype Loss (MSE/Cosine distance on normalized)
        # ||z - p||^2 = 2 - 2*cos(z,p)
        # We minimize dist => maximize cos => minimize -cos
        # loss = gate * (2 - 2*max_sim)  or simply gate * (1 - max_sim) for direction
        
        # Using Euclidean distance squared on normalized vectors:
        # dist_sq = ||z - p||^2 = 2(1 - cos)
        
        dist_sq = 2 * (1 - max_sim)
        loss = (gate * dist_sq).mean()
        
        return loss

class TotalLoss(nn.Module):
    def __init__(self, base_criterion=nn.CrossEntropyLoss(), lambda_proto=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.proto_criterion = GatedDistillationLoss()
        self.lambda_proto = lambda_proto
        
    def forward(self, outputs, targets, embeddings, prototypes, round_num=None):
        ce_loss = self.base_criterion(outputs, targets)
        
        # Curriculum learning for lambda
        curr_lambda = self.lambda_proto
        if round_num is not None:
             if round_num < 5:
                  curr_lambda = 0.1 * round_num / 5.0
             else:
                  curr_lambda = 1.0
                  
        proto_loss = self.proto_criterion(embeddings, prototypes)
        
        return ce_loss + curr_lambda * proto_loss, ce_loss, proto_loss
