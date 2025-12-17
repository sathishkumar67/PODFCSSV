# FCL-SSLVision: Implementation Checklist & Quick Reference

## Quick Start Checklist

### Phase 0: Environment & Data (Week 1-2)
- [ ] Install PyTorch 1.12+, torchvision, timm
- [ ] Download and organize CIFAR-100 or Tiny-ImageNet
- [ ] Create non-IID data splits using Dirichlet distribution (α=0.5)
- [ ] Set up 10 simulated client nodes with non-uniform class distributions
- [ ] Create 5 sequential tasks with 20 classes each
- [ ] Implement DataLoader with proper task filtering
- [ ] Set up logging and visualization infrastructure

**Key Files to Create**:
- `data/prepare_data.py`
- `data/dirichlet_split.py`
- `config/experiment_config.yaml`

---

### Phase 1: SSL Pretraining (Week 3-5)
- [ ] Implement Masked Image Modeling (MAE-style)
  - [ ] Patch tokenization (16×16 or 8×8)
  - [ ] Random masking (75% ratio)
  - [ ] Vision Transformer encoder
  - [ ] Simple decoder (2-4 layers)
  - [ ] MSE reconstruction loss
- [ ] Implement federated SSL pretraining
  - [ ] Client-side local training loop
  - [ ] Server-side model aggregation (FedAvg)
  - [ ] Gradient computation and communication
- [ ] Test SSL convergence on single client
- [ ] Measure reconstruction loss decay
- [ ] Visualize learned patches

**Key Files to Create**:
- `ssl/masked_image_modeling.py`
- `ssl/mae_model.py`
- `training/train_ssl.py`
- `utils/visualization.py` (patch visualization)

**Expected Outcomes**:
- Reconstruction loss < 0.1 after 50 epochs
- Reasonable visual quality of reconstructed patches

---

### Phase 2: ViT Setup with Adapters (Week 6-7)
- [ ] Load pretrained ViT-Base or ViT-Small from timm
- [ ] Freeze ViT backbone weights
- [ ] Implement adapter modules
  - [ ] Adapter architecture (down-project + GELU + up-project)
  - [ ] Insert adapters after attention and FFN layers
  - [ ] Only ~1-5% of parameters trainable
- [ ] Implement class-incremental classifier expansion
  - [ ] Expandable linear layer
  - [ ] Old weight preservation
  - [ ] New weight initialization strategy
- [ ] Test forward pass with dummy inputs
- [ ] Verify parameter count and trainable parameters
- [ ] Measure inference time and memory

**Key Files to Create**:
- `models/vision_transformer.py`
- `models/adapter_modules.py`
- `models/incremental_classifier.py`

**Expected Outcomes**:
- Adapter parameters: 500K-5M out of 86M total
- Inference time < 100ms per image on GPU
- Memory usage < 8GB for batch size 32

---

### Phase 3: Prototype Management (Week 8-9)
- [ ] Implement local prototype extraction
  - [ ] Feature extraction from ViT (before classification)
  - [ ] Per-class mean computation
  - [ ] Prototype storage and management
  - [ ] Uncertainty estimation (variance-based)
- [ ] Implement server-side aggregation
  - [ ] Collect prototypes from all clients
  - [ ] Weighted averaging based on confidence scores
  - [ ] Handle missing classes at certain clients
  - [ ] Store global prototype repository
- [ ] Implement prototype quality assessment
  - [ ] Confidence score computation
  - [ ] Distance-based uncertainty
  - [ ] Outlier detection for spurious prototypes
- [ ] Test prototype visualization
  - [ ] t-SNE projection of prototypes
  - [ ] Class cluster visualization
  - [ ] Prototype drift monitoring

**Key Files to Create**:
- `distillation/prototype_extractor.py`
- `distillation/prototype_aggregator.py`
- `utils/prototype_visualization.py`

**Expected Outcomes**:
- Clean class separation in prototype space
- Prototype variance < 0.5 within each class
- Stable prototypes across communication rounds

---

### Phase 4: Distillation Losses (Week 10-11)
- [ ] Implement prototype anchored distillation loss
  - [ ] Soft target generation from prototypes
  - [ ] KL divergence computation
  - [ ] Temperature-scaled softmax
  - [ ] Weighted combination of CE + KL losses
- [ ] Implement mutual distillation (client↔server)
  - [ ] Client-to-server knowledge transfer
  - [ ] Server-to-client guidance
  - [ ] Bidirectional KL losses
- [ ] Implement supplementary regularization losses
  - [ ] Feature consistency loss (if needed)
  - [ ] Prototype regularization
- [ ] Test loss computation and gradient flow
- [ ] Verify loss values are reasonable (0-5 range typically)

**Key Files to Create**:
- `distillation/prototype_distillation_loss.py`
- `distillation/mutual_distillation.py`

**Expected Outcomes**:
- CE loss: 0.5-2.0 (depending on dataset)
- Distillation loss: 0.1-0.5
- Total loss: 0.6-2.5

---

### Phase 5: Replay Buffer (Week 12-13)
- [ ] Implement memory replay buffer
  - [ ] Fixed-size buffer initialization
  - [ ] FIFO sample insertion
  - [ ] Per-class sample balancing
  - [ ] Efficient random sampling
- [ ] Implement uncertainty-aware sample selection
  - [ ] Entropy-based uncertainty computation
  - [ ] Select top-k uncertain samples
  - [ ] Confidence-based weighting
- [ ] Implement buffer update strategies
  - [ ] LRU (Least Recently Used) eviction
  - [ ] Quality-based replacement
  - [ ] Class balance maintenance
- [ ] Test buffer with multiple tasks
  - [ ] Verify sample diversity
  - [ ] Monitor class distribution
  - [ ] Measure replay contribution to performance

**Key Files to Create**:
- `continual/replay_buffer.py`
- `continual/uncertainty_selector.py`

**Expected Outcomes**:
- Buffer capacity: 1000-2000 samples (≈1-2 images per class)
- Replay provides ~5-10% accuracy improvement
- Buffer remains balanced across classes

---

### Phase 6: Federated Orchestration (Week 14-15)
- [ ] Implement ClientNode class
  - [ ] Local task training loop
  - [ ] Prototype extraction and upload
  - [ ] Model weight updates and downloads
  - [ ] Replay buffer management
  - [ ] Loss computation with distillation
- [ ] Implement FederatedServer class
  - [ ] Client selection strategy (random sampling)
  - [ ] Model aggregation (FedAvg)
  - [ ] Prototype aggregation with confidence weighting
  - [ ] Gradient update broadcast
- [ ] Implement main training loop
  - [ ] Task iteration (5-10 tasks)
  - [ ] Communication rounds per task (5-10 rounds)
  - [ ] Local epochs per round (10 epochs)
  - [ ] Periodic evaluation
- [ ] Implement communication protocol
  - [ ] Model serialization/deserialization
  - [ ] Efficient gradient transmission (optional compression)
  - [ ] Prototype sharing format

**Key Files to Create**:
- `federated/client.py`
- `federated/server.py`
- `federated/communication.py`
- `training/main_federated_training.py`

**Expected Outcomes**:
- Successful training on 5 tasks × 10 clients
- No communication errors or model corruption
- Clear accuracy trends across tasks
- Training time: ~30 minutes per task on GPU

---

### Phase 7: Evaluation Metrics (Week 16-17)
- [ ] Implement accuracy tracking
  - [ ] Per-task accuracy after each task
  - [ ] Average accuracy across all tasks
  - [ ] Cumulative accuracy over time
- [ ] Implement forgetting metrics
  - [ ] Backward forgetting measure
  - [ ] Degradation per old task
  - [ ] Forgetting curves visualization
- [ ] Implement transfer metrics
  - [ ] Backward transfer (learning new helps old?)
  - [ ] Forward transfer (pre-training helps future?)
- [ ] Implement communication efficiency tracking
  - [ ] Bytes transmitted per client per round
  - [ ] Compression ratio (if applicable)
  - [ ] Communication rounds comparison
- [ ] Implement statistical analysis
  - [ ] Mean and std dev across runs
  - [ ] Confidence intervals
  - [ ] Statistical significance tests

**Key Files to Create**:
- `utils/metrics.py`
- `utils/evaluation.py`
- `evaluation/offline_evaluation.py`

**Expected Outcomes**:
- Average Accuracy: 70-80% on CIFAR-100
- Forgetting: < 15% (vs 30-40% for baselines)
- Communication reduction: 50-80% with adapters
- 3-5 independent runs for robustness

---

### Phase 8: Experiments & Ablation (Week 18-19)
- [ ] Baseline experiments
  - [ ] Run FedAvg without any enhancements
  - [ ] Run FedAvg + simple replay (no distillation)
  - [ ] Run FedAvg + prototypes (no replay)
  - [ ] Run Full Method (all components)
- [ ] Ablation studies
  - [ ] w/o SSL pretraining → train from scratch
  - [ ] w/o Prototype anchoring → only CE loss
  - [ ] w/o Replay buffer → only current task
  - [ ] w/o Server aggregation → local only
- [ ] Hyperparameter sensitivity
  - [ ] Distillation temperature: [2, 4, 6, 8]
  - [ ] Replay buffer size: [500, 1000, 2000]
  - [ ] Learning rate: [0.001, 0.01, 0.1]
- [ ] Scalability experiments
  - [ ] Vary number of clients: [5, 10, 20, 50]
  - [ ] Vary number of tasks: [3, 5, 10, 20]
  - [ ] Measure scaling efficiency
- [ ] Non-IID heterogeneity tests
  - [ ] Vary Dirichlet α: [0.1, 0.5, 1.0, 10.0]
  - [ ] Measure robustness to extreme heterogeneity

**Key Files to Create**:
- `experiments/baseline_experiments.py`
- `experiments/ablation_study.py`
- `experiments/hyperparameter_sweep.py`
- `experiments/analysis.py`

**Expected Outcomes**:
- Clear performance ranking: Full Method > Ablations > Baselines
- Significant improvements documented
- Publication-ready comparison tables
- Statistical significance demonstrated

---

## Code Structure Template

```python
# Core training loop structure
def main():
    # Initialize
    config = load_config()
    clients = setup_clients(config)
    server = setup_server(config)
    
    # Task loop
    for task_id in range(config.num_tasks):
        task_classes = get_task_classes(task_id)
        
        # Communication rounds per task
        for round_id in range(config.rounds_per_task):
            # Client selection
            selected = select_clients(clients, config.client_fraction)
            
            # Local training
            updates, prototypes = [], []
            for client in selected:
                upd, proto = client.train(task_id, server.model, 
                                         server.prototypes)
                updates.append(upd)
                prototypes.append(proto)
            
            # Server aggregation
            server.aggregate(updates, prototypes)
            
            # Broadcast
            broadcast_model(clients, server.model)
        
        # Task evaluation
        evaluate_all_tasks(clients, server, task_id)
    
    # Final evaluation and reporting
    report_results()

if __name__ == "__main__":
    main()
```

---

## Common Pitfalls to Avoid

1. **Data Leakage**: Ensure test set is never used during training
2. **Non-IID Inconsistency**: Use same data split for all experiments
3. **Random Seed**: Fix seeds for reproducibility
4. **Memory Issues**: Monitor GPU memory, use checkpointing
5. **Prototype Collapse**: Ensure prototypes don't all converge to same value
6. **Communication Overhead**: Track actual transmission sizes
7. **Gradual Drift**: Monitor model divergence from global model
8. **Forgetting Patterns**: Ensure forgetting primarily in old tasks, not new

---

## Testing Checkpoints

### After Week 5 (SSL Pretraining)
```python
# Checkpoint: SSL should converge
assert reconstruction_loss < 0.1
assert model_can_load_and_infer()
```

### After Week 7 (ViT Setup)
```python
# Checkpoint: ViT should classify
assert test_accuracy_random_init > 0.1
assert adapter_params < 1e7  # < 10M
```

### After Week 9 (Prototypes)
```python
# Checkpoint: Prototypes should be meaningful
assert prototypes_are_normalized()
assert class_separation > 0.5  # silhouette score
```

### After Week 11 (Distillation)
```python
# Checkpoint: Distillation improves over baseline
assert with_distill > without_distill
assert distill_loss_is_positive()
```

### After Week 13 (Replay)
```python
# Checkpoint: Replay reduces forgetting
assert with_replay > without_replay
assert buffer_maintains_class_balance()
```

### After Week 15 (Federated)
```python
# Checkpoint: Federated training works
assert federated_acc > local_avg
assert model_converges_across_rounds()
```

### After Week 17 (Evaluation)
```python
# Checkpoint: Metrics are computed correctly
assert avg_acc in [0, 100]
assert forgetting in [-10, 50]
```

### After Week 19 (Experiments)
```python
# Checkpoint: All experiments complete
assert len(results) >= num_baselines + num_ablations
assert results_are_statistically_significant()
assert paper_can_be_written()
```

---

## Performance Benchmarks (Target Values)

| Method | CIFAR-100 Avg Acc | Forgetting | Communication |
|--------|------------------|-----------|----------------|
| FedAvg (Baseline) | 45-50% | 35-40% | 100% |
| FedAvg + Replay | 55-60% | 25-30% | 100% |
| + Distillation | 65-70% | 15-20% | 100% |
| + SSL Pretraining | 70-75% | 10-15% | 100% |
| + Adapters | 70-75% | 10-15% | 20-30% |
| **Full Method** | **75-80%** | **5-10%** | **20-30%** |

---

## Paper Writing Checklist

- [ ] Abstract (150-250 words)
- [ ] Introduction with motivation
- [ ] Related work section (FedL, CL, SSL, Distillation)
- [ ] Problem formulation and notation
- [ ] Methodology section with all components
- [ ] Experimental setup and datasets
- [ ] Results and tables
- [ ] Ablation study analysis
- [ ] Discussion and limitations
- [ ] Conclusion and future work
- [ ] References (30-50 papers)
- [ ] Appendix with proofs/detailed algorithms

---

## Submission Venues

**Tier-1 Conferences**:
- ICML, NeurIPS, ICLR (top-tier)
- CVPR, ICCV (computer vision)
- IJCAI, AAAI (general AI)

**Tier-2 Journals**:
- IEEE Transactions on Neural Networks
- Knowledge-Based Systems
- Neurocomputing

**Domain-Specific**:
- IEEE INFOCOM (distributed systems)
- Journal of Machine Learning Research (JMLR)

---

## Final Notes

✅ This is a **feasible and impactful research project** combining four important areas
✅ State-of-the-art components are all available and well-documented
✅ 18-20 weeks is realistic for full implementation and validation
✅ Results should be publication-ready for top-tier venues
✅ Incremental testing ensures no surprises at the end

**Start coding now, good luck! 🚀**