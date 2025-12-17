import torch
from data.custom_dataset import get_dataloader
from models.mae_vit import MAE_ViT

# 1. Test Data Loader
print("Testing Data Loader...")
loader = get_dataloader(
    root='./data', 
    split_file='./data/federated_splits.json', 
    client_id=0, 
    task_id=0, 
    batch_size=4
)
images, labels = next(iter(loader))
print(f"Batch Shape: {images.shape} (Expect [4, 3, 224, 224])")

# 2. Test Model
print("\nTesting MAE Model...")
model = MAE_ViT(model_name='vit_small_patch16_224')
loss, pred, mask = model(images)
print(f"Loss: {loss.item()}")
print(f"Prediction Shape: {pred.shape}")
print("✅ Phase 1 Infrastructure Ready!")