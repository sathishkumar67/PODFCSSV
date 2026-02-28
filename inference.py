import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import ViTMAEForPreTraining
from src.mae_with_adapter import inject_adapters
import os
import argparse

def load_model(checkpoint_path, device, bottleneck_dim=256):
    """
    Load the ViT-MAE model with adapters and the specified checkpoint.
    """
    print(f"Loading base model: facebook/vit-mae-base")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    
    print(f"Injecting adapters with bottleneck_dim={bottleneck_dim}")
    model = inject_adapters(model, bottleneck_dim=bottleneck_dim)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # The checkpoint structure from save_checkpoint in main.py is:
        # {
        #     'round': round_idx,
        #     'model_state_dict': base_model.state_dict(),
        #     'proto_bank': proto_bank.get_prototypes(),
        #     'history': training_history
        # }
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using randomly initialized adapters.")

    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, image_size=224):
    """
    Load and preprocess an image for ViT-MAE.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

@torch.no_grad()
def run_inference(model, image_tensor, device):
    """
    Run a forward pass to get reconstruction loss and features.
    """
    image_tensor = image_tensor.to(device)
    
    # 1. Full MAE Forward Pass (Reconstruction)
    outputs = model(image_tensor)
    loss = outputs.loss.item() if hasattr(outputs, 'loss') else None
    
    # 2. Feature Extraction (Global Average Pooling)
    # Following the logic in src/client.py FederatedClient._extract_features
    encoder_output = model.vit(image_tensor)
    embeddings = encoder_output.last_hidden_state.mean(dim=1)
    
    return loss, embeddings

def main():
    parser = argparse.ArgumentParser(description="Inference script for ViT-MAE with adapters.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", help="Path to the model checkpoint.")
    parser.add_argument("--bottleneck_dim", type=int, default=256, help="Bottleneck dimension for adapters.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device, args.bottleneck_dim)

    # Preprocess image
    if not os.path.exists(args.image):
        print(f"Error: Image folder/file {args.image} not found.")
        return
    
    image_tensor = preprocess_image(args.image)

    # Run inference
    loss, embeddings = run_inference(model, image_tensor, device)

    print("\n--- Inference Results ---")
    if loss is not None:
        print(f"Reconstruction Loss: {loss:.6f}")
    print(f"Embedding Shape: {embeddings.shape}")
    print(f"Embedding (first 5 values): {embeddings[0, :5].cpu().numpy()}")
    print("--------------------------")

if __name__ == "__main__":
    main()
