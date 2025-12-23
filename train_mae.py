import argparse
import os
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as TF
from PIL import Image

from transformers import ViTMAEForPreTraining, ViTImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-MAE on images (self-supervised)")
    parser.add_argument("data_dir", type=str, help="Path to dataset root (expects ImageFolder structure)")
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base",
                        choices=["facebook/vit-mae-base", "facebook/vit-mae-large", "facebook/vit-mae-huge"],
                        help="MAE model variant to use")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="MAE mask ratio (0-1)")
    parser.add_argument("--image_size", type=int, default=224, help="Input size for ViT")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16)")
    parser.add_argument("--output_dir", type=str, default="outputs/mae", help="Where to save checkpoints and viz")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def collate_processor(processor):
    def _collate(batch):
        # batch: list of (PIL image, label). We ignore label
        images = [b[0] for b in batch]
        out = processor(images=images, return_tensors="pt")
        # pixel_values: [B, 3, H, W]
        return out["pixel_values"], None
    return _collate


def save_image_grid(tensor_bchw: torch.Tensor, out_path: str, nrow: int = 4):
    # tensor expected 0..1 on cpu
    B, C, H, W = tensor_bchw.shape
    nrow = min(nrow, B)
    ncol = math.ceil(B / nrow)
    canvas = torch.ones(C, ncol * H, nrow * W)
    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            if idx >= B:
                break
            canvas[:, r * H:(r + 1) * H, c * W:(c + 1) * W] = tensor_bchw[idx]
            idx += 1
    canvas = (canvas.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(canvas).save(out_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    model = ViTMAEForPreTraining.from_pretrained(args.model_name)
    model.config.mask_ratio = float(args.mask_ratio)
    model.to(device)

    processor = ViTImageProcessor.from_pretrained(args.model_name)
    # Override size if needed
    if args.image_size:
        processor.size = {"height": args.image_size, "width": args.image_size}

    # Dataset: any folder with images in class subfolders (labels ignored)
    train_dataset = datasets.ImageFolder(root=args.data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_processor(processor),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.time()

        for batch_idx, (pixel_values, _) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(pixel_values)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % 50 == 0:
                avg = running_loss / batch_idx
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] loss={avg:.4f}")

        epoch_loss = running_loss / max(1, len(train_loader))
        scheduler.step()
        dur = time.time() - start
        print(f"Epoch {epoch} done in {dur:.1f}s, loss={epoch_loss:.4f}")

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = Path(args.output_dir) / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_name": args.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }, best_path)
            print(f"Saved best checkpoint to {best_path}")

        # Periodic save
        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_path = Path(args.output_dir) / f"epoch-{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_name": args.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # Visualize a few reconstructions at epoch end
        try:
            model.eval()
            with torch.no_grad():
                pixel_values, _ = next(iter(train_loader))
                pixel_values = pixel_values.to(device)
                outputs = model(pixel_values)
                reconstructed = outputs.logits
                reconstructed = model.unpatchify(reconstructed)

                mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
                std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
                reconstructed = reconstructed * std + mean
                reconstructed = torch.clamp(reconstructed, 0, 1).detach().cpu()

                # Also denormalize originals for side-by-side
                orig = pixel_values.detach().cpu()
                orig = orig * std.cpu() + mean.cpu()
                orig = torch.clamp(orig, 0, 1)

                save_image_grid(reconstructed[:8], str(Path(args.output_dir) / f"recon_epoch_{epoch}.png"), nrow=4)
                save_image_grid(orig[:8], str(Path(args.output_dir) / f"orig_epoch_{epoch}.png"), nrow=4)
                print("Saved visualization grids.")
        except Exception as e:
            print(f"Visualization failed: {e}")

    # Save final
    final_path = Path(args.output_dir) / "final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_name": args.model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": best_loss,
    }, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
