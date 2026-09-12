"""
train.py
Unified Training Script for BC, ACT, and Diffusion Policies.
Usage:
    python train.py --policy bc --epochs 25
    python train.py --policy act --epochs 30
    python train.py --policy diffusion --epochs 35
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset.robosuite_dataset import RobosuiteVisuomotorDataset
from src.models.policies import BCPolicy, ACTPolicy, DiffusionPolicy

def parse_args():
    parser = argparse.ArgumentParser(description="Train Visuomotor Manipulation Policies")
    parser.add_argument("--policy", type=str, required=True, choices=["bc", "act", "diffusion"], help="Policy architecture to train")
    parser.add_argument("--dataset", type=str, default="data/lift_demos.hdf5")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="models")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"TRAINING CONFIGURATION: {args.policy.upper()} POLICY")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 60)

    pred_horizon = 1 if args.policy == "bc" else 16
    dataset = RobosuiteVisuomotorDataset(dataset_path=args.dataset, pred_horizon=pred_horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.policy == "bc":
        model = BCPolicy().to(device)
    elif args.policy == "act":
        model = ACTPolicy(chunk_size=16).to(device)
    elif args.policy == "diffusion":
        model = DiffusionPolicy(chunk_size=16, num_diffusion_steps=20).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    save_path = os.path.join(args.save_dir, f"{args.policy}_checkpoint.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in loader:
            images = batch["image"].to(device)
            proprio = batch["proprio"].to(device)
            actions = batch["action"].to(device)

            optimizer.zero_grad()
            loss, _ = model.compute_loss(images, proprio, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.5f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
                "policy_type": args.policy
            }, save_path)

    print("=" * 60)
    print(f"Training Complete! Best model saved to: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()