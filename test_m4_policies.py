"""
test_m4_policies.py
Validates BC, ACT, and Diffusion Policy modules on single batches.
"""

import torch
from torch.utils.data import DataLoader
from src.dataset.robosuite_dataset import RobosuiteVisuomotorDataset
from src.models.policies import BCPolicy, ACTPolicy, DiffusionPolicy

def test_module():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing policy verifications on: {device}")

    # 1. Verify Standard BC (pred_horizon=1)
    print("\n[1/3] Testing Standard BC Baseline...")
    bc_loader = DataLoader(RobosuiteVisuomotorDataset("data/lift_demos.hdf5", pred_horizon=1), batch_size=4, shuffle=True)
    batch_bc = next(iter(bc_loader))
    bc_model = BCPolicy().to(device)
    loss_bc, metrics_bc = bc_model.compute_loss(batch_bc["image"].to(device), batch_bc["proprio"].to(device), batch_bc["action"].to(device))
    loss_bc.backward()
    print(f"  -> BC Loss: {loss_bc.item():.4f} | Backward pass OK.")

    # 2. Verify ACT Policy (pred_horizon=16)
    print("\n[2/3] Testing ACT Policy (CVAE + Action Chunking Transformer)...")
    act_loader = DataLoader(RobosuiteVisuomotorDataset("data/lift_demos.hdf5", pred_horizon=16), batch_size=4, shuffle=True)
    batch_chunk = next(iter(act_loader))
    act_model = ACTPolicy(chunk_size=16).to(device)
    loss_act, metrics_act = act_model.compute_loss(batch_chunk["image"].to(device), batch_chunk["proprio"].to(device), batch_chunk["action"].to(device))
    loss_act.backward()
    print(f"  -> ACT Loss: {loss_act.item():.4f} (L1: {metrics_act['l1_loss']:.4f}, KL: {metrics_act['kl_loss']:.4f}) | Backward pass OK.")

    # 3. Verify Diffusion Policy (pred_horizon=16)
    print("\n[3/3] Testing Diffusion Policy (DDPM Denoiser)...")
    diff_model = DiffusionPolicy(chunk_size=16, num_diffusion_steps=20).to(device)
    loss_diff, metrics_diff = diff_model.compute_loss(batch_chunk["image"].to(device), batch_chunk["proprio"].to(device), batch_chunk["action"].to(device))
    loss_diff.backward()
    print(f"  -> Diffusion Loss: {loss_diff.item():.4f} | Backward pass OK.")

    # Test reverse denoising inference sampling
    sample_traj = diff_model.sample(batch_chunk["image"][:2].to(device), batch_chunk["proprio"][:2].to(device))
    print(f"  -> Diffusion Sampling Trajectory Output Shape: {sample_traj.shape} (Expected: [2, 16, 7])")

    print("\n" + "=" * 60)
    print("M4 POLICY HEADS ALL FUNCTIONAL AND BACKWARD-COMPATIBLE.")
    print("=" * 60)

if __name__ == "__main__":
    test_module()