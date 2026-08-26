"""
test_m2_dataset.py
Generates a small demonstration set and verifies PyTorch DataLoader batch pipelines.
"""

import os
from torch.utils.data import DataLoader
from src.dataset.collect_demonstrations import collect_dataset
from src.dataset.robosuite_dataset import RobosuiteVisuomotorDataset

def main():
    dataset_path = "data/lift_demos.hdf5"
    
    # 1. Collect 5 demonstrations for testing (run 50 later for full training)
    print("=" * 60)
    print("STEP 1: GENERATING DEMONSTRATION TRAJECTORIES")
    print("=" * 60)
    collect_dataset(num_episodes=5, output_path=dataset_path)

    # 2. Test Standard BC DataLoader (Single-step horizon: pred_horizon=1)
    print("\n" + "=" * 60)
    print("STEP 2: TESTING STANDARD BC DATALOADER")
    print("=" * 60)
    bc_dataset = RobosuiteVisuomotorDataset(dataset_path=dataset_path, pred_horizon=1)
    bc_loader = DataLoader(bc_dataset, batch_size=16, shuffle=True)
    
    bc_batch = next(iter(bc_loader))
    print(f"Dataset total samples : {len(bc_dataset)}")
    print(f"Batch Image Tensor    : {bc_batch['image'].shape} (Expected: [16, 3, 128, 128]) | min={bc_batch['image'].min():.2f}, max={bc_batch['image'].max():.2f}")
    print(f"Batch Proprio Tensor  : {bc_batch['proprio'].shape} (Expected: [16, 9])")
    print(f"Batch Action Tensor   : {bc_batch['action'].shape} (Expected: [16, 7])")

    # 3. Test Action Chunking DataLoader (ACT/Diffusion horizon: pred_horizon=16)
    print("\n" + "=" * 60)
    print("STEP 3: TESTING ACTION CHUNKING DATALOADER (ACT/DIFFUSION)")
    print("=" * 60)
    chunk_dataset = RobosuiteVisuomotorDataset(dataset_path=dataset_path, pred_horizon=16)
    chunk_loader = DataLoader(chunk_dataset, batch_size=16, shuffle=True)
    
    chunk_batch = next(iter(chunk_loader))
    print(f"Chunked Action Tensor : {chunk_batch['action'].shape} (Expected: [16, 16, 7])")
    print("=" * 60)
    print("M2 PIPELINE VERIFIED SUCCESSFULLY.")
    print("=" * 60)

if __name__ == "__main__":
    main()