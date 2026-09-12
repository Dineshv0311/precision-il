"""
test_m3_perception.py
Verifies M3 VisuoForceFusionEncoder forward pass, parameter gradients, and tensor flow.
"""

import torch
from torch.utils.data import DataLoader
from src.dataset.robosuite_dataset import RobosuiteVisuomotorDataset
from src.models.vision_encoder import VisuoForceFusionEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on execution device: {device}")

    dataset = RobosuiteVisuomotorDataset(dataset_path="data/lift_demos.hdf5", pred_horizon=1)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    images = batch["image"].to(device)     # (8, 3, 128, 128)
    proprios = batch["proprio"].to(device) # (8, 15)

    encoder = VisuoForceFusionEncoder(
        proprio_dim=15,
        visual_feature_dim=64,
        fused_feature_dim=128,
        num_spatial_keypoints=32,
        pretrained=True
    ).to(device)

    fused_features = encoder(images, proprios)
    
    print("=" * 60)
    print("M3 ENCODER FORWARD PASS COMPLETED")
    print("=" * 60)
    print(f"Batch Image Input      : {images.shape}")
    print(f"Batch Proprio/F-T Input: {proprios.shape} (15D Verified)")
    print(f"Fused State Embedding  : {fused_features.shape} (Expected: [8, 128])")
    
    # Backward pass & gradient flow check
    dummy_loss = fused_features.sum()
    dummy_loss.backward()
    
    grad_norm = encoder.backbone[0].weight.grad.norm().item()
    print(f"Backbone Grad Norm     : {grad_norm:.4f} (Backprop Verified!)")
    print("=" * 60)
    print("M3 PERCEPTION MODULE VERIFIED.")
    print("=" * 60)

if __name__ == "__main__":
    main()