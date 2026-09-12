"""
src/models/vision_encoder.py
Perception module with ResNet-18 + Spatial Softmax and Multimodal Fusion.
Fuses RGB Visual Embeddings with 15D Proprioception + Force/Torque Wrench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Keypoint Extractor.
    Converts 2D feature maps of shape (B, C, H, W) into (B, 2*C) normalized 
    spatial keypoint coordinates (x, y) in [-1, 1].
    """
    def __init__(self, height, width, num_channels, temperature=1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.temperature = temperature

        # Create static spatial coordinate grids [-1, 1]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing="xy"
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * c, h * w) / self.temperature
        softmax_attention = F.softmax(x, dim=-1)

        expected_x = torch.sum(softmax_attention * self.pos_x, dim=-1, keepdim=True)
        expected_y = torch.sum(softmax_attention * self.pos_y, dim=-1, keepdim=True)

        expected_xy = torch.cat([expected_x, expected_y], dim=-1)
        return expected_xy.reshape(b, c * 2)


class VisuoForceFusionEncoder(nn.Module):
    """
    Multimodal Perceptual Encoder:
    1. Vision: ResNet-18 -> Spatial Softmax Keypoints -> Visual Feature (64D)
    2. Force-Proprio: LayerNorm + Linear Projection -> State Feature (64D)
    3. Fusion: Concatenation + MLP -> Unified Multimodal Representation (128D)
    """
    def __init__(
        self,
        proprio_dim=15,
        visual_feature_dim=64,
        fused_feature_dim=128,
        num_spatial_keypoints=32,
        pretrained=True
    ):
        super().__init__()
        
        # 1. Visual Backbone: ResNet-18 (truncated before GAP)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # 1x1 Conv to reduce channel depth: 512 -> num_spatial_keypoints (32)
        self.conv_proj = nn.Conv2d(512, num_spatial_keypoints, kernel_size=1)
        
        # Spatial Softmax on 4x4 spatial grid: 32 channels * 2 coordinates = 64D
        self.spatial_softmax = SpatialSoftmax(height=4, width=4, num_channels=num_spatial_keypoints)
        
        self.visual_proj = nn.Sequential(
            nn.Linear(num_spatial_keypoints * 2, visual_feature_dim),
            nn.LayerNorm(visual_feature_dim),
            nn.ReLU()
        )

        # 2. Proprioception + Force/Torque Projection
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 3. Multimodal Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(visual_feature_dim + 64, fused_feature_dim),
            nn.LayerNorm(fused_feature_dim),
            nn.ReLU(),
            nn.Linear(fused_feature_dim, fused_feature_dim),
            nn.LayerNorm(fused_feature_dim),
            nn.ReLU()
        )

    def forward(self, image, proprio):
        feat_map = self.backbone(image)
        feat_proj = self.conv_proj(feat_map)
        keypoints = self.spatial_softmax(feat_proj)
        z_vis = self.visual_proj(keypoints)

        z_proprio = self.proprio_proj(proprio)

        z_combined = torch.cat([z_vis, z_proprio], dim=-1)
        z_fused = self.fusion_mlp(z_combined)
        
        return z_fused