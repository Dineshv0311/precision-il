"""
src/dataset/robosuite_dataset.py
PyTorch Dataset abstraction for loading multi-modal (Vision + Proprioception)
demonstrations with configurable action chunking and sequence slicing.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class RobosuiteVisuomotorDataset(Dataset):
    """
    Multimodal Dataset for Imitation Learning.
    
    Args:
        dataset_path (str): Path to HDF5 dataset file.
        pred_horizon (int): Number of future action steps to predict (Chunk Size).
                            1 for Standard BC, >1 (e.g., 16) for ACT / Diffusion.
        obs_horizon (int): Number of historical observation frames to condition on.
    """
    def __init__(self, dataset_path="data/lift_demos.hdf5", pred_horizon=1, obs_horizon=1):
        self.dataset_path = dataset_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        
        self.indices = []
        self._cache_indices()

    def _cache_indices(self):
        """Precompute frame indices across all trajectories for fast sampling."""
        with h5py.File(self.dataset_path, "r") as f:
            demos = list(f["data"].keys())
            for demo in demos:
                demo_len = len(f[f"data/{demo}/actions"])
                # We can sample up to the trajectory end minus horizon bounds
                for t in range(demo_len):
                    self.indices.append((demo, t, demo_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_name, t, demo_len = self.indices[idx]
        
        with h5py.File(self.dataset_path, "r") as f:
            demo_grp = f[f"data/{demo_name}"]
            
            # --- Extract Observations ---
            # Shape: (H, W, C) -> PyTorch convention (C, H, W) normalized to [0, 1]
            img = demo_grp["obs/agentview_image"][t]
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            
            proprio = demo_grp["obs/proprio"][t].astype(np.float32)
            
            # --- Extract Action Sequence Chunk ---
            # If pred_horizon == 1 (Standard BC), extract single action at step t
            # If pred_horizon > 1 (ACT/Diffusion), extract a slice of length pred_horizon
            end_t = min(t + self.pred_horizon, demo_len)
            actions = demo_grp["actions"][t:end_t].astype(np.float32)
            
            # Pad terminal actions if slice is shorter than pred_horizon
            if len(actions) < self.pred_horizon:
                pad_len = self.pred_horizon - len(actions)
                last_action = actions[-1:] if len(actions) > 0 else np.zeros((1, 7), dtype=np.float32)
                padding = np.repeat(last_action, pad_len, axis=0)
                actions = np.vstack([actions, padding]) if len(actions) > 0 else padding

            if self.pred_horizon == 1:
                actions = actions.squeeze(0)  # Shape: (7,)

        return {
            "image": torch.from_numpy(img),             # (3, 128, 128)
            "proprio": torch.from_numpy(proprio),       # (9,)
            "action": torch.from_numpy(actions)         # (7,) for BC or (pred_horizon, 7) for ACT/Diffusion
        }