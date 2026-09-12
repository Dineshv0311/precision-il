"""
demo.py
Interactive single-episode rollout demonstration for live panel review.
Usage:
    python demo.py --policy act
    python demo.py --policy bc
    python demo.py --policy diffusion
"""

import os
import sys
import argparse
import numpy as np
import torch
import imageio
import robosuite as suite
from robosuite.controllers import load_controller_config

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Windows DLL hook
if sys.platform == "win32":
    import site
    for path in site.getsitepackages():
        mujoco_dir = os.path.join(path, "mujoco")
        if os.path.exists(mujoco_dir):
            try:
                os.add_dll_directory(mujoco_dir)
            except AttributeError:
                os.environ["PATH"] = mujoco_dir + ";" + os.environ["PATH"]

from src.eval.evaluate_benchmark import load_policy_model, extract_multimodal_state

def run_live_demo(policy_name="act"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"models/{policy_name}_checkpoint.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint not found at {ckpt_path}. Train it first.")
        return

    print(f"Loading {policy_name.upper()} policy onto {device}...")
    model = load_policy_model(policy_name, ckpt_path, device)

    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        horizon=100,
    )

    obs = env.reset()
    frames = []
    action_buffer = []
    replan_stride = 8
    success = False

    print(f"Executing live rollout with {policy_name.upper()}...")
    for step in range(env.horizon):
        img_t, proprio_t, raw_frame = extract_multimodal_state(obs, device)
        frames.append(raw_frame)

        with torch.no_grad():
            if policy_name == "bc":
                action = model(img_t, proprio_t).squeeze(0).cpu().numpy()
            elif policy_name == "act":
                if len(action_buffer) == 0:
                    pred_chunk, _, _ = model(img_t, proprio_t)
                    action_buffer = list(pred_chunk.squeeze(0).cpu().numpy()[:replan_stride])
                action = action_buffer.pop(0)
            elif policy_name == "diffusion":
                if len(action_buffer) == 0:
                    pred_chunk = model.sample(img_t, proprio_t)
                    action_buffer = list(pred_chunk.squeeze(0).cpu().numpy()[:replan_stride])
                action = action_buffer.pop(0)

        # Snap gripper command
        action[6] = 1.0 if action[6] > 0.0 else -1.0
        obs, reward, done, info = env.step(action)

        if env._check_success():
            success = True
            print(f"--> [SUCCESS] Target lifted successfully at step {step}!")
            break

    env.close()
    
    os.makedirs("outputs", exist_ok=True)
    out_video = f"outputs/live_demo_{policy_name}.mp4"
    imageio.mimwrite(out_video, frames, fps=20)
    print(f"Rollout outcome: {'SUCCESS' if success else 'TIMEOUT'}")
    print(f"Saved video replay to: {out_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="act", choices=["bc", "act", "diffusion"])
    args = parser.parse_args()
    run_live_demo(args.policy)