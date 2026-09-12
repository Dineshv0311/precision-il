"""
src/eval/evaluate_benchmark.py
Closed-Loop Benchmark Suite for Vision-Force Manipulation Policies.
Evaluates BC, ACT, and Diffusion Policy under systematically varied initial-pose error.
"""

import os
import sys

# 1. Add project root to Python search path (Fixes ModuleNotFoundError: No module named 'src')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 2. Windows DLL hook for MuJoCo
if sys.platform == "win32":
    import site
    for path in site.getsitepackages():
        mujoco_dir = os.path.join(path, "mujoco")
        if os.path.exists(mujoco_dir):
            try:
                os.add_dll_directory(mujoco_dir)
            except AttributeError:
                os.environ["PATH"] = mujoco_dir + ";" + os.environ["PATH"]

import json
import argparse
import numpy as np
import torch
import imageio
import robosuite as suite
from robosuite.controllers import load_controller_config

# Now this import will resolve cleanly from anywhere:
from src.models.policies import BCPolicy, ACTPolicy, DiffusionPolicy

# Perturbation Tier Definitions: (position_jitter_meters, orientation_jitter_degrees)
PERTURBATION_TIERS = {
    "Tier_0_Nominal": (0.000, 0.0),
    "Tier_1_Low":     (0.010, 5.0),
    "Tier_2_Medium":  (0.025, 15.0),
    "Tier_3_High":    (0.040, 30.0),
}


def apply_pose_perturbation(env, pos_jitter_m, rot_jitter_deg):
    """Applies programmatic Cartesian and yaw perturbations to the cube in MuJoCo."""
    if pos_jitter_m == 0.0 and rot_jitter_deg == 0.0:
        return

    # Find cube joint in MuJoCo kinematic tree
    joint_name = None
    for name in ["cube_joint0", "cube_jnt", "cube_joint"]:
        if name in env.sim.model.joint_names:
            joint_name = name
            break

    if joint_name is not None:
        qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
        
        # Robosuite returns (start_idx, end_idx) tuple; extract start_idx
        start_idx = qpos_addr[0] if isinstance(qpos_addr, (tuple, list, np.ndarray)) else qpos_addr

        # Apply planar (X, Y) translation jitter
        dx = np.random.uniform(-pos_jitter_m, pos_jitter_m)
        dy = np.random.uniform(-pos_jitter_m, pos_jitter_m)
        env.sim.data.qpos[start_idx] += dx
        env.sim.data.qpos[start_idx + 1] += dy

        # Apply planar Yaw rotation jitter via quaternion multiplication
        if rot_jitter_deg > 0:
            yaw = np.radians(np.random.uniform(-rot_jitter_deg, rot_jitter_deg))
            half_yaw = yaw / 2.0
            dqw = np.cos(half_yaw)
            dqz = np.sin(half_yaw)

            # Current quaternion: [qw, qx, qy, qz]
            qw = env.sim.data.qpos[start_idx + 3]
            qx = env.sim.data.qpos[start_idx + 4]
            qy = env.sim.data.qpos[start_idx + 5]
            qz = env.sim.data.qpos[start_idx + 6]

            # Hamilton product: q_new = dq * q
            new_qw = dqw * qw - dqz * qz
            new_qx = dqw * qx + dqz * qy
            new_qy = dqw * qy - dqz * qx
            new_qz = dqw * qz + dqz * qw

            norm = np.sqrt(new_qw**2 + new_qx**2 + new_qy**2 + new_qz**2) + 1e-8
            env.sim.data.qpos[start_idx + 3] = new_qw / norm
            env.sim.data.qpos[start_idx + 4] = new_qx / norm
            env.sim.data.qpos[start_idx + 5] = new_qy / norm
            env.sim.data.qpos[start_idx + 6] = new_qz / norm

        env.sim.forward()


def extract_multimodal_state(obs, device):
    """Extracts and normalizes RGB (C, H, W) and 15D proprioception/wrench tensors."""
    img = obs["agentview_image"][::-1, :, :].copy()
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0) / 255.0

    eef_force = obs.get("robot0_eef_force", np.zeros(3, dtype=np.float32))
    eef_torque = obs.get("robot0_eef_torque", np.zeros(3, dtype=np.float32))

    proprio = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
        eef_force,
        eef_torque
    ]).astype(np.float32)
    proprio_tensor = torch.from_numpy(proprio).float().unsqueeze(0)

    return img_tensor.to(device), proprio_tensor.to(device), img


def load_policy_model(policy_name, checkpoint_path, device):
    """Loads model architecture and restores trained weights."""
    if policy_name == "bc":
        model = BCPolicy().to(device)
    elif policy_name == "act":
        model = ACTPolicy(chunk_size=16).to(device)
    elif policy_name == "diffusion":
        model = DiffusionPolicy(chunk_size=16, num_diffusion_steps=20).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_benchmark(policies=["bc", "act", "diffusion"], trials_per_tier=5, save_videos=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/eval_rollouts", exist_ok=True)

    # Initialize standard Robosuite Lift environment
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
        reward_shaping=False,
        control_freq=20,
        horizon=100,
    )

    benchmark_results = {p: {} for p in policies}

    print("=" * 75)
    print("STARTING SYSTEMATIC INITIAL-POSE ERROR BENCHMARK")
    print(f"Policies: {policies} | Trials per tier: {trials_per_tier} | Device: {device}")
    print("=" * 75)

    for pol_name in policies:
        ckpt_path = f"models/{pol_name}_checkpoint.pt"
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping {pol_name}.")
            continue

        model = load_policy_model(pol_name, ckpt_path, device)
        print(f"\nEvaluating Policy: >>> {pol_name.upper()} <<<")

        for tier_name, (pos_jitter, rot_jitter) in PERTURBATION_TIERS.items():
            successes = 0
            tier_frames = []

            for trial in range(trials_per_tier):
                obs = env.reset()
                apply_pose_perturbation(env, pos_jitter, rot_jitter)

                # Re-sync observation after physical perturbation
                obs = env._get_observations()

                video_frames = []
                action_buffer = []  # For chunked execution (ACT / Diffusion)
                replan_stride = 8   # Receding horizon replanning stride

                for step in range(env.horizon):
                    img_t, proprio_t, raw_frame = extract_multimodal_state(obs, device)
                    if save_videos and trial == 0:
                        video_frames.append(raw_frame)

                    with torch.no_grad():
                        if pol_name == "bc":
                            action = model(img_t, proprio_t).squeeze(0).cpu().numpy()
                        elif pol_name == "act":
                            if len(action_buffer) == 0:
                                pred_chunk, _, _ = model(img_t, proprio_t)
                                action_buffer = list(pred_chunk.squeeze(0).cpu().numpy()[:replan_stride])
                            action = action_buffer.pop(0)
                        elif pol_name == "diffusion":
                            if len(action_buffer) == 0:
                                pred_chunk = model.sample(img_t, proprio_t)
                                action_buffer = list(pred_chunk.squeeze(0).cpu().numpy()[:replan_stride])
                            action = action_buffer.pop(0)
                    action[6] = 1.0 if action[6] > 0.0 else -1.0
                    obs, reward, done, info = env.step(action)

                    if env._check_success():
                        successes += 1
                        break

                if save_videos and trial == 0 and len(video_frames) > 0:
                    tier_frames = video_frames

            success_rate = (successes / trials_per_tier) * 100.0
            benchmark_results[pol_name][tier_name] = success_rate
            print(f"  [{tier_name:<16}] Success Rate: {success_rate:>5.1f}% ({successes}/{trials_per_tier})")

            # Save sample video of first trial for each tier
            if save_videos and len(tier_frames) > 0:
                vid_path = f"outputs/eval_rollouts/{pol_name}_{tier_name}.mp4"
                imageio.mimwrite(vid_path, tier_frames, fps=20)

    env.close()

    # Save metrics to JSON
    with open("outputs/benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)

    print("\n" + "=" * 75)
    print("BENCHMARK COMPLETE. Results saved to outputs/benchmark_results.json")
    print("=" * 75)
    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5, help="Rollouts per tier (5 for fast eval, 10-20 for final report)")
    args = parser.parse_args()
    run_benchmark(trials_per_tier=args.trials)