"""
Module 1: Environment Instantiation & Verification Script
Tasks:
1. Initialize robosuite Lift environment with Panda robot.
2. Configure OSC_POSE controller.
3. Capture RGB camera observations + proprioceptive states.
4. Run closed-loop random action step and verify tensor outputs.
5. Save verification frame (PNG) and rollout video (MP4).
"""

import os
import numpy as np
import imageio

if os.name == "nt" and os.environ.get("MUJOCO_GL", "").lower().strip() == "egl":
    os.environ["MUJOCO_GL"] = "wgl"

import robosuite as suite
from robosuite.controllers import load_controller_config

def setup_lift_environment():
    # 1. Load operational space controller (OSC_POSE)
    controller_config = load_controller_config(default_controller="OSC_POSE")
    
    # 2. Instantiate robosuite Lift task
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,                # Set to False for headless simulation / offscreen render
        has_offscreen_renderer=True,       # Required to pull RGB frames programmatically
        use_camera_obs=True,               # Return camera frames in observation dict
        camera_names=["agentview", "robot0_eye_in_hand"], # Global third-person + wrist camera
        camera_heights=256,
        camera_widths=256,
        reward_shaping=False,              # Unused for Imitation Learning (sparse/binary task)
        control_freq=20,                   # Policy runs at 20 Hz (MuJoCo steps at 500 Hz internally)
        horizon=100,                       # Max steps per episode
    )
    return env

def main():
    os.makedirs("outputs", exist_ok=True)
    env = setup_lift_environment()
    obs = env.reset()

    print("=" * 60)
    print("M1 ENVIRONMENT INITIALIZED SUCCESSFULLY")
    print("=" * 60)
    
    # Verify Key Multimodal Sensor Observations
    print("\n[Perception & Proprioception Keys Available]:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key:<25} : shape {str(value.shape):<15} | dtype: {value.dtype}")

    # Inspect the primary state vectors needed for M3 & M4
    rgb_agentview = obs["agentview_image"]          # (256, 256, 3) uint8
    eef_pos = obs["robot0_eef_pos"]                  # (3,) x, y, z
    eef_quat = obs["robot0_eef_quat"]                # (4,) quaternion (w, x, y, z)
    gripper_qpos = obs["robot0_gripper_qpos"]        # (2,) finger joint positions
    
    # Proprioceptive vector: [eef_pos (3), eef_quat (4), gripper_qpos (2)] = 9D
    proprio_vector = np.concatenate([eef_pos, eef_quat, gripper_qpos])
    print(f"\n[Primary Proprioceptive Vector Shape]: {proprio_vector.shape} (Expected: 9-dim)")

    # Execute random actions to verify action space dynamics
    video_frames = []
    print(f"\n[Action Space Dimension]: {env.action_dim} (Expected: 7 -> [dx, dy, dz, dax, day, daz, gripper])")
    print("Running 50 test steps with random actions...")

    for step in range(50):
        # Action in range [-1, 1]
        action = np.random.uniform(-1, 1, size=env.action_dim)
        obs, reward, done, info = env.step(action)
        
        # Robosuite renders images in standard OpenGL coordinate system (flipped vertically)
        # Flip vertically [::-1] so it saves right-side up
        frame = obs["agentview_image"][::-1, :, :]
        video_frames.append(frame)

        if done:
            break

    # Save verification assets
    snapshot_path = os.path.join("outputs", "m1_verification_frame.png")
    video_path = os.path.join("outputs", "m1_test_rollout.mp4")
    
    imageio.imwrite(snapshot_path, video_frames[0])
    imageio.mimwrite(video_path, video_frames, fps=20)

    print(f"\nSaved snapshot: {snapshot_path}")
    print(f"Saved rollout video: {video_path}")
    print("=" * 60)
    print("M1 Verification Complete. Ready for M2 (Data Acquisition).")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    main()