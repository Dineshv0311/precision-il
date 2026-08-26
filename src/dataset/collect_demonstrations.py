"""
src/dataset/collect_demonstrations.py
Scripted Expert Trajectory Generator for the Robosuite Lift Task.
Saves observations (RGB + Proprioception) and actions into HDF5 format.
"""

import os
import sys
import h5py
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

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


class ScriptedLiftExpert:
    """State-machine expert solving the Lift task via Cartesian P-control."""
    
    def __init__(self, kp=6.0):
        self.kp = kp
        self.state = "HOVER"
        self.hover_height = 0.12  # meters above cube
        self.grasp_height = 0.015 # meters above cube center
        self.lift_target = 0.18   # height to lift
        self.counter = 0

    def reset(self):
        self.state = "HOVER"
        self.counter = 0

    def get_action(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]
        
        # Target action: [dx, dy, dz, ax, ay, az, grasp]
        action = np.zeros(7, dtype=np.float32)
        
        target_pos = cube_pos.copy()

        if self.state == "HOVER":
            target_pos[2] += self.hover_height
            diff = target_pos - eef_pos
            action[:3] = np.clip(diff * self.kp, -1.0, 1.0)
            action[6] = -1.0  # Open gripper
            
            if np.linalg.norm(diff[:2]) < 0.015 and abs(diff[2]) < 0.02:
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            target_pos[2] += self.grasp_height
            diff = target_pos - eef_pos
            action[:3] = np.clip(diff * self.kp, -1.0, 1.0)
            action[6] = -1.0  # Open gripper
            
            if abs(diff[2]) < 0.01:
                self.state = "GRASP"
                self.counter = 0

        elif self.state == "GRASP":
            target_pos[2] += self.grasp_height
            diff = target_pos - eef_pos
            action[:3] = np.clip(diff * self.kp, -0.5, 0.5)
            action[6] = 1.0   # Close gripper
            self.counter += 1
            if self.counter > 15:  # Allow gripper fingers to stabilize contact
                self.state = "LIFT"

        elif self.state == "LIFT":
            target_pos[2] += self.lift_target
            diff = target_pos - eef_pos
            action[:3] = np.clip(diff * self.kp, -1.0, 1.0)
            action[6] = 1.0   # Keep gripper closed

        return action


def collect_dataset(num_episodes=50, output_path="data/lift_demos.hdf5"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,  # Optimized for vision backbone training efficiency
        camera_widths=128,
        reward_shaping=False,
        control_freq=20,
        horizon=100,
    )

    expert = ScriptedLiftExpert()
    
    hdf5_file = h5py.File(output_path, "w")
    data_group = hdf5_file.create_group("data")
    
    successful_episodes = 0
    total_attempts = 0

    print(f"Starting data collection: Target = {num_episodes} successful demonstrations...")

    while successful_episodes < num_episodes:
        total_attempts += 1
        obs = env.reset()
        expert.reset()
        
        ep_agentview = []
        ep_wrist = []
        ep_proprio = []
        ep_actions = []
        
        success = False

        for step in range(env.horizon):
            action = expert.get_action(obs)
            
            # Record current observation before stepping
            # Apply vertical flip [::-1] for OpenGL alignment
            agentview_img = obs["agentview_image"][::-1, :, :].copy()
            wrist_img = obs["robot0_eye_in_hand_image"][::-1, :, :].copy()
            
            proprio = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"]
            ]) # 9-dimensional vector
            
            ep_agentview.append(agentview_img)
            ep_wrist.append(wrist_img)
            ep_proprio.append(proprio)
            ep_actions.append(action)

            obs, reward, done, info = env.step(action)
            
            if env._check_success():
                success = True

        if success:
            ep_grp = data_group.create_group(f"demo_{successful_episodes}")
            obs_grp = ep_grp.create_group("obs")
            
            obs_grp.create_dataset("agentview_image", data=np.array(ep_agentview, dtype=np.uint8), compression="gzip")
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=np.array(ep_wrist, dtype=np.uint8), compression="gzip")
            obs_grp.create_dataset("proprio", data=np.array(ep_proprio, dtype=np.float32), compression="gzip")
            ep_grp.create_dataset("actions", data=np.array(ep_actions, dtype=np.float32), compression="gzip")
            
            successful_episodes += 1
            print(f"Recorded Demo {successful_episodes}/{num_episodes} (Trajectory Length: {len(ep_actions)})")
        else:
            print("Trajectory failed or timed out before success. Discarding...")

    hdf5_file.attrs["total_episodes"] = successful_episodes
    hdf5_file.close()
    env.close()
    print(f"\nDataset saved successfully to {output_path} (Success Rate: {successful_episodes}/{total_attempts})")


if __name__ == "__main__":
    collect_dataset(num_episodes=50, output_path="data/lift_demos.hdf5")