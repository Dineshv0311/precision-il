"""
src/dataset/collect_demonstrations.py
Scripted Expert Trajectory Generator for the Robosuite Lift Task.
Captures RGB images, 6-DoF Force/Torque Wrench, and Proprioception (15D State).
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
        self.hover_height = 0.12   # meters above cube
        self.grasp_height = 0.015  # meters above cube center
        self.lift_target = 0.18    # target height to lift
        self.counter = 0

    def reset(self):
        self.state = "HOVER"
        self.counter = 0

    def get_action(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]
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
            action[6] = -1.0
            if abs(diff[2]) < 0.01:
                self.state = "GRASP"
                self.counter = 0

        elif self.state == "GRASP":
            target_pos[2] += self.grasp_height
            diff = target_pos - eef_pos
            action[:3] = np.clip(diff * self.kp, -0.5, 0.5)
            action[6] = 1.0   # Close gripper
            self.counter += 1
            if self.counter > 15:  # Allow contact forces to settle
                self.state = "LIFT"

        elif self.state == "LIFT":
            target_pos[2] += self.lift_target
            diff = target_pos - eef_pos
            action[:3] = np.clip(diff * self.kp, -1.0, 1.0)
            action[6] = 1.0   # Keep closed

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
        camera_heights=128,
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

    print(f"Collecting {num_episodes} demonstrations with 15D Vision-Force-Proprioception states...")

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
            
            # 1. Vision: Flip vertically for OpenGL coordinate alignment
            agentview_img = obs["agentview_image"][::-1, :, :].copy()
            wrist_img = obs["robot0_eye_in_hand_image"][::-1, :, :].copy()
            
            # 2. Force / Torque Wrench (Safe fallback if zero contact)
            eef_force = obs.get("robot0_eef_force", np.zeros(3, dtype=np.float32))
            eef_torque = obs.get("robot0_eef_torque", np.zeros(3, dtype=np.float32))

            # 3. 15-Dimensional Multimodal Proprioception Vector
            proprio = np.concatenate([
                obs["robot0_eef_pos"],       # 3D: [x, y, z]
                obs["robot0_eef_quat"],      # 4D: [w, x, y, z]
                obs["robot0_gripper_qpos"],  # 2D: [finger_left, finger_right]
                eef_force,                   # 3D: [fx, fy, fz]
                eef_torque                   # 3D: [tx, ty, tz]
            ]).astype(np.float32)            # Total = 15D
            
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
            print(f"Recorded Demo {successful_episodes}/{num_episodes} (Length: {len(ep_actions)})")
        else:
            print("Trajectory failed / timed out. Discarding...")

    hdf5_file.attrs["total_episodes"] = successful_episodes
    hdf5_file.close()
    env.close()
    print(f"\n[DONE] Saved dataset with 15D state vectors to {output_path}")


if __name__ == "__main__":
    # Generate 50 high-quality expert demonstrations for our benchmark
    collect_dataset(num_episodes=50, output_path="data/lift_demos.hdf5")