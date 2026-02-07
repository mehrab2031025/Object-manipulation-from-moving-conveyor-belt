import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
import time


class UR5ConveyorEnv(gym.Env):
    """
    UR5 grasping from conveyor belt.
    Uses default camera view for vision.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.n_joints = 6

        # Action space: joint velocities
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )

        # Observation space: 14 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

        # Conveyor parameters
        self.conveyor_speed_range = (0.03, 0.07)
        self.current_conveyor_speed = 0.05

        self.max_steps = 500
        self.current_step = 0

        # Get body/site IDs
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.obj_body_id]]
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # Camera renderer
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Home position
        home_pos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        self.data.qpos[:self.n_joints] = home_pos
        self.data.ctrl[:self.n_joints] = home_pos

        # Random conveyor speed
        self.current_conveyor_speed = np.random.uniform(0.03, 0.07)

        # Reset conveyor position
        conveyor_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "conveyor_slide")
        conveyor_qpos_addr = self.model.jnt_qposadr[conveyor_jnt_id]
        self.data.qpos[conveyor_qpos_addr] = -0.5

        # Object position
        self.data.qpos[self.obj_qpos_addr:self.obj_qpos_addr + 3] = [-0.8, -0.8, 0.13]
        self.data.qpos[self.obj_qpos_addr + 3:self.obj_qpos_addr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Initialize prev_distance
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        obj_pos = self.data.qpos[self.obj_qpos_addr:self.obj_qpos_addr + 3]
        self.prev_distance = np.linalg.norm(gripper_pos - obj_pos)

        self.current_step = 0
        self.grasped = False

        return self._get_obs(), {}

    def step(self, action):
        scaled_action = action * 0.5
        self.data.ctrl[:self.n_joints] = scaled_action
        self.data.ctrl[6] = self.current_conveyor_speed

        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        joint_pos = self.data.qpos[:self.n_joints].copy()
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        obj_pos = self.data.qpos[self.obj_qpos_addr:self.obj_qpos_addr + 3].copy()
        obj_class = 0.0
        obj_yaw = 0.0

        obs = np.concatenate([
            joint_pos,
            gripper_pos,
            [obj_class],
            obj_pos,
            [obj_yaw]
        ]).astype(np.float32)

        return obs

    def _compute_reward(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        obj_pos = self.data.qpos[self.obj_qpos_addr:self.obj_qpos_addr + 3]

        # Distance to object
        distance = np.linalg.norm(gripper_pos - obj_pos)

        # 1. Distance reward: MAIN REWARD
        # +1000 per meter of improvement
        distance_reward = 1000 * (self.prev_distance - distance)
        self.prev_distance = distance

        # 2. Success bonus: +3000 when within 3cm
        success_reward = 0
        if distance < 0.03:
            success_reward = 3000
            print(f"  *** GRASP SUCCESS! Distance={distance:.4f}m")

        # 3. Small time penalty (not bonus!) to encourage speed
        # -0.01 per step
        time_penalty = -0.01

        # 4. Collision/object lost penalty
        penalty = 0
        if abs(obj_pos[1] - (-0.8)) > 0.25:  # Object fell off
            penalty = -100

        # Total reward (should be mostly from distance_reward)
        total_reward = distance_reward + success_reward + time_penalty + penalty

        # Debug print
        if self.current_step % 50 == 0:
            print(f"  dist={distance:.3f}m, dist_reward={distance_reward:.1f}, "
                  f"total={total_reward:.1f}")

        return total_reward

    def _check_termination(self):
        obj_pos = self.data.qpos[self.obj_qpos_addr:self.obj_qpos_addr + 3]

        if obj_pos[2] < 0.05:
            return True
        if obj_pos[0] > 0.5:
            return True
        if abs(obj_pos[1] - (-0.8)) > 0.3:
            return True

        return False

    def get_camera_image(self):
        """Capture image from default camera view."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)

        # Use default camera
        self.renderer.update_scene(self.data)
        rgb = self.renderer.render()
        return rgb

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_PATH = os.path.join(CURRENT_DIR, "ur5e_robot", "ur5e_with_conveyor.xml")

    print("Testing UR5 Conveyor Environment...")
    env = UR5ConveyorEnv(XML_PATH)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")

    # Run episode
    total_reward = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        time.sleep(0.01)

        if i % 50 == 0:
            obj_pos = obs[10:13]
            print(f"Step {i}: Object at {obj_pos}, reward={reward:.2f}")

        if terminated or truncated:
            print(f"Episode ended at step {i}, total reward={total_reward:.2f}")
            break

    # Capture camera image
    print("\nCapturing camera image...")
    rgb = env.get_camera_image()
    print(f"Image shape: {rgb.shape}")
    print(f"Image min: {rgb.min()}, max: {rgb.max()}")

    # Save and display
    plt.imsave("camera_view.png", rgb)
    print("Saved to camera_view.png")

    plt.imshow(rgb)
    plt.title("Camera View")
    plt.axis('off')
    plt.show()

    env.close()
    print("Test complete!")

