import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
import time  # Add this
from simple_detector import ColorDetector


class UR5VisionEnv(gym.Env):
    """
    UR5 grasping with VISION - uses camera to detect object.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self.detector = ColorDetector()
        self.renderer = None

        self.n_joints = 6

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.conveyor_speed_range = (0.03, 0.07)
        self.current_conveyor_speed = 0.05
        self.max_steps = 500
        self.current_step = 0

        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.obj_body_id]]
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        self.detection_success_rate = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Home position
        home_pos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        self.data.qpos[:self.n_joints] = home_pos
        self.data.ctrl[:self.n_joints] = home_pos

        # Random conveyor speed
        self.current_conveyor_speed = np.random.uniform(0.5, 1)

        # Reset conveyor
        conveyor_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "conveyor_slide")
        conveyor_qpos_addr = self.model.jnt_qposadr[conveyor_jnt_id]
        self.data.qpos[conveyor_qpos_addr] = -0.5

        # Object position - start more to the right so it's visible
        self.data.qpos[self.obj_qpos_addr:self.obj_qpos_addr + 3] = [-0.4, -0.8, 0.13]
        self.data.qpos[self.obj_qpos_addr + 3:self.obj_qpos_addr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Initialize
        obs = self._get_obs()
        gripper_pos = obs[6:9]
        obj_pos = obs[10:13]
        self.prev_distance = np.linalg.norm(gripper_pos - obj_pos)

        self.current_step = 0
        self.grasped = False

        return obs, {}

    def step(self, action):
        scaled_action = action * 0.5
        self.data.ctrl[:self.n_joints] = scaled_action

        # *** FIX: Control conveyor ***
        self.data.ctrl[6] = 1000

        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        joint_pos = self.data.qpos[:self.n_joints].copy()
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()

        # Get camera image
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.renderer.update_scene(self.data)
        image = self.renderer.render()

        # Detect object
        detected_pos = self.detector.get_object_position(image)

        if detected_pos is not None:
            obj_pos = np.array(detected_pos)
            self.last_detected_pos = obj_pos
            self.detection_success_rate.append(1)
        else:
            if hasattr(self, 'last_detected_pos'):
                obj_pos = self.last_detected_pos
            else:
                # Better default - use actual sim position if no detection
                obj_pos = self.data.qpos[self.obj_qpos_addr:self.obj_qpos_addr + 3].copy()
            self.detection_success_rate.append(0)

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

    def _compute_reward(self, obs):
        gripper_pos = obs[6:9]
        obj_pos = obs[10:13]

        distance = np.linalg.norm(gripper_pos - obj_pos)

        distance_reward = 5000 * (self.prev_distance - distance)
        self.prev_distance = distance

        success_reward = 0
        if distance < 0.03:
            success_reward = 3000

        time_penalty = -0.01

        total_reward = distance_reward + success_reward + time_penalty

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

    def get_detection_rate(self):
        if not self.detection_success_rate:
            return 0.0
        return np.mean(self.detection_success_rate[-100:])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


# Test with delay
if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_PATH = os.path.join(CURRENT_DIR, "ur5e_robot", "ur5e_with_conveyor.xml")

    print("Testing VISION-based environment with delay...")
    env = UR5VisionEnv(XML_PATH)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Object position (from vision): {obs[10:13]}")

    # Run with delay
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # *** ADD DELAY ***
        time.sleep(0.01)  # 10ms delay

        if i % 10 == 0:  # Print more frequently
            print(f"Step {i}: Object pos = {obs[10:13]}, "
                  f"Detection rate = {env.get_detection_rate():.1%}")

        if terminated or truncated:
            break

    print(f"Final detection success rate: {env.get_detection_rate():.1%}")
    env.close()
    print("Test complete!")