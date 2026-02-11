import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from stable_baselines3 import PPO
from ur5_vision_env import UR5VisionEnv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "ur5e_robot", "ur5e_with_conveyor.xml")


def main():
    print("Loading VISION-based trained model...")

    env = UR5VisionEnv(XML_PATH)
    model = PPO.load("models_vision/best_model.zip", env=env)

    print("Model loaded! Starting demonstration...")
    print("This robot uses CAMERA to detect the object!")

    obs, info = env.reset()
    mj_model = env.model
    mj_data = env.data

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -30
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0, -0.4, 0.3]

        episode_reward = 0
        step = 0
        successes = 0

        while viewer.is_running():
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if reward > 1000:  # Success bonus
                successes += 1
                print(f"*** GRASP SUCCESS #{successes} at step {step}! ***")

            if step % 50 == 0:
                obj_pos = obs[10:13]
                print(f"Step {step}: Object at x={obj_pos[0]:.2f}, "
                      f"Vision detection rate: {env.get_detection_rate():.1%}")

            viewer.sync()
            time.sleep(0.01)

            if terminated or truncated:
                print(f"\n*** Episode Complete ***")
                print(f"Total Reward: {episode_reward:.0f}")
                print(f"Successes: {successes}")
                time.sleep(2)

                obs, info = env.reset()
                episode_reward = 0
                step = 0
                successes = 0


if __name__ == "__main__":
    main()