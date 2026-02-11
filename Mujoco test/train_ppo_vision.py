from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from ur5_vision_env import UR5VisionEnv
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "ur5e_robot", "ur5e_with_conveyor.xml")


def main():
    print("Training PPO with VISION...")
    print("This uses camera to detect object position")

    # Create vision-based environment
    env = UR5VisionEnv(XML_PATH)
    eval_env = UR5VisionEnv(XML_PATH)

    # PPO with same hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard_logs_vision/"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_vision/",
        log_path="./logs_vision/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print("Training for 200,000 steps...")
    model.learn(
        total_timesteps=200_000,
        callback=eval_callback,
        progress_bar=True
    )

    model.save("ur5_vision_ppo_final")
    print("Training complete!")
    print("Best model saved to ./models_vision/best_model.zip")


if __name__ == "__main__":
    main()