import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sim.visual_env import VisualDroneEnv
import os

def main():
    print("Initializing 2D Drone Environment for Training...")
    # render_mode=None ensures it trains instantly without opening Pygame
    env = VisualDroneEnv(render_mode=None)
    
    # MLOps Check: Validate that our custom environment follows Gymnasium rules
    check_env(env, warn=True)
    
    # MLOps: Define log directories
    log_dir = "./experiments/logs/tensorboard/"
    model_dir = "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Building PPO Agent (policy_v2)...")
    # MultiInputPolicy is REQUIRED because our Observation Space is a Dictionary (coords + battery)
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0005,
        tensorboard_log=log_dir
    )

    print("Commencing Training (100,000 timesteps)...")
    # 100k steps is plenty of time for PPO to learn a 10x10 maze
    model.learn(total_timesteps=100_000, tb_log_name="PPO_GridMaze")

    print("Training Complete. Saving model...")
    model.save(f"{model_dir}/policy_v2_ppo")
    print("Model saved successfully to models/policy_v2_ppo.zip")

    env.close()

if __name__ == "__main__":
    main()