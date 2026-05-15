import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure  # NEW: Import the logger
from sim.visual_env import VisualDroneEnv
import os

def main():
    print("Initializing 20x20 SAR POMDP Environment for Training...")
    env = VisualDroneEnv(render_mode=None)
    
    # MLOps Check: Validate that our custom environment follows Gymnasium rules
    check_env(env, warn=True)
    
    # MLOps: Define log directories
    log_dir = "./experiments/logs/tensorboard/PPO_SAR_Grid_1/" # Best practice: create a specific folder per run
    model_dir = "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Building PPO Agent (policy_sar_v1)...")
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,      # Slightly lowered to help it learn complex patterns safely
        ent_coef=0.05,             # NEW: Forces the drone to be curious and explore the fog of war
        batch_size=128,            # Processes memory in larger chunks
    )

    # NEW: Configure the logger to output to terminal (stdout), CSV, and TensorBoard
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Increased timesteps for a much harder environment
    train_steps = 2_000_000  
    print(f"Commencing Training ({train_steps:,} timesteps)... This may take a while!")
    
    # We removed tb_log_name here because the new_logger handles the directory now
    model.learn(total_timesteps=train_steps) 

    print("Training Complete. Saving model...")
    model.save(f"{model_dir}/policy_sar_ppo")
    print("Model saved successfully to models/policy_sar_ppo.zip")

    env.close()

if __name__ == "__main__":
    main()