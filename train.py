import gymnasium as gym
import mlflow # NEW: Import MLflow
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure  
from sim.visual_env import VisualDroneEnv
import os

def main():
    print("Initializing 20x20 SAR POMDP Environment for Training...")
    env = VisualDroneEnv(render_mode=None)
    
    # MLOps Check: Validate that our custom environment follows Gymnasium rules
    check_env(env, warn=True)
    
    # MLOps: Define log directories
    log_dir = "./experiments/logs/tensorboard/PPO_SAR_Grid_1/" 
    model_dir = "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ==========================================
    # MLflow Setup & Configuration
    # ==========================================
    mlflow.set_tracking_uri("sqlite:///mlflow.db") # Saves data locally
    mlflow.set_experiment("Drone_Search_And_Rescue") # Creates/Connects to your project workspace

    # Extract hyperparameters into variables so we can log them cleanly
    train_steps = 2_000_000
    learning_rate = 0.0003
    ent_coef = 0.05
    batch_size = 128

    # Start the MLflow tracking run
    with mlflow.start_run() as run:
        print(f"Started MLflow Run ID: {run.info.run_id}")
        
        # 1. Log the parameters so you have a permanent record of this experiment's settings
        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("total_timesteps", train_steps)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("ent_coef", ent_coef)
        mlflow.log_param("batch_size", batch_size)

        print("Building PPO Agent (policy_sar_v1)...")
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            learning_rate=learning_rate,      
            ent_coef=ent_coef,             
            batch_size=batch_size,            
        )

        # Attach your existing custom logger (outputs to stdout, csv, and tensorboard)
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        print(f"Commencing Training ({train_steps:,} timesteps)... This may take a while!")
        model.learn(total_timesteps=train_steps) 

        # Save the model locally first
        print("Training Complete. Saving model...")
        model_filename = f"{model_dir}/policy_sar_ppo.zip"
        model.save(f"{model_dir}/policy_sar_ppo")
        print(f"Model saved successfully to {model_filename}")

        # 2. Upload the saved model directly into MLflow's artifact registry
        mlflow.log_artifact(model_filename)
        print("Model binary successfully logged to MLflow as an artifact!")

    env.close()

if __name__ == "__main__":
    main()