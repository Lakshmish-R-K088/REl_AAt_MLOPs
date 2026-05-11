import gymnasium as gym
from stable_baselines3 import PPO
from sim.visual_env import VisualDroneEnv
import time

def main():
    print("Loading Visual Environment...")
    # 'human' mode turns the Pygame window back on
    env = VisualDroneEnv(render_mode="human")
    
    print("Loading trained PPO policy...")
    # Point exactly to your new saved model
    model = PPO.load("models/policy_v2_ppo")

    obs, info = env.reset()
    terminated = False
    step_count = 0

    print("--- STARTING OPTIMIZED MISSION ---")
    
    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        time.sleep(0.3) # Adjust this to speed up/slow down the animation
        
    print(f"Mission Complete in {step_count} steps!")
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    main()