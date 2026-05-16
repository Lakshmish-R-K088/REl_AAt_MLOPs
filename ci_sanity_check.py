# ci_sanity_check.py
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from sim.visual_env import VisualDroneEnv

def test_pipeline():
    print("🤖 Starting CI Sanity Test...")
    
    # 1. Initialize environment headlessly (render_mode=None)
    print("Step 1: Initializing 20x20 SAR POMDP Environment...")
    env = VisualDroneEnv(render_mode=None)
    
    # 2. Run the Stable-Baselines3 MLOps environment validator
    print("Step 2: Validating Gymnasium compliance via SB3 check_env...")
    check_env(env, warn=True)
    
    # 3. Perform a quick 5-step test simulation to ensure transitions work
    print("Step 3: Running execution smoke test...")
    obs, info = env.reset()
    for step in range(5):
        action = env.action_space.sample()  # random actions
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            
    env.close()
    print("✅ CI Pipeline Check Complete! Environment is perfectly stable.")

if __name__ == "__main__":
    test_pipeline()