import time
from stable_baselines3 import DQN
from sim.env import DroneInspectionEnv

def main():
    print("Loading the Drone Environment...")
    env = DroneInspectionEnv()
    
    print("Loading trained policy_v1...")
    # Load the trained brain we just saved
    model = DQN.load("experiments/models/policy_v1")
    
    # Get the initial state
    obs, info = env.reset()
    
    print("\n--- STARTING INSPECTION MISSION ---\n")
    terminated = False
    truncated = False
    total_reward = 0
    step = 0
    
    while not terminated and not truncated:
        # The agent looks at the observation and decides the best action
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Visualize what is happening
        print(f"Step {step}: Action Taken: {action}")
        env.render() # This prints the coordinates, battery, and scan progress
        print(f"Current Reward: {total_reward}\n")
        
        # Slow it down so we can actually watch it!
        time.sleep(0.5)
        
    print("--- MISSION COMPLETE ---")
    print(f"Final Score: {total_reward}")

if __name__ == "__main__":
    main()