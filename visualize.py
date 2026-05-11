import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from sim.env import DroneInspectionEnv

def main():
    print("Loading environment and real trained model...")
    env = DroneInspectionEnv()
    
    # Loading your ACTUAL trained weights
    model = DQN.load("experiments/models/policy_v1")
    
    obs, info = env.reset()
    
    # Initialize the 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    terminated = False
    truncated = False
    
    # Track the path to draw a trail
    path_x, path_y, path_z = [], [], []
    
    plt.ion() # Turn on interactive plotting mode
    plt.show()
    
    while not terminated and not truncated:
        # The real model predicting the next action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        pos = obs["agent_pos"]
        path_x.append(pos[0])
        path_y.append(pos[1])
        path_z.append(pos[2])
        
        # Clear the graph and redraw the new state
        ax.clear()
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_zlim(0, 9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Real Model Navigation - Battery: {obs['battery'][0]}")
        
        # Draw the trail
        ax.plot(path_x, path_y, path_z, color='blue', marker='.', linestyle='dashed', alpha=0.5)
        
        # Draw the drone
        ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, label='Drone')
        
        plt.draw()
        plt.pause(0.2) # Pause for 200ms to create animation effect
        
    plt.ioff()
    print("Mission Ended. Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    main()