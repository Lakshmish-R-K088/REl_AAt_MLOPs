import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneInspectionEnv(gym.Env):
    """
    Custom Environment for an Autonomous Infrastructure Patrol Drone.
    Targeting SDG 9: Industry, Innovation and Infrastructure.
    """
    def __init__(self):
        super(DroneInspectionEnv, self).__init__()
        
        # Grid dimensions (e.g., a 10x10x10 area around a bridge)
        self.grid_size = 10
        self.max_battery = 200
        
        # Action Space: 6 Discrete movements 
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Forward, 5: Backward
        self.action_space = spaces.Discrete(6)
        
        # Observation Space: Drone coordinates, battery life, and scanned grid 
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=self.grid_size-1, shape=(3,), dtype=np.int32),
            "battery": spaces.Box(low=0, high=self.max_battery, shape=(1,), dtype=np.int32),
            "scanned_grid": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, self.grid_size), dtype=np.int32)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize drone position at the origin
        self.agent_pos = np.array([0, 0, 0], dtype=np.int32)
        
        # Fully charge the drone
        self.battery = np.array([self.max_battery], dtype=np.int32)
        
        # Initialize an empty grid to track what has been scanned (0 = unscanned, 1 = scanned)
        self.scanned_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.int32)
        
        # Mark initial position as scanned
        self.scanned_grid[tuple(self.agent_pos)] = 1
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self):
        return {
            "agent_pos": self.agent_pos,
            "battery": self.battery,
            "scanned_grid": self.scanned_grid
        }

    def step(self, action):
        # 1. Apply Action (Update Coordinates)
        if action == 0: self.agent_pos[2] += 1 # Up
        elif action == 1: self.agent_pos[2] -= 1 # Down
        elif action == 2: self.agent_pos[0] -= 1 # Left
        elif action == 3: self.agent_pos[0] += 1 # Right
        elif action == 4: self.agent_pos[1] += 1 # Forward
        elif action == 5: self.agent_pos[1] -= 1 # Backward
        
        # Decrease battery
        self.battery -= 1
        
        reward = 0
        terminated = False
        truncated = False
        
        # 2. Check for Crashes or Out of Bounds
        if not (0 <= self.agent_pos[0] < self.grid_size and 
                0 <= self.agent_pos[1] < self.grid_size and 
                0 <= self.agent_pos[2] < self.grid_size):
            reward = -100 # Terminal penalty for crashing/out of bounds [cite: 339]
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}
            
        # 3. Check Battery Status
        if self.battery[0] <= 0:
            reward = -100 # Terminal penalty for running out of battery [cite: 339]
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        # 4. Calculate Scan Reward
        pos_tuple = tuple(self.agent_pos)
        if self.scanned_grid[pos_tuple] == 0:
            reward = 10  # Reward for scanning a new section [cite: 337]
            self.scanned_grid[pos_tuple] = 1
        else:
            reward = -1  # Penalty for moving to an already scanned area (efficiency check) [cite: 338]

        # Check if 80% of the grid is scanned (Success Condition)
        total_scanned = np.sum(self.scanned_grid)
        if total_scanned >= 0.8 * (self.grid_size ** 3):
            reward += 100 # Massive bonus for completing the mission
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # Simple print statement for debugging
        print(f"Drone Pos: {self.agent_pos}, Battery: {self.battery[0]}, Scanned: {np.sum(self.scanned_grid)}")