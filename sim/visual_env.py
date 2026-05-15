import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class VisualDroneEnv(gym.Env):
    """20x20 Grid-World POMDP Search & Rescue Environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.grid_size = 20
        self.window_size = 600  # Increased for larger grid visibility
        self.max_battery = 200  # More battery needed for a 20x20 search
        
        # NEW: Search & Rescue Parameters
        self.view_radius = 2    # Drone can now see a 5x5 grid around itself
        self.num_targets = 5    # How many people/resources to find
        
        # Action Space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: Drone (X,Y), Battery, and Memory Map
        # Explored Map: -1 (Fog), 0 (Path), 1 (Wall), 2 (Resource)
        self.observation_space = spaces.Dict({
            "drone_pos": spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            "battery": spaces.Box(low=0, high=self.max_battery, shape=(1,), dtype=np.int32),
            "explored_map": spaces.Box(low=-1, high=2, shape=(self.grid_size, self.grid_size), dtype=np.int32)
        })

        # 20x20 Infrastructure Map (0 = Path, 1 = Wall)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Procedurally build a "City Block" style map with obstacles
        for i in range(2, 18, 4):
            for j in range(2, 18, 4):
                self.grid[i:i+2, j:j+2] = 1  # Create 2x2 buildings/walls
                
        # Add some random scattered walls to make it messy
        np.random.seed(42) # Keep walls deterministic for consistent training
        for _ in range(20):
            rx, ry = np.random.randint(0, 20, 2)
            self.grid[ry, rx] = 1

        # Clear start position just in case
        self.grid[0, 0] = 0 
        self.start_pos = np.array([0, 0], dtype=np.int32)   
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

    def _update_explored_map(self):
        """Drone scans adjacent grids and saves them to memory"""
        x, y = self.drone_pos
        
        y_min = max(0, y - self.view_radius)
        y_max = min(self.grid_size, y + self.view_radius + 1)
        x_min = max(0, x - self.view_radius)
        x_max = min(self.grid_size, x + self.view_radius + 1)
        
        # Reveal grid values AND resources
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                if self.resource_map[i, j] == 1:
                    self.explored_map[i, j] = 2  # 2 = Resource spotted!
                else:
                    self.explored_map[i, j] = self.grid[i, j]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = self.start_pos.copy()
        self.battery = self.max_battery
        self.resources_collected = 0
        
        # DYNAMIC SPAWNING: Randomly place resources on empty cells
        self.resource_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        empty_cells = np.argwhere(self.grid == 0)
        
        # Remove start pos from valid spawns
        empty_cells = [cell for cell in empty_cells if not (cell[0] == 0 and cell[1] == 0)]
        
        # Pick random locations for resources each episode
        chosen_indices = np.random.choice(len(empty_cells), self.num_targets, replace=False)
        for idx in chosen_indices:
            y, x = empty_cells[idx]
            self.resource_map[y, x] = 1

        # Reset memory map
        self.explored_map = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        self._update_explored_map()
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        new_pos = self.drone_pos.copy()
        if action == 0: new_pos[1] -= 1   
        elif action == 1: new_pos[1] += 1 
        elif action == 2: new_pos[0] -= 1 
        elif action == 3: new_pos[0] += 1 

        self.battery -= 1
        reward = -0.5  # Slight penalty to encourage speed
        terminated = False
        
        # Crashes into boundary
        if np.any(new_pos < 0) or np.any(new_pos >= self.grid_size):
            reward = -50
            terminated = True
            
        # Crashes into wall
        elif self.grid[new_pos[1], new_pos[0]] == 1: 
            reward = -50
            terminated = True
            
        # Battery Death
        elif self.battery <= 0:
            reward = -50
            terminated = True
            self.drone_pos = new_pos 
            self._update_explored_map()
            
        # Valid Move
        else:
            self.drone_pos = new_pos
            
            # Check for Resource Collection
            if self.resource_map[new_pos[1], new_pos[0]] == 1:
                self.resource_map[new_pos[1], new_pos[0]] = 0 # Consume the resource
                self.resources_collected += 1
                reward = 50 # Big reward for finding a target!
                
            self._update_explored_map() # Scan the new area
            
            # Win Condition: Found everyone!
            if self.resources_collected >= self.num_targets:
                reward += 100
                terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 60))
            pygame.display.set_caption("SAR Drone - Radar Exploration")
            self.font = pygame.font.SysFont("Arial", 22)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size + 60))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.grid_size)

        # Draw the Explored Map
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.explored_map[y, x] == -1: # Fog of War
                    pygame.draw.rect(canvas, (15, 15, 15), pygame.Rect(pix_square_size * x, pix_square_size * y, pix_square_size, pix_square_size))
                elif self.explored_map[y, x] == 1: # Wall
                    pygame.draw.rect(canvas, (100, 100, 100), pygame.Rect(pix_square_size * x, pix_square_size * y, pix_square_size, pix_square_size))
                elif self.explored_map[y, x] == 2: # Resource / Person
                    pygame.draw.rect(canvas, (255, 165, 0), pygame.Rect(pix_square_size * x, pix_square_size * y, pix_square_size, pix_square_size))

        # Draw Drone
        pygame.draw.rect(
            canvas,
            (0, 150, 255),
            pygame.Rect(pix_square_size * self.drone_pos[0], pix_square_size * self.drone_pos[1], pix_square_size, pix_square_size)
        )

        # Draw Gridlines
        for i in range(self.grid_size + 1):
            pygame.draw.line(canvas, (40, 40, 40), (0, pix_square_size * i), (self.window_size, pix_square_size * i), width=1)
            pygame.draw.line(canvas, (40, 40, 40), (pix_square_size * i, 0), (pix_square_size * i, self.window_size), width=1)

        # Draw UI
        pygame.draw.rect(canvas, (30, 30, 30), pygame.Rect(0, self.window_size, self.window_size, 60))
        ui_text = self.font.render(f"Battery: {self.battery}/{self.max_battery}   |   Rescued: {self.resources_collected}/{self.num_targets}", True, (255, 255, 255))
        canvas.blit(ui_text, (20, self.window_size + 15))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _get_obs(self):
        return {
            "drone_pos": self.drone_pos.astype(np.int32), 
            "battery": np.array([self.battery], dtype=np.int32),
            "explored_map": self.explored_map.astype(np.int32)
        }

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()