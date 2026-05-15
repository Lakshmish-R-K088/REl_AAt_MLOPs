import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class VisualDroneEnv(gym.Env):
    """2D Grid-World Environment with Battery Constraint"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.grid_size = 10
        self.window_size = 500  
        self.max_battery = 30  # Hard time limit to force efficiency
        
        # Action Space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: Drone (X, Y) and Battery Level
        self.observation_space = spaces.Dict({
            "drone_pos": spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            "battery": spaces.Box(low=0, high=self.max_battery, shape=(1,), dtype=np.int32)
        })

        # The Infrastructure Map (0 = Path, 1 = Wall)
        self.grid = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0] 
        ])
        
        self.start_pos = np.array([0, 0], dtype=np.int32)   # Top-Left
        self.target_pos = np.array([9, 9], dtype=np.int32)  # Bottom-Right
        
        # Pygame setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = self.start_pos.copy()
        self.battery = self.max_battery
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # 1. Calculate intended movement
        new_pos = self.drone_pos.copy()
        if action == 0: new_pos[1] -= 1   # Up
        elif action == 1: new_pos[1] += 1 # Down
        elif action == 2: new_pos[0] -= 1 # Left
        elif action == 3: new_pos[0] += 1 # Right

        # 2. Drain Battery
        self.battery -= 1
        
        reward = -1 # -1 point for every step taken (efficiency penalty)
        terminated = False
        
        # 3. Check for boundary crashes
        if np.any(new_pos < 0) or np.any(new_pos >= self.grid_size):
            reward = -100
            terminated = True
            
        # 4. Check for Wall crashes
        elif self.grid[new_pos[1], new_pos[0]] == 1: # Note: numpy arrays are accessed [row, col] -> [y, x]
            reward = -100
            terminated = True
            
        # 5. Check Battery Death
        elif self.battery <= 0:
            reward = -100
            terminated = True
            self.drone_pos = new_pos # Update position before dying
            
        # 6. Valid Move & Win Condition Check
        else:
            self.drone_pos = new_pos
            if np.array_equal(self.drone_pos, self.target_pos):
                reward = 100 # +100 for reaching the target!
                terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 50)) # Extra 50px for battery UI
            pygame.display.set_caption("2D Drone Pathfinding")
            self.font = pygame.font.SysFont("Arial", 24)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size + 50))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.grid_size)

        # Draw the Grid (Walls = Gray, Target = Green)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] == 1: # Draw Wall
                    pygame.draw.rect(canvas, (100, 100, 100), pygame.Rect(pix_square_size * x, pix_square_size * y, pix_square_size, pix_square_size))
                elif x == self.target_pos[0] and y == self.target_pos[1]: # Draw Target
                    pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect(pix_square_size * x, pix_square_size * y, pix_square_size, pix_square_size))

        # Draw Drone (Blue)
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(pix_square_size * self.drone_pos[0], pix_square_size * self.drone_pos[1], pix_square_size, pix_square_size)
        )

        # Draw Gridlines
        for i in range(self.grid_size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (0, pix_square_size * i), (self.window_size, pix_square_size * i), width=1)
            pygame.draw.line(canvas, (200, 200, 200), (pix_square_size * i, 0), (pix_square_size * i, self.window_size), width=1)

        # Draw Battery UI
        pygame.draw.rect(canvas, (30, 30, 30), pygame.Rect(0, self.window_size, self.window_size, 50))
        battery_text = self.font.render(f"Battery Remaining: {self.battery} / {self.max_battery}", True, (255, 255, 255))
        canvas.blit(battery_text, (20, self.window_size + 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _get_obs(self):
        return {"drone_pos": self.drone_pos.astype(np.int32), "battery": np.array([self.battery], dtype=np.int32)}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()