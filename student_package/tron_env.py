import numpy as np
import pygame
import random

# --- Constants ---
GRID_SIZE = 20  # Map size 20x20
CELL_SIZE = 20  # Cell size for rendering
FPS = 15        # Game speed

# Grid States
EMPTY = 0
WALL = 1
P1_HEAD = 2
P1_TRAIL = 3
P2_HEAD = 4
P2_TRAIL = 5

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Colors (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 255)
RED = (255, 50, 50)
GRAY = (100, 100, 100)

class BlindTronEnv:
    def __init__(self, grid_size=GRID_SIZE, render_mode=False):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.p1_pos = [0, 0]
        self.p2_pos = [0, 0]
        self.p1_dir = -1 # No initial direction
        self.p2_dir = -1
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2 # Prevent infinite loops

        if self.render_mode:
            pygame.init()
            self.screen_width = grid_size * CELL_SIZE
            self.screen_height = grid_size * CELL_SIZE
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Blindfolded Tron (RNN Testbed)")
            self.clock = pygame.time.Clock()

    def reset(self):
        """Reset game state"""
        self.grid.fill(EMPTY)
        self.steps = 0
        
        # Random spawn points, keep distance
        while True:
            p1_r, p1_c = random.randint(2, self.grid_size-3), random.randint(2, self.grid_size-3)
            p2_r, p2_c = random.randint(2, self.grid_size-3), random.randint(2, self.grid_size-3)
            if abs(p1_r - p2_r) + abs(p1_c - p2_c) > self.grid_size / 2:
                break
                
        self.p1_pos = [p1_r, p1_c]
        self.p2_pos = [p2_r, p2_c]
        self.p1_dir = -1
        self.p2_dir = -1
        
        self.grid[p1_r, p1_c] = P1_HEAD
        self.grid[p2_r, p2_c] = P2_HEAD
        
        return self._get_obs(1), self._get_obs(2)

    def _is_valid_move(self, current_dir, proposed_action):
        # Prevent immediate 180-degree turns
        if current_dir == UP and proposed_action == DOWN: return False
        if current_dir == DOWN and proposed_action == UP: return False
        if current_dir == LEFT and proposed_action == RIGHT: return False
        if current_dir == RIGHT and proposed_action == LEFT: return False
        return True

    def _get_new_pos(self, pos, action):
        r, c = pos
        if action == UP: r -= 1
        elif action == DOWN: r += 1
        elif action == LEFT: c -= 1
        elif action == RIGHT: c += 1
        return [r, c]

    def _check_collision(self, pos):
        r, c = pos
        # Wall boundaries
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        # Collision with any non-empty object (including self and opponent trails/heads)
        if self.grid[r, c] != EMPTY:
            return True
        return False

    def _get_obs(self, player_id):
        """
        Key part: Generate observation (Ray-casting / Lidar)
        Returns a vector of length 10:
        [N_dist, NE_dist, E_dist, SE_dist, S_dist, SW_dist, W_dist, NW_dist, Norm_X, Norm_Y]
        Distance is normalized to [0, 1], 0 means right in front, 1 means far away.
        """
        pos = self.p1_pos if player_id == 1 else self.p2_pos
        obs = []
        
        # 8-direction ray casting (r_step, c_step)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        for dr, dc in directions:
            dist = 0
            r, c = pos
            # Cast ray until hitting boundary or obstacle
            while True:
                r += dr
                c += dc
                dist += 1
                if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size or self.grid[r, c] != EMPTY:
                    break
            # Normalize distance. Max view distance is grid_size
            normalized_dist = min(dist, self.grid_size) / self.grid_size
            obs.append(normalized_dist)

        # Add normalized self coordinates
        obs.append(pos[0] / self.grid_size)
        obs.append(pos[1] / self.grid_size)
        
        return np.array(obs, dtype=np.float32)

    def step(self, action1, action2):
        """Execute one step"""
        self.steps += 1
        done = False
        winner = 0 # 0: Draw, 1: P1 Wins, 2: P2 Wins
        
        # Handle invalid 180-degree turns: if invalid, keep current direction (if initially invalid, random direction)
        if not self._is_valid_move(self.p1_dir, action1): action1 = self.p1_dir if self.p1_dir != -1 else random.choice([0,1,2,3])
        if not self._is_valid_move(self.p2_dir, action2): action2 = self.p2_dir if self.p2_dir != -1 else random.choice([0,1,2,3])

        self.p1_dir = action1
        self.p2_dir = action2

        # Calculate new positions
        p1_new = self._get_new_pos(self.p1_pos, action1)
        p2_new = self._get_new_pos(self.p2_pos, action2)
        
        # Check collisions
        p1_crashed = self._check_collision(p1_new)
        p2_crashed = self._check_collision(p2_new)
        # Special case: head-on collision
        head_on_collision = (p1_new == p2_new)

        if (p1_crashed and p2_crashed) or head_on_collision or self.steps >= self.max_steps:
            done = True
            winner = 0 # Draw
        elif p1_crashed:
            done = True
            winner = 2 # P2 wins
        elif p2_crashed:
            done = True
            winner = 1 # P1 wins
        else:
            # Valid move, update grid
            self.grid[self.p1_pos[0], self.p1_pos[1]] = P1_TRAIL
            self.grid[self.p2_pos[0], self.p2_pos[1]] = P2_TRAIL
            self.p1_pos = p1_new
            self.p2_pos = p2_new
            self.grid[self.p1_pos[0], self.p1_pos[1]] = P1_HEAD
            self.grid[self.p2_pos[0], self.p2_pos[1]] = P2_HEAD

        obs1 = self._get_obs(1)
        obs2 = self._get_obs(2)
        
        if self.render_mode: self.render()
        
        return obs1, obs2, done, winner

    def render(self):
        if not self.render_mode: return
        self.screen.fill(BLACK)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_val = self.grid[r, c]
                color = BLACK
                if cell_val == WALL: color = GRAY
                elif cell_val == P1_HEAD: color = BLUE
                elif cell_val == P1_TRAIL: color = (0, 0, 150) # Darker Blue
                elif cell_val == P2_HEAD: color = RED
                elif cell_val == P2_TRAIL: color = (150, 0, 0) # Darker Red
                
                if cell_val != EMPTY:
                    pygame.draw.rect(self.screen, color, 
                                     (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw grid lines (optional)
        # for i in range(self.grid_size + 1):
        #     pygame.draw.line(self.screen, GRAY, (0, i*CELL_SIZE), (self.screen_width, i*CELL_SIZE))
        #     pygame.draw.line(self.screen, GRAY, (i*CELL_SIZE, 0), (i*CELL_SIZE, self.screen_height))
            
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()
