import numpy as np
import tkinter as tk
import time

from HyperParameters import *

AGENT_COLORS = ["red", "blue"]
VISITED_COLOR = "#ccffcc"
OBSTACLE_COLOR = "black"

class Explore:
    def __init__(self, render_mode=False):
        self.grid_size = GRID_SIZE
        # self.max_steps = steps_per_episode
        self.current_step = 0
        self.agent_positions = [[0, 0], [GRID_SIZE - 1, GRID_SIZE - 1]]
        self.visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.obstacles = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.render_mode = render_mode
        if self.render_mode:
            self._init_render()

    def _init_render(self):
        self.window = tk.Tk()
        self.window.title("Exploration")
        self.canvas = tk.Canvas(self.window, bg='white',
                                height=GRID_SIZE * UNIT,
                                width =GRID_SIZE * UNIT)
        self.canvas.pack()
        self._draw_base()

    def _draw_base(self):
        self.canvas.delete("all")
        for c in range(0, GRID_SIZE * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, GRID_SIZE * UNIT)
        for r in range(0, GRID_SIZE * UNIT, UNIT):
            self.canvas.create_line(0, r, GRID_SIZE * UNIT, r)

    def _draw_env(self):
        self._draw_base()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.obstacles[r, c]:
                    x0, y0 = c * UNIT + 1, r * UNIT + 1
                    x1, y1 = (c + 1) * UNIT - 1, (r + 1) * UNIT - 1
                    self.canvas.create_rectangle(x0, y0, x1, y1,
                                                 fill=OBSTACLE_COLOR, outline="")

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.visited[r, c] and not self.obstacles[r, c]:
                    x0, y0 = c * UNIT + 1, r * UNIT + 1
                    x1, y1 = (c + 1) * UNIT - 1, (r + 1) * UNIT - 1
                    self.canvas.create_rectangle(x0, y0, x1, y1,
                                                 fill=VISITED_COLOR, outline="")

        for i, (r, c) in enumerate(self.agent_positions):
            x0, y0 = c * UNIT + 10, r * UNIT + 10
            x1, y1 = x0 + 20, y0 + 20
            self.canvas.create_oval(x0, y0, x1, y1, fill=AGENT_COLORS[i])
        self.window.update()
        time.sleep(0.1)

    def _is_valid_move(self, x, y):
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and self.obstacles[x, y] == 0

    def reset(self):
        self.current_step = 0
        self.agent_positions = [[0, 0], [GRID_SIZE - 1, GRID_SIZE - 1]]
        self.visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.obstacles = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        # 固定障碍物
        # for i in range(2, 5):
        #     self.obstacles[i, 3] = 1
        self.obstacles[1, 1] = 1
        # self.obstacles[2, 2] = 1
# 
        for pos in self.agent_positions:
            self.visited[pos[0], pos[1]] = 1

        if self.render_mode:
            self._draw_env()

        return self._get_observation()

    def step(self, actions):
        self.current_step += 1
        total_reward = 0
        total_reward_sparse = 0
        new_positions = self.agent_positions.copy()
        done=False

        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            new_x, new_y = x, y

            if   action == 0 and new_y > 0           : new_y -= 1  # up
            elif action == 1 and new_y < GRID_SIZE-1 : new_y += 1  # down
            elif action == 2 and new_x > 0           : new_x -= 1  # left
            elif action == 3 and new_x < GRID_SIZE-1 : new_x += 1  # right

            # Check if move is valid (within bounds and not into obstacle)
            if not self._is_valid_move(new_x, new_y):
                total_reward -= 1  # illegal move
                continue

            # Check for collision with other agents
            if [new_x, new_y] == new_positions[1-i]:
                total_reward -= 1  # collision penalty
                continue

            # Check if the agent actually moved
            if (new_x, new_y) == (x, y):
                total_reward -= 1  # lazy penalty
            else:
                if self.visited[new_x, new_y] == 0:
                    total_reward += 2  # unexplored cell
                else:
                    total_reward -= 0.5  # explored cell

                new_positions[i] = [new_x, new_y]
                self.visited[new_x, new_y] = 1

        self.agent_positions = new_positions

        # Check if grid is fully explored
        total_accessible = (self.obstacles == 0).sum()
        explored = self.visited[self.obstacles == 0].sum()
        if explored == total_accessible:
            total_reward += 20
            total_reward_sparse += 20
            # self.reset()
            done=True

        if self.render_mode:
            self._draw_env()

        obs = self._get_observation()



        return obs, total_reward, total_reward_sparse, done, {}


    def dummy_step(self, actions):
        temp_positions = [pos.copy() for pos in self.agent_positions]
        temp_visited = self.visited.copy()

        for i, action in enumerate(actions):
            x, y = temp_positions[i]
            new_x, new_y = x, y

            if   action == 0 and new_y > 0           : new_y -= 1  # up
            elif action == 1 and new_y < GRID_SIZE-1 : new_y += 1  # down
            elif action == 2 and new_x > 0           : new_x -= 1  # left
            elif action == 3 and new_x < GRID_SIZE-1 : new_x += 1  # right

            if self._is_valid_move(new_x, new_y):
                temp_positions[i] = [new_x, new_y]
                temp_visited[new_x, new_y] = 1

        flat_visited = temp_visited.flatten()
        obs = np.array(temp_positions[0] + temp_positions[1] + flat_visited.tolist(), dtype=np.float32)
        return obs

    def _get_observation(self):
        flat_visited = self.visited.flatten()
        obs = np.array(self.agent_positions[0] + self.agent_positions[1] + flat_visited.tolist(), dtype=np.float32)
        return obs

    def render(self):
        if self.render_mode:
            self._draw_env()
