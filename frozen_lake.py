from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from mapping import OceanMapper,Coordinate
from gymnasium.envs.toy_text.utils import categorical_sample
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False
def generate_random_map(size: int = 100, p: float = 0.8) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]

class Shipping(FrozenLakeEnv):
    def __init__(
            self,
        time_of_day=0,
        fuel_model=None,
        speed_model=None,
        render_mode: Optional[str] = 'human',
        desc=None,
        map_name=None,
        is_slippery=True
    ):
        super().__init__( render_mode,
        desc,
        map_name,
        is_slippery)
        self.fuel_model=fuel_model
        self.fuel_capacity=1
        self.time_of_day=time_of_day
        self.speed_model=speed_model
        self.time=0
        self.s=0
    def fuel_consumed(self,coord1,coord2):
            v=self.speed(coord1)
            if self.fuel_model:
                self.fuel_capacity-= self.fuel_model.predict([coord1,coord2,v,self.time])
            return 0
    def speed(self,coord1):
            if self.speed_model:
                return self.speed_model.predict([coord1,self.time])
            return 0.1       
    def time_update(self,coord1,coord2):
            # self.time+=self.distance(coord1,coord2)/self.speed(coord1)
            self.time+=1
    def distance(coord1,coord2):
            #use haversine metric
            return 0    
    def update_probability_matrix(self,row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            
            terminated = bytes(newletter) in b"GH"+bytes(bool(self.fuel_capacity))
            reward = float(newletter == b"G")+self.fuel_capacity-self.time
            
            return newstate, reward, terminated
    def step(self, a,j):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        past=self.s
        self.s = s
        self.lastaction = a
        self.fuel_consumed(past,s)
        self.time_update(past,s)
        newletter = self.desc[s//100, s%100]
        self.visit_dict[:s//100, :s%100]=np.maximum(0,self.visit_dict[:s//100, :s%100]+1)
        if self.visit_dict[s//100, s%100]<0:
         visit_reward=1000
        else:
             visit_reward=0 
        t = bytes(newletter) in b"GH"+bytes(bool(self.fuel_capacity))
        if t:
             terrmination_reward=10**7
        else:
             terrmination_reward= 0    
        r = 10**8*int(float(newletter == b"G"))+self.fuel_capacity-10*self.time-np.linalg.norm((self.visit_dict))+visit_reward-terrmination_reward
        win=newletter == b"G"
        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        trunc=False
        if j>=500:
             trunc=True
        return int(s), r, t, trunc, win
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        
        super().reset(seed=seed)
        self.visit_dict=np.zeros((len(self.desc),len(self.desc[0])))-10**6
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()

        return int(self.s), {"prob": 1}
if __name__ == '__main__':
    mapper = OceanMapper(grid_size=100)
    top_left = Coordinate(lat=25.0, lon=-85.0)  # Near Florida
    bottom_right = Coordinate(lat=19.0, lon=-75.0)  # Near Cuba
    desc1 = mapper.get_ocean_map(top_left, bottom_right)

    frozen_lake = Shipping(
        time_of_day=0,
        fuel_model=None,
        speed_model=None,
        render_mode = 'human',
        desc=desc1,
        map_name=None,
        is_slippery=True)
    frozen_lake.reset()
    print("reset")
    frozen_lake.render()