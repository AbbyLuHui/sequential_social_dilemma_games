import torch
torch.set_default_dtype(torch.float64)
import pyro
import pyro.distributions as dist
from pyro.distributions import Categorical, Empirical, Dirichlet
from social_dilemmas.search_inference import factor, HashingMarginal, memoize, Search
from social_dilemmas.envs.agent import NormAgent


LETTER_TO_NUMBER = {'G':0, 'R':1, 'B':2}
NORM_VIEW_SIZE=7
REWARD_PRIOR = {0: torch.tensor([0., 0., 0.]),
                1: torch.tensor([0., 0., 0.]),
                2: torch.tensor([0., 0., 0.])}

class ExplorerAgent(NormAgent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, norm,reward):
        super().__init__(agent_id, start_pos, start_orientation, grid, norm, reward)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        self.reward_count = REWARD_PRIOR

    def return_reward_prior(self):
        rew_prior = {}
        for reward_index in range(len(self.reward)):
            rew_prior[reward_index] = pyro.sample("reward-tensor-"+str(reward_index),
                                                                 dist.Dirichlet(self.reward_count[reward_index]))
        print("TENSOR PRIOR: ", rew_prior)
        return rew_prior


    def consume(self, char):
        if char in self.reward:
            self.reward_count[LETTER_TO_NUMBER[char]][self.reward[char]] += 1
            print("REWARD COUNT: ", self.reward_count)
            return ' '
        else:
            return char

    def policy(self):
        apple_locs = {}
        for row_elem in range(self.grid.shape[0]):
            for column_elem in range(self.grid.shape[1]):
                item = self.grid[row_elem][column_elem]
                if item in self.norm:
                    apple_locs[(row_elem, column_elem)]=item
        goal = self.find_final_goal(self.pos[0], self.pos[1], apple_locs)
        return super().determine_action(goal[0], goal[1])

    def find_final_goal(self, x, y, obs):
        goal = [float('inf'), float('inf')]
        for item in obs:
            item_cost = abs(x - item[0]) + abs(y - item[1])
            goal_cost = abs(x - goal[0]) + abs(y - goal[1])
            if item_cost < goal_cost:
                goal[0] = item[0]
                goal[1] = item[1]
        return goal
