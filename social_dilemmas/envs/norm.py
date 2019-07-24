import numpy as np
import random
import torch
import pyro
import pyro.distributions as dist
from social_dilemmas.constants import NORM_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS
from social_dilemmas.envs.agent import NormAgent
from social_dilemmas.explorer_dirichlet import ExplorerAgent

REWARD_PRIOR = {0: [0., 0.2, 0.8],
                1: [0.2, 0.8, 0.],
                2: [0.2, 0.8, 0.]} #norm: [reward=0, reward=1, reward=2]
LETTER_TO_NUMBER = {'G': 0,
                    'R': 1,
                    'B': 2}


class NormEnv(MapEnv):

    def __init__(self, ascii_map=NORM_MAP, num_agents=1, render=False, norm=dict(), reward=dict()):
        super().__init__(ascii_map, num_agents, render, norm, reward)
        self.pos_dict={'G':[], 'R':[], 'B':[]}
        self.respawn_prob={'G':0.005,'R':0.005,'B':0.005}
        # make a dict of the potential apple spawn points
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] in self.pos_dict:
                    self.pos_dict[self.base_map[row, col]].append([row,col])

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        # FIXME(ev) this is an information leak
        agents = list(self.agents.values())
        return agents[0].observation_space


    def custom_map_update(self):
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            reward_dict = {reward: int(pyro.sample("reward", dist.Categorical(torch.tensor(REWARD_PRIOR[LETTER_TO_NUMBER[reward]]))))\
                           for reward in self.norm}
            agent = NormAgent(agent_id, spawn_point, rotation, map_with_agents, self.norm, reward_dict)
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        for item in self.pos_dict:
            for i in range(len(self.pos_dict[item])):
                row, col = self.pos_dict[item][i]
                if self.world_map[row, col] not in self.pos_dict and [row, col] not in self.agent_pos:
                    rand_num = np.random.rand(1)[0]
                    if rand_num < self.respawn_prob[item]:
                        spawn_points.append((row, col, item))
        return spawn_points


class ExploreEnv(NormEnv):
    def __init__(self, ascii_map=NORM_MAP, num_agents=1, render=False, norm=dict(), reward=dict(), inferred_reward=dict()):
        super().__init__(ascii_map, num_agents, render, norm, reward)
        self.inferred_reward=inferred_reward

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()
        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            reward_dict = {reward: int(pyro.sample("reward", dist.Categorical(torch.tensor(REWARD_PRIOR[LETTER_TO_NUMBER[reward]]))))\
                           for reward in self.norm}
            print("Real Reward Explorer: ", reward_dict)
            agent = ExplorerAgent(agent_id, spawn_point, rotation, map_with_agents, self.norm, reward_dict)
            self.agents[agent_id] = agent