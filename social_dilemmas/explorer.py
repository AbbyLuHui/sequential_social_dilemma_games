import torch
torch.set_default_dtype(torch.float64)
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate, infer_discrete
from pyro.distributions import Categorical, Empirical
from social_dilemmas.search_inference import factor, HashingMarginal, memoize, Search
from social_dilemmas.envs.agent import NormAgent

def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

LETTER_TO_NUMBER = {'G':0, 'R':1, 'B':2}
NORM_VIEW_SIZE=7
REWARD_PRIOR = {0: [1/3, 1/3, 1/3],
                1: [1/3, 1/3, 1/3],
                2: [1/3, 1/3, 1/3]}

class ExplorerAgent(NormAgent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, norm,reward):
        super().__init__(agent_id, start_pos, start_orientation, grid, norm, reward)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        self.inferred_reward = REWARD_PRIOR

    def setup_reward_prior(self):
        rew_prior = {}
        for reward_index in range(len(self.reward)):
            rew_prior["reward-"+str(reward_index)] = float(pyro.sample("reward-"+str(reward_index),
                                                                 dist.Categorical(probs=torch.FloatTensor(self.inferred_reward[reward_index]))))
        return rew_prior

    @Marginal
    def model(self, data):
        rew_prior = self.setup_reward_prior()
        fruit_util = {}
        for reward_index in range(len(rew_prior)):
            if reward_index == data[0]:
                fruit_utility = torch.zeros(len(self.reward))
                for i in range(len(fruit_utility)):
                    if rew_prior["reward-"+str(reward_index)] == i:
                        fruit_utility[i]=1
                    else:
                        fruit_utility[i]=0
                fruit_util['util-'+str(reward_index)] = pyro.sample("util-"+str(reward_index),dist.Categorical(probs=fruit_utility),obs=torch.tensor(data[1]))
        rew_prior = {i:int(rew_prior[i]) for i in rew_prior}
        return tuple(rew_prior.values())

    def consume(self, char):
        if char in self.reward:
            self.reward_this_turn += self.reward[char]
            result=(LETTER_TO_NUMBER[char], self.reward[char])
            print(result)
            print(self.inferred_reward)
            support = self.model(result).enumerate_support()
            data = [self.model(result).log_prob(s).exp().item() for s in support]

            #update self.inferred reward
            inferred_reward={}
            for reward_index in range(len(self.reward)):
                reward_value_list=[0*i for i in range(len(self.inferred_reward[reward_index]))]
                for enum_index in range(len(support)):
                    reward_value_list[support[enum_index][reward_index]] += data[enum_index]
                inferred_reward[reward_index] = reward_value_list
            #print("Inferred reward: ", inferred_reward)
            self.inferred_reward=inferred_reward.copy()
            print("Inferred reward prior update: ", self.inferred_reward)
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




