import torch
torch.set_default_dtype(torch.float64)
import pyro
import pyro.distributions as dist
import pyro.infer
from pyro.distributions import Categorical, Empirical



from social_dilemmas.envs.agent import NormAgent

DEPTH = 2
NORM_DICT={0: "G",
           1: "R",
           2: "B"}

REWARD_PRIOR = {0: [0.9, 0.1, 0],
                1: [0.9, 0.1, 0],
                2: [1/3, 1/3, 1/3]} #norm: [reward=0, reward=1, reward=2]

REWARD_LIST=[0,1,2]

ACTION_DICT={0: "GO UP",
             1: "GO DOWN",
             2: "GO LEFT",
             3: "GO RIGHT"}

ENV_NORM = {0:1, 1:0, 2:0}

class Observer():
    def __init__(self, grid):
        self.grid = grid
        self.env_norm = ENV_NORM
        self.agent_norm = {"G":True, "R":False, "B":False}
        self.rew_prior = {}
        #self.rew_prior = {"G":0.5, "R":0.5, "B":0.5}
        self.n_prior= {"G":1/3, "R":1/3, "B":1/3}
        self.agent_no=0
        self.reward_dict = {}
        self.update_reward_dict()
        print(self.reward_dict)

    def update_reward_dict(self):
        #create reward_dict for easy interpretation of categorical distribution result
        length = len(REWARD_LIST)
        for a in range(length):
            for b in range(length):
                for c in range(length):
                    self.reward_dict[(length**2)*a+length*b+c] = [REWARD_LIST[a], REWARD_LIST[b], REWARD_LIST[c]]

    #assume norm follows normal distribution
    """def norm_prior(self):
        n_prior = {}
        for norm in self.agent_norm:
            n_prior[norm] = pyro.sample("norm_" + norm, dist.Normal(self.n_prior[norm], 1))
        return n_prior"""

    #assume norm follows categorical distribution
    def norm_prior(self):
        #use total to normalize prior
        total=0
        norm_no=0
        a = self.n_prior
        for item in a:
            total = total + a[item]
            norm_no+=1
        prob_tensor = torch.zeros(norm_no)
        i=0
        for item in a:
            prob_tensor[i] = a[item]/total
            i+=1
        #print("prob tensor: ", prob_tensor)
        n_prior = pyro.sample("norm", dist.Categorical(prob_tensor))
        return n_prior

    def reward_prior(self):
        reward_prior = {}
        #set self.reward_prior across 27 possibility
        for k in range(self.agent_no):
            self.rew_prior['agent-%d'%k]=torch.zeros(len(REWARD_LIST)**len(NORM_DICT))
        #reward prior according to REWARD_PRIOR
        for agent_no in range(self.agent_no):
            for rew_no in range(len(self.rew_prior['agent-%d'%agent_no])):
                prior = 1
                for norm_no in NORM_DICT:
                    rew = self.reward_dict[rew_no][norm_no]
                    prior = prior * REWARD_PRIOR[norm_no][rew]
                self.rew_prior['agent-%d'%agent_no][rew_no] = prior
        #uniform reward prior
        #for k in range(self.agent_no):
        #    for i in range(len(self.rew_prior['agent-%d'%k])):
        #        self.rew_prior['agent-%d'%k][i] = 1/(len(REWARD_LIST)**len(NORM_DICT))
        for j in range(self.agent_no):
            reward_prior["agent-" + str(j)] = pyro.sample("reward_agent" + str(j), \
                                                          dist.Categorical(self.rew_prior["agent-"+str(j)]))
        #for norm in self.agent_norm:
        #    for i in range(agent_no):
        #        reward_prior[norm+str(i)] = pyro.sample("reward_" + norm + str(i), dist.Normal(self.rew_prior[norm], 0.1))
        return reward_prior

    #get locations of agents from one observation from the grid
    def get_agent_locs(self):
        agent_locs={}
        agent_no = 0
        for row_elem in range(self.grid.shape[0]):
            for col_elem in range(self.grid.shape[1]):
                if self.grid[row_elem][col_elem] in "0123456789":
                    agent_locs[int(self.grid[row_elem][col_elem])-1]=([row_elem, col_elem])
                    agent_no+=1
        self.agent_no=agent_no
        return agent_locs

    #create instances of all agent in the grid
    def agent(self):
        agent_locs= self.get_agent_locs()
        agent_list=[]
        rew = self.reward_prior()
        for i in range(self.agent_no):
            index=0
            reward={}
            for norm in self.agent_norm:
                reward[norm]= self.reward_dict[int(rew['agent-'+str(i)])][index]
                index+=1
            ag = NormAgent("agent%d" %i, agent_locs[i], 'UP', self.grid, self.agent_norm, reward)
            agent_list.append(ag)
        return agent_list

    def update_grid(self, grid):
        self.grid = grid

    #model of norm, action conditioned upon norm
    def model(self):
        n_prior = self.norm_prior()
        for norm in self.agent_norm:
            self.agent_norm[norm] = True if norm == NORM_DICT[int(n_prior)] else False
        agent_list = self.agent()
        action_prob_list = []
        i=0
        for ag in agent_list:
            deterministic_action = ag.policy(2)
            action_prob = torch.zeros(5)
            action_prob[deterministic_action]=1.0
            action_prob_list.append(action_prob)
            i+=1
        act={}
        for j in range(i):
            act["act%d"%j] = pyro.sample('action%d'%j, dist.Categorical(action_prob_list[j]))


    def observation(self, action):
        i=0
        action_data={}
        for act in action:
            action_data["action%d"%i] = torch.tensor(float(act))
            i+=1
        action_cond = pyro.condition(self.model, data=action_data)
        posterior = pyro.infer.Importance(action_cond, num_samples=1000)
        posterior = posterior.run()
        marginal = posterior.marginal(sites=['norm', 'reward_agent0','reward_agent1'])
        #probability table for both sites
        #marginal = posterior.marginal(sites=['norm', 'reward_agent0', 'reward_agent1', 'reward_agent2', 'reward_agent3', 'reward_agent4'])
        empirical = marginal.empirical


        for j in range(len(self.agent_norm)):
            print('Norm', NORM_DICT[j], ' : {:1.2f}'.format(empirical['norm'].log_prob(j).exp()))
        #for k in range(self.agent_no):
        #    new_reward_dict = self.reward_dict.copy()
        #    for rew in range(len(self.rew_prior['agent-0'])):
        #        print("Reward agent-", k, "reward: ", rew, ": {:1.2f}".format(empirical['reward_agent%d'%k].log_prob(rew).exp()))
            """new_reward_dict[rew] = [i * empirical['reward_agent%d'%k].log_prob(rew).exp() for i in new_reward_dict[rew]]
            agent_reward=[]
            print(new_reward_dict)
            for norm in self.env_norm:
                norm_reward=0
                for index in new_reward_dict:
                    new_reward_dict[index][norm]+=norm_reward
                agent_reward.append(norm_reward)
            print("Agent-", k, "Reward: ", agent_reward)"""


        #compute loss function
        loss_norm_mse = sum([(self.env_norm[norm] - empirical['norm'].log_prob(norm).exp())**2 \
                             for norm in self.env_norm])/len(self.env_norm)

        #update norm prior
        for i in range(len(self.agent_norm)):
            self.n_prior[NORM_DICT[i]] = empirical['norm'].log_prob(i).exp()
        #update reward prior
        for agent_no in range(self.agent_no):
            for rew in range(len(self.rew_prior['agent-0'])):
                self.rew_prior['agent-%d'%agent_no][rew] = empirical['reward_agent%d'%agent_no].log_prob(rew).exp()
        return loss_norm_mse



        """print("Observed Action: ")
        print(action_data)
        print("P(A|N=0 ")
        self.n_prior= {"G":1, "R":0, "B":0}
        posterior = pyro.infer.Importance(self.model, num_samples=10).run()
        marginal = posterior.marginal(sites=['action' + str(i) for i in range(5)])
        empirical = marginal.empirical
        print({'action' + str(i) : empirical['action' + str(i)].sample() for i in range(5)})

        print("P(A|N=1 ")
        self.n_prior= {"G":0, "R":1, "B":0}
        posterior = pyro.infer.Importance(self.model, num_samples=10).run()
        marginal = posterior.marginal(sites=['action' + str(i) for i in range(5)])
        empirical = marginal.empirical
        print({'action' + str(i) : empirical['action' + str(i)].sample() for i in range(5)})

        print("P(A|N=2 ")
        self.n_prior= {"G":0, "R":0, "B":1}
        posterior = pyro.infer.Importance(self.model, num_samples=10).run()
        marginal = posterior.marginal(sites=['action' + str(i) for i in range(5)])
        empirical = marginal.empirical
        print({'action' + str(i) : empirical['action' + str(i)].sample() for i in range(5)})
        print()

        self.n_prior= {"G":1/3, "R":1/3, "B":1/3}
        """

        """
        1. Get desire inference to work
        2. Make output file, save video, print out loss function
        3. Make the simulation less abstract
        4. Start to program phase 1
        """

