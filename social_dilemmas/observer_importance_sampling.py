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

UNIFORM_REWARD_PRIOR = {0: torch.tensor([1/3, 1/3, 1/3]),
                1: torch.tensor([1/3, 1/3, 1/3]),
                2: torch.tensor([1/3, 1/3, 1/3])} #norm: [reward=0, reward=1, reward=2]

REWARD_LIST=[0,1,2]

ACTION_DICT={0: "GO UP",
             1: "GO DOWN",
             2: "GO LEFT",
             3: "GO RIGHT"}


class Observer():
    def __init__(self, grid, explorer_reward):
        self.grid = grid
        self.agent_norm = {"G":True, "R":False, "B":False}
        self.rew_prior = {} #across 27 possibilities
        self.n_prior= [1/3, 1/3, 1/3]
        self.agent_no=0
        self.reward_dict = {}
        self.update_reward_dict()
        self.probability_dict={}
        locs = self.get_agent_locs() #update agent_no

        #take in explorer reward sample, then update the inferred overarching distribution
        self.REWARD_PRIOR= {}
        for ag in range(self.agent_no):
            self.REWARD_PRIOR["agent-{}".format(ag)]=explorer_reward
        print("EXPLORER REWARD: ", explorer_reward)

    def update_reward_dict(self):
        #create reward_dict for easy interpretation of categorical distribution result
        #0: [0,0,0], 1: [0,0,1] etc
        length = len(REWARD_LIST)
        for a in range(length):
            for b in range(length):
                for c in range(length):
                    self.reward_dict[(length**2)*a+length*b+c] = [REWARD_LIST[a], REWARD_LIST[b], REWARD_LIST[c]]

    def norm_prior(self):
        prob_tensor = torch.zeros(len(self.agent_norm))
        total = sum(self.n_prior)
        temp_list = [i / total for i in self.n_prior] #normalize
        for i in range(len(self.agent_norm)):
            prob_tensor[i] = temp_list[i]
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
                    prior = prior * self.REWARD_PRIOR['agent-%d'%agent_no][norm_no][rew]
                self.rew_prior['agent-%d'%agent_no][rew_no] = prior
        for j in range(self.agent_no):
            reward_prior["agent-" + str(j)] = pyro.sample("reward_agent" + str(j), \
                                                          dist.Categorical(self.rew_prior["agent-"+str(j)]))
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
        # set up initial positions
        agent_locs= self.get_agent_locs()
        # set up norm prior
        n_prior = self.norm_prior()
        for norm in self.agent_norm:
            self.agent_norm[norm] = False if norm == NORM_DICT[int(n_prior)] else True
        # set up reward prior
        rew = self.reward_prior()
        # set up agents
        agent_list=[]
        for ag in range(self.agent_no):
            reward = {NORM_DICT[reward_index] : self.reward_dict[int(rew['agent-'+str(ag)])][reward_index]
                      for reward_index in range(len(self.agent_norm))}
            one_agent = NormAgent("agent%d" %ag, agent_locs[ag], 'UP', self.grid, self.agent_norm, reward)
            agent_list.append(one_agent)
        return agent_list

    def update_grid(self, grid):
        self.grid = grid

    #pyro model
    def model(self):
        # set up agents according to norm prior and reward prior
        agent_list = self.agent()

        #condition action upon norm & reward prior
        action_prob_list = []
        for ag in agent_list:
            deterministic_action = ag.policy(2)
            action_prob = torch.zeros(5)
            action_prob[deterministic_action]=1.0
            action_prob_list.append(action_prob)
        act={}
        for j in range(self.agent_no):
            act["act%d"%j] = pyro.sample('action%d'%j, dist.Categorical(action_prob_list[j]))


    def observation(self, action):
        action = action[0:len(action)-1]
        action_data={"action%d"%i : torch.tensor(float(action[i])) for i in range(len(action))}
        print("Action: ", action)

        #pyro importance sampling
        action_cond = pyro.condition(self.model, data=action_data)
        posterior = pyro.infer.Importance(action_cond, num_samples=800)
        posterior = posterior.run()
        marginal = posterior.marginal(sites=['norm'] + ['reward_agent%d'%rew_no for rew_no in range(self.agent_no)])
        empirical = marginal.empirical

        #calculate norm posterior
        inferred_norm = {}
        for j in range(len(self.agent_norm)):
            inferred_norm[j] = float(empirical['norm'].log_prob(j).exp())

        #calculate reward posterior
        inferred_reward = {}
        for k in range(self.agent_no):
            new_reward_dict = self.reward_dict.copy()
            agent_reward=[]
            for rew in range(len(self.rew_prior['agent-0'])):
                rew_possibility = float(empirical['reward_agent%d'%k].log_prob(rew).exp())
                new_reward_dict[rew] = [i * rew_possibility for i in new_reward_dict[rew]]
                if rew==0:
                    agent_reward=new_reward_dict[0]
                else:
                    agent_reward=[x+y for x,y in zip(agent_reward, new_reward_dict[rew])]
            inferred_reward["agent-{}".format(k)] = agent_reward

        #update norm prior
        for i in range(len(self.agent_norm)):
            self.n_prior[i] = empirical['norm'].log_prob(i).exp()

        #update reward prior
        for agent_no in range(self.agent_no):
            for rew in range(len(self.rew_prior['agent-0'])):
                self.rew_prior['agent-%d'%agent_no][rew] = empirical['reward_agent%d'%agent_no].log_prob(rew).exp()

        print("Inferred norm: ", inferred_norm)
        print("Inferred reward: ", inferred_reward)
        return inferred_norm, inferred_reward



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

