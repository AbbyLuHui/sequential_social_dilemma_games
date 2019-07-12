import torch
torch.set_default_dtype(torch.float64)
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate, infer_discrete
from pyro.distributions import Categorical, Empirical
from social_dilemmas.search_inference import factor, HashingMarginal, memoize, Search
from social_dilemmas.envs.agent import NormAgent

DEPTH = 2
NORM_DICT={0: "G",
           1: "R",
           2: "B"}

REWARD_PRIOR = {0: [0., 0.1, 0.9],
                1: [0.9, 0.1, 0.],
                2: [1/3, 1/3, 1/3]} #norm: [reward=0, reward=1, reward=2]

REWARD_LIST=[0,1,2]

ACTION_DICT={0: "GO UP",
             1: "GO DOWN",
             2: "GO LEFT",
             3: "GO RIGHT"}

ENV_NORM = {0:1, 1:0, 2:0}

#State = collections.namedtuple("State", ["norm", "action"])

def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

class Observer():
    def __init__(self, grid):
        self.grid = grid
        self.env_norm = ENV_NORM
        self.agent_norm = {"G":True, "R":False, "B":False}
        self.rew_prior = {}
        #self.rew_prior = {"G":0.5, "R":0.5, "B":0.5}
        self.n_prior= [1/3,1/3,1/3]
        self.agent_no=0
        self.reward_dict = {}
        self.update_reward_dict()
        self.real_reward=[]
        self.probability_dict={}

    def update_reward_dict(self):
        #create reward_dict for easy interpretation of categorical distribution result
        length = len(REWARD_LIST)
        for a in range(length):
            for b in range(length):
                for c in range(length):
                    self.reward_dict[(length**2)*a+length*b+c] = [REWARD_LIST[a], REWARD_LIST[b], REWARD_LIST[c]]

    #assume norm follows categorical distribution
    def norm_prior(self):
        prob_tensor = torch.zeros(len(self.env_norm))
        for i in range(len(self.env_norm)):
            prob_tensor[i] = self.n_prior[i]
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
        return agent_list, rew

    def update_grid(self, grid):
        self.grid = grid

    def update_reward(self, reward):
        for rew in reward:
            self.real_reward.append([rew[norm] for norm in self.agent_norm])

    #model of norm, action conditioned upon norm
    @Marginal
    def model(self,data):
        n_prior = self.norm_prior()
        for norm in self.agent_norm:
            self.agent_norm[norm] = True if norm == NORM_DICT[int(n_prior)] else False
        agent_list, rew = self.agent()
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
            act["act%d"%j] = pyro.sample('action%d'%j, dist.Categorical(action_prob_list[j]), obs=torch.tensor(data[j]))
        return (n_prior.item(),) + tuple(rew["agent-{}".format(ag)].item() for ag in range(self.agent_no)) + \
               tuple(act["act{}".format(k)].item() for k in range(self.agent_no))


    def observation(self, action):
        support = self.model(action).enumerate_support()
        data = [self.model(action).log_prob(s).exp().item() for s in self.model(action).enumerate_support()]
        self.probability_dict={support[index]: data[index] for index in range(len(support))}
        #compute norm, reward
        norm = 0
        reward = {"agent-{}".format(index):[0*i for i in range(len(self.env_norm))] for index in range(self.agent_no)}
        for key in self.probability_dict:
            norm += key[0] * self.probability_dict[key]
            for index in range(self.agent_no):
                reward_list = self.reward_dict[key[index+1]]
                new_reward_list = [x * self.probability_dict[key] for x in reward_list]
                reward["agent-{}".format(index)] = [new_reward_list[i]+reward["agent-{}".format(index)][i] \
                                                    for i in range(len(self.env_norm))]
        print("NORM:", norm)
        print("REWARD: ", reward)
        return norm, reward


        #print reward inference, compute reward loss function
        """loss_reward_total=0
        for k in range(self.agent_no):
            new_reward_dict = self.reward_dict.copy()
            for rew in range(len(self.rew_prior['agent-0'])):
                #print("Reward agent-", k, "reward: ", rew, ": {:1.2f}".format(empirical['reward_agent%d'%k].log_prob(rew).exp()))
                new_reward_dict[rew] = [i * float(empirical['reward_agent%d'%k].log_prob(rew).exp()) for i in new_reward_dict[rew]]
                agent_reward=[]
                #print(new_reward_dict)
                for norm in self.env_norm:
                    norm_reward=0
                    for index in new_reward_dict:
                        norm_reward+=new_reward_dict[index][norm]
                    agent_reward.append(norm_reward)
            print("Agent-", k, "Reward: ", agent_reward)
            for rew in range(len(self.env_norm)):
                loss_reward_total+=(self.real_reward[k][rew] - agent_reward[rew])**2
        loss_reward_mse = loss_reward_total / len(self.env_norm)


        #compute norm loss function
        loss_norm_mse = sum([(self.env_norm[norm] - empirical['norm'].log_prob(norm).exp())**2 \
                             for norm in self.env_norm])/len(self.env_norm)

        #update norm prior
        for i in range(len(self.agent_norm)):
            self.n_prior[NORM_DICT[i]] = empirical['norm'].log_prob(i).exp()
        #update reward prior
        for agent_no in range(self.agent_no):
            for rew in range(len(self.rew_prior['agent-0'])):
                self.rew_prior['agent-%d'%agent_no][rew] = empirical['reward_agent%d'%agent_no].log_prob(rew).exp()
        return loss_norm_mse, loss_reward_mse"""



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
        3. Exact enumeration
        4. Make the simulation less abstract
        5. Start to program phase 1
        
        Git log
        git add  xxx
        git commit -a
        git status
        git push
        
        Wednesday:
        exact enumeration
        
        Problem:
        Cannot differentiate between norm and reward
        Cannot attribute correctly when desire is low and norm is high, will assume desire and norm are both high
        Will be interesting to see what happens when norm is weekened
        
        
        Interesting case:
        When the agent just does not like blue apple (e.g. reward = 0), it will just stay there and not move.
        """

