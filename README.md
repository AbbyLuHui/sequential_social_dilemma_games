[![Build Status](https://travis-ci.com/eugenevinitsky/sequential_social_dilemma_games.svg?branch=master)](https://travis-ci.com/eugenevinitsky/sequential_social_dilemma_games)

# Inferring Social Norms in Temporally Extended Multi-Agent Environments
This repo is an open-source implementation of DeepMind's Sequential Social Dilemma (SSD) multi-agent game-theoretic environments [1], adapted for norm inference purposes. The model features a modified version of Bayesian Theory of Mind (BToM) [2]. By observing factors such as world state and agent actions, the computer will make inferences on latent variables such as norm and desire. The inference process is implemented with probabilistic programming language Pyro (http://pyro.ai).

The implemented environments are structured to be compatible with OpenAIs gym environments (https://github.com/openai/gym) as well as RLlib's Multiagent Environment (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)


## Relevant papers

1. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).

2. Baker, C. L., Ettinger, J. J., Saxe, R., & Tenenbaum, J. B. (2017) [Rational quantitative attribution of beliefs, desires and percepts in human mentalizing] (https://www.nature.com/articles/s41562-017-0064). In Nature Human Behavior 1.

3. Tan, Z., Ong, D. C. (2019) [Bayesian Inference on Social Norms as Shared Constraints on Behavior] (https://arxiv.org/abs/1905.11110). arXiv preprint arXiv:1905.11110.

4. Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z. & de Freitas, N. (2018). [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647). arXiv preprint arXiv:1810.08647. 


# Setup instructions
Run `python setup.py develop`
Then, activate your environment by running `source activate causal`.

To then set up the branch of Ray on which we have built the causal influence code, clone the repo to your desired folder:
`git clone https://github.com/natashamjaques/ray.git`.

Next, go to the rllib folder:
` cd ray/python/ray/rllib ` and run the script `python setup-rllib-dev.py`. This will copy the rllib folder into the pip install of Ray and allow you to use the version of RLlib that is in your local folder by creating a softlink. 

# Tests
Tests are located in the test folder and can be run individually or run by running `python -m pytest`. Many of the less obviously defined rules for the games can be understood by reading the tests, each of which outline some aspect of the game. 
