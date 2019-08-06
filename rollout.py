"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import utility_funcs
import numpy as np
import os
import sys
import shutil
import tensorflow as tf
import torch
import matplotlib
import matplotlib.pyplot as plt
import csv

from social_dilemmas.envs.norm import NormEnv, ExploreEnv
from social_dilemmas.observer_importance_sampling import Observer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'vid_path', os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')),
    'Path to directory where videos are saved.')
tf.app.flags.DEFINE_string(
    'env', 'norm', 'explore'
    'Name of the environment to rollout. Can be explore or norm.')
tf.app.flags.DEFINE_string(
    'render_type', 'pretty',
    'Can be pretty or fast. Implications obvious.')
tf.app.flags.DEFINE_integer(
    'fps', 8,
    'Number of frames per second.')

REWARD_PRIOR = {0: [1/3, 1/3, 1/3],
                1: [1/3, 1/3, 1/3],
                2: [1/3, 1/3, 1/3]}

class Controller(object):

    def __init__(self, env_name='norm'):
        self.env_name = env_name
        if env_name == 'norm':
            print('Initializing norm environment')
            self.env = NormEnv(num_agents=2, render=True,
                               norm={'G':True, 'R':False,'B':True})
        elif env_name == 'explore':
            print('Initializing explore environment')
            self.env = ExploreEnv(num_agents=1, render=True,
                                  norm={'G':True,'R':True, 'B':True}, reward={'G':0,'R':0,'B':0})
        else:
            print('Error! Not a valid environment type')
            return

        self.env.reset()

        # TODO: initialize agents here
    def explore(self, horizon=500, save_path=None):
        for i in range(horizon):
            agents = list(self.env.agents.values())
            # List of actions: 3- go right; 2 - go left; 1 - go down; 0 - go up;
            action_list = []
            for j in range(self.env.num_agents):
                act = agents[j].policy()
                action_list.append(act)
            obs, rew, dones, info, = self.env.step({'agent-%d'%k: action_list[k] for k in range(len(agents))})
            sys.stdout.flush()

            if save_path is not None:
                self.env.render(filename=save_path + 'frame' + str(i).zfill(6) + '.png')
            global REWARD_PRIOR
            prior = agents[0].return_reward_prior()
            REWARD_PRIOR = prior


    def rollout(self, horizon=500, save_path=None):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
            save_path: If provided, will save each frame to disk at this
                location.
        """
        rewards = []
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]
        observer = Observer(list(self.env.agents.values())[0].grid.copy(), REWARD_PRIOR)
        loss_norm=[]
        loss_reward=[]
        norm_alltime = {i:[] for i in range(len(self.env.norm))}
        reward_alltime = {"agent-%d"%agent:{fruit:[] for fruit in range(len(list(self.env.agents.values())[0].reward))} for agent in range(len(self.env.agents))}
        for hor in range(horizon):
            agents = list(self.env.agents.values())
            observer.update_grid(agents[0].grid)
            action_dim = agents[0].action_space.n
            depth = 2
            # List of actions: 3- go right; 2 - go left; 1 - go down; 0 - go up;
            action_list = []
            for j in range(self.env.num_agents):
                act = agents[j].policy(depth)
                action_list.append(act)
            obs, rew, dones, info, = self.env.step({'agent-%d'%k: action_list[k] for k in range(len(agents))})
            #observer makes an observation of the actions, return inferred norm and reward
            action_list.append(hor)
            norm, reward = observer.observation(tuple(action_list))

            # list of norm_alltime and reward_alltime for writing to file purpose
            for i in range(len(norm)):
                norm_alltime[i].append(norm[i])
            for i in range(len(agents)):
                for j in range(len(list(self.env.agents.values())[0].reward)):
                    reward_alltime['agent-%d'%i][j].append(reward['agent-%d'%i][j])

            # loss from exact enumeration
            #compute loss norm
            true_norm = [a_norm for a_norm in agents[0].norm if not agents[0].norm[a_norm]]
            norm_list = [1 if a_norm in true_norm else 0 for a_norm in self.env.norm]
            norm_diff = [norm_list[j] - norm[j] for j in range(len(norm_list))]
            loss_n = sum([norm_diff[j]**2 for j in range(len(norm_diff))]) / len(norm_diff)

            loss_norm.append(float(loss_n))

            #compute loss reward
            loss_mse_total=0
            norm_list=[a_norm for a_norm in self.env.norm]
            for agent in range(len(agents)):
                reward_diff = [agents[agent].reward[r] - reward["agent-{}".\
                    format(agent)][norm_list.index(r)] for r in agents[0].norm]
                loss_se_per_agent=0
                for r in reward_diff:
                    loss_se_per_agent += r**2
                loss_mse_total += loss_se_per_agent / len(agents[0].norm)
            loss_r = loss_mse_total / len(agents)
            loss_reward.append(float(loss_r))


            #loss from importance sampling
            for agent in range(self.env.num_agents):
                print("agent {} real reward:".format(agent), agents[agent].reward)

            sys.stdout.flush()

            if save_path is not None:
                self.env.render(filename=save_path + 'frame' + str(hor).zfill(6) + '.png')

            rgb_arr = self.env.map_to_colors()
            full_obs[hor] = rgb_arr.astype(np.uint8)
            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

        print("Loss norm: ", loss_norm)
        print("Loss reward: ", loss_reward)
        print("Norm all time: ", norm_alltime)
        print("Reward all time: ", reward_alltime)
        all_results = []
        timesteps =  (i for i in range(horizon))
        all_results.append(timesteps)
        all_results.append(tuple(loss_norm))
        all_results.append(tuple(loss_reward))
        for i in range(len(norm_alltime)):
            all_results.append(tuple(norm_alltime[i]))
        for agent in reward_alltime:
            for rew in reward_alltime[agent]:
                all_results.append(tuple(reward_alltime[agent][rew]))
        final_result = zip(*all_results)
        return rewards, observations, full_obs, final_result

    def render_rollout(self, horizon=500, path=None,
                       render_type='pretty', fps=8):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out.
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
            fps: Integer frames per second.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
        video_name = self.env_name + '_trajectory'

        if render_type == 'pretty':
            image_path = os.path.join(path, 'frames/')
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            if self.env_name=='explore':
                self.explore(horizon=horizon, save_path=image_path)
                utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
                                                        video_name=video_name)

            else:
                rewards, observations, full_obs, final_result = \
                    self.rollout(horizon=horizon, save_path=image_path)
                utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
                                                        video_name=video_name)

                with open('2-agents-50-hor-imps-uniform-rew-prior-5000.csv', 'w') as writeFile:
                    writer = csv.writer(writeFile)
                    writer.writerows(final_result)

            # Clean up images
            shutil.rmtree(image_path)
        else:
            if self.env_name=='explore':
                self.explore(horizon=horizon)
                utility_funcs.make_vidoe_from_rgb_imgs(path, image_path, fps=fps,
                                                        video_name=video_name)
            else:
                rewards, observations, full_obs, final_result = self.rollout(horizon=horizon)
                utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=fps,
                                                       video_name=video_name)


def main(unused_argv):
    c = Controller(env_name=FLAGS.env)
    c.render_rollout(path=FLAGS.vid_path, render_type=FLAGS.render_type,
                     fps=FLAGS.fps)


if __name__ == '__main__':
    tf.app.run(main)
