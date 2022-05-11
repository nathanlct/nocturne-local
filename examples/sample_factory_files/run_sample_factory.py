"""
To run in single agent mode on one file for testing
python -m run_sample_factory algorithm=APPO ++algorithm.train_in_background_thread=True ++algorithm.num_workers=10 ++algorithm.experiment=EXPERIMENT_NAME ++single_agent_mode=True ++num_files=1 

To run in multiagent mode on one file for testing
python -m run_sample_factory algorithm=APPO ++algorithm.train_in_background_thread=True ++algorithm.num_workers=10 ++algorithm.experiment=EXPERIMENT_NAME ++single_agent_mode=False ++num_files=1 

To run on all files set num_files=-1

For debugging
python -m run_sample_factory algorithm=APPO ++algorithm.train_in_background_thread=False ++algorithm.num_workers=1 ++force_envs_single_thread=False
After training for a desired period of time, evaluate the policy by running:
python -m sample_factory_examples.enjoy_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example
"""
import os
import random
import sys

import hydra
import gym
import numpy as np
from omegaconf import OmegaConf
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm
from sample_factory_examples.train_custom_env_custom_model import CustomEncoder, override_default_params_func

from nocturne_utils.wrappers import create_env


class SampleFactoryEnv():

    def __init__(self, env):
        self.env = env
        if self.env.single_agent_mode:
            self.num_agents = 1
        else:
            self.num_agents = self.env.cfg[
                'max_num_vehicles']  # TODO(ev) pick a good value
        self.agent_ids = [i for i in range(self.num_agents)]
        self.is_multiagent = True
        obs = self.env.reset()
        # used to track which agents are done
        self.already_done = [False for _ in self.agent_ids]
        self.episode_rewards = np.zeros(self.num_agents)

    def step(self, actions):
        agent_actions = {}
        for action, agent_id, already_done in zip(actions, self.agent_ids,
                                                  self.already_done):
            if already_done:
                continue
            agent_actions[self.agent_id_to_env_id_map[agent_id]] = action
        next_obses, rew, done, info = self.env.step(agent_actions)
        rew_n = []
        done_n = []
        info_n = []

        for agent_id in self.agent_ids:
            # first check that the agent_id ever had a corresponding vehicle
            # and then check that there's actually an observation for it i.e. it's not done
            if agent_id in self.agent_id_to_env_id_map.keys(
            ) and self.agent_id_to_env_id_map[agent_id] in next_obses.keys():
                map_key = self.agent_id_to_env_id_map[agent_id]
                # since the environment may have just reset, we don't actually have
                # reward objects yet
                rew_n.append(rew.get(map_key, 0))
                agent_info = info.get(map_key, {})
                # track the per-agent reward for later logging
                self.episode_rewards[agent_id] += rew.get(map_key, 0)
                self.num_steps[agent_id] += 1
                self.goal_achieved[agent_id] = self.goal_achieved[
                    agent_id] or agent_info['goal_achieved']
                self.collided[agent_id] = self.collided[
                    agent_id] or agent_info['collided']
            else:
                rew_n.append(0)
                agent_info = {}
            if self.already_done[agent_id]:
                agent_info['is_active'] = False
            else:
                agent_info['is_active'] = True
            info_n.append(agent_info)
        # now stick in some extra state information if needed
        # anything in episode_extra_stats is logged at the end of the episode
        if done['__all__']:
            # log any extra info that you need
            avg_rew = np.mean(self.episode_rewards[self.valid_indices])
            avg_len = np.mean(self.num_steps[self.valid_indices])
            avg_goal_achieved = np.mean(self.goal_achieved[self.valid_indices])
            avg_collided = np.mean(self.collided[self.valid_indices])
            for info in info_n:
                info['episode_extra_stats'] = {}
                info['episode_extra_stats']['avg_rew'] = avg_rew
                info['episode_extra_stats']['avg_agent_len'] = avg_len
                info['episode_extra_stats'][
                    'goal_achieved'] = avg_goal_achieved
                info['episode_extra_stats']['collided'] = avg_collided

        # update the dones so we know if we need to reset
        # sample factory does not call reset for you
        for env_id, done_val in done.items():
            # handle the __all__ signal that's just in there for
            # telling when the environment should stop
            if env_id == '__all__':
                continue
            if done_val:
                agent_id = self.env_id_to_agent_id_map[env_id]
                self.already_done[agent_id] = True

        # okay, now if all the agents are done set done to True for all of them
        # otherwise, False. Sample factory uses info['is_active'] to track if agents
        # are done, not the done signal
        # also, convert the obs_dict into the right format
        if done['__all__']:
            done_n = [True] * self.num_agents
            obs_n = self.reset()
        else:
            done_n = [False] * self.num_agents
            obs_n = self.obs_dict_to_list(next_obses)
        return obs_n, rew_n, done_n, info_n

    def obs_dict_to_list(self, obs_dict):
        obs_n = []
        for agent_id in self.agent_ids:
            # first check that the agent_id ever had a corresponding vehicle
            # and then check that there's actually an observation for it i.e. it's not done
            if agent_id in self.agent_id_to_env_id_map.keys(
            ) and self.agent_id_to_env_id_map[agent_id] in obs_dict.keys():
                map_key = self.agent_id_to_env_id_map[agent_id]
                obs_n.append(obs_dict[map_key])
            else:
                obs_n.append(self.dead_feat)
        return obs_n

    def reset(self):
        # track the agent_ids that actually take an action during the episode
        self.valid_indices = []
        self.episode_rewards = np.zeros(self.num_agents)
        self.num_steps = np.zeros(self.num_agents)
        self.goal_achieved = np.zeros(self.num_agents)
        self.collided = np.zeros(self.num_agents)
        self.already_done = [False for _ in self.agent_ids]
        next_obses = self.env.reset()
        env_keys = sorted(list(next_obses.keys()))
        # agent ids is a list going from 0 to (num_agents - 1)
        # however, the vehicle IDs might go from 0 to anything
        # we want to initialize a mapping that is maintained through the episode and always
        # uniquely convert the vehicle ID to an agent id
        self.agent_id_to_env_id_map = {
            agent_id: env_id
            for agent_id, env_id in zip(self.agent_ids, env_keys)
        }
        self.env_id_to_agent_id_map = {
            env_id: agent_id
            for agent_id, env_id in zip(self.agent_ids, env_keys)
        }
        # if there isn't a mapping from an agent id to a vehicle id, that agent should be
        # set to permanently inactive
        for agent_id in self.agent_ids:
            if agent_id not in self.agent_id_to_env_id_map.keys():
                self.already_done[agent_id] = True
            else:
                # check that this isn't actually a fake padding agent used
                # when keep_inactive_agents is True
                if agent_id in self.agent_id_to_env_id_map.keys() and \
                    self.agent_id_to_env_id_map[agent_id] not in self.env.dead_agent_ids:
                    self.valid_indices.append(agent_id)
        obs_n = self.obs_dict_to_list(next_obses)
        return obs_n

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, mode=None):
        return self.env.render(mode)

    def seed(self, seed=None):
        self.env.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_custom_multi_env_func(full_env_name, cfg=None, env_config=None):
    env = create_env(cfg)
    # env = RecordingWrapper(
    #     env, os.path.join(os.getcwd(), cfg['algorithm']['experiment'],
    #                       'videos'), 0)
    return SampleFactoryEnv(env)


def add_extra_params_func(env, parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument('--custom_env_episode_len',
                   default=10,
                   type=int,
                   help='Number of steps in the episode')


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='my_custom_multi_env_',
        make_env_func=make_custom_multi_env_func,
        # add_extra_params_func=add_extra_params_func,
        override_default_params_func=override_default_params_func,
    )

    # register_custom_encoder('custom_env_encoder', CustomEncoder)


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """Script entry point."""
    register_custom_components()
    # cfg = parse_args()
    # TODO(ev) hacky renaming and restructuring, better to do this cleanly
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # copy algo keys into the main keys
    for key, value in cfg_dict['algorithm'].items():
        cfg_dict[key] = value
    # we didn't set a train directory so use the hydra one
    if cfg_dict['train_dir'] is None:
        cfg_dict['train_dir'] = os.getcwd()
        print(f'storing the results in {os.getcwd()}')
    else:
        output_dir = cfg_dict['train_dir']
        print(f'storing results in {output_dir}')

    # recommendation from Aleksei to keep horizon length fixed
    # and number of agents fixed and just pad missing / exited
    # agents with a vector of -1s
    if not cfg_dict['single_agent_mode']:
        cfg_dict['subscriber']['keep_inactive_agents'] = True

    # put it into a namespace so sample factory code runs correctly
    class Bunch(object):

        def __init__(self, adict):
            self.__dict__.update(adict)

    cfg = Bunch(cfg_dict)
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())