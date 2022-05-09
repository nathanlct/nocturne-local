# TODO(ev) refactor, this is wildly similar to visualize_sample_factory
# run a policy over the entire train set

from collections import deque
import itertools
import json
import sys
import time
import os

import numpy as np
from pyvirtualdisplay import Display
import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution, CategoricalActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from run_sample_factory import register_custom_components

from cfgs.config import PROCESSED_TEST_NO_TL
from run_sample_factory import SampleFactoryEnv


def enjoy(cfgs, max_num_frames=1e9):
    actor_critics = []
    for i, cfg in enumerate(cfgs):
        cfg = load_from_checkpoint(cfg)

        render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
        if render_action_repeat is None:
            log.warning('Not using action repeat!')
            render_action_repeat = 1
        log.debug('Using action repeat %d during evaluation',
                  render_action_repeat)

        cfg.env_frameskip = 1  # for evaluation
        cfg.num_envs = 1

        def make_env_func(env_config):
            return create_env(cfg.env, cfg=cfg, env_config=env_config)

        env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
        env.seed(0)

        is_multiagent = is_multiagent_env(env)
        if not is_multiagent:
            env = MultiAgentWrapper(env)

        if hasattr(env.unwrapped, 'reset_on_init'):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False

        actor_critic = create_actor_critic(cfg, env.observation_space,
                                           env.action_space)

        device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index
        checkpoints = LearnerWorker.get_checkpoints(
            LearnerWorker.checkpoint_dir(cfg, policy_id))
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])
        actor_critics.append([i, actor_critic])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]

    goal_array = np.zeros((len(actor_critics), len(actor_critics)))
    collision_array = np.zeros((len(actor_critics), len(actor_critics)))
    for (index_1, actor_1), (index_2, actor_2) in itertools.product(
            actor_critics, actor_critics):
        goal_frac = 0
        collision_frac = 0
        for i, file in enumerate(os.listdir(PROCESSED_TEST_NO_TL)[0:100]):

            num_frames = 0
            env.files = [os.path.join(PROCESSED_TEST_NO_TL, file)]
            obs = env.reset()
            valid_indices = env.valid_indices
            indices_1 = []
            indices_2 = []
            # pick which valid indices go to which policy
            for index in valid_indices:
                val = np.random.uniform()
                if val < 0.5:
                    indices_1.append(index)
                else:
                    indices_2.append(index)
            rnn_states = torch.zeros(
                [env.num_agents, get_hidden_size(cfg)],
                dtype=torch.float32,
                device=device)
            episode_reward = np.zeros(env.num_agents)
            finished_episode = [False] * env.num_agents
            goal_achieved = [False] * len(env.valid_indices)

            while not all(finished_episode):
                with torch.no_grad():
                    obs_torch = AttrDict(transform_dict_observations(obs))
                    for key, x in obs_torch.items():
                        obs_torch[key] = torch.from_numpy(x).to(device).float()

                    policy_outputs = actor_1(obs_torch,
                                             rnn_states,
                                             with_action_distribution=True)
                    policy_outputs_2 = actor_2(obs_torch,
                                               rnn_states,
                                               with_action_distribution=True)

                    # sample actions from the distribution by default
                    # also update the indices that should be drawn from the second policy
                    # with its outputs
                    actions = policy_outputs.actions
                    actions[indices_2] = policy_outputs_2.actions[indices_2]

                    action_distribution = policy_outputs.action_distribution
                    if isinstance(action_distribution,
                                  ContinuousActionDistribution):
                        if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                            actions = action_distribution.means
                            actions[
                                indices_2] = policy_outputs_2.action_distribution.means[
                                    indices_2]
                    if isinstance(action_distribution,
                                  CategoricalActionDistribution):
                        if not cfg.discrete_actions_sample:
                            actions = policy_outputs['action_logits'].argmax(
                                axis=1)
                            actions[indices_2] = policy_outputs_2[
                                'action_logits'].argmax(axis=1)[indices_2]

                    actions = actions.cpu().numpy()

                    rnn_states = policy_outputs.rnn_states

                    for _ in range(render_action_repeat):

                        obs, rew, done, infos = env.step(actions)

                        episode_reward += rew
                        num_frames += 1

                        for agent_i, done_flag in enumerate(done):
                            if done_flag:
                                finished_episode[agent_i] = True
                                episode_rewards[agent_i].append(
                                    episode_reward[agent_i])
                                true_rewards[agent_i].append(
                                    infos[agent_i].get(
                                        'true_reward',
                                        episode_reward[agent_i]))
                                log.info(
                                    'Episode finished for agent %d at %d frames. Reward: %.3f, true_reward: %.3f',
                                    agent_i, num_frames,
                                    episode_reward[agent_i],
                                    true_rewards[agent_i][-1])
                                rnn_states[agent_i] = torch.zeros(
                                    [get_hidden_size(cfg)],
                                    dtype=torch.float32,
                                    device=device)
                                episode_reward[agent_i] = 0

                        # if episode terminated synchronously for all agents, pause a bit before starting a new one
                        if all(done):
                            time.sleep(0.05)

                        if all(finished_episode):
                            avg_episode_rewards_str, avg_true_reward_str = '', ''
                            for agent_i in range(env.num_agents):
                                avg_rew = np.mean(episode_rewards[agent_i])
                                avg_true_rew = np.mean(true_rewards[agent_i])
                                if not np.isnan(avg_rew):
                                    if avg_episode_rewards_str:
                                        avg_episode_rewards_str += ', '
                                    avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                                if not np.isnan(avg_true_rew):
                                    if avg_true_reward_str:
                                        avg_true_reward_str += ', '
                                    avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'
                            avg_goal = infos[0]['episode_extra_stats'][
                                'goal_achieved']
                            avg_collisions = infos[0]['episode_extra_stats'][
                                'collided']
                            goal_frac += avg_goal
                            collision_frac += avg_collisions
                            log.info(
                                f'Avg goal achieved, {goal_frac / (i + 1)}')
                            log.info(
                                f'Avg num collisions, {collision_frac / (i + 1)}'
                            )
                            log.info(
                                'Avg episode rewards: %s, true rewards: %s',
                                avg_episode_rewards_str, avg_true_reward_str)
                            log.info(
                                'Avg episode reward: %.3f, avg true_reward: %.3f',
                                np.mean([
                                    np.mean(episode_rewards[i])
                                    for i in range(env.num_agents)
                                ]),
                                np.mean([
                                    np.mean(true_rewards[i])
                                    for i in range(env.num_agents)
                                ]))
        # goal_array[index_1,
        #            index_2] = goal_frac / len(os.listdir(PROCESSED_TEST_NO_TL))
        # collision_array[index_1, index_2] = collision_frac / len(
        #     os.listdir(PROCESSED_TEST_NO_TL))
        goal_array[index_1, index_2] = goal_frac / (i + 1)
        collision_array[index_1, index_2] = collision_frac / (i + 1)

    np.savetxt('zsc_goal.txt', goal_array, delimiter=',')
    np.savetxt('zsc_collision.txt', collision_array, delimiter=',')
    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    disp = Display()
    disp.start()
    register_custom_components()
    file_paths = [
        '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.09/seed_sweepv2/11.38.50/0/seed_sweepv2/cfg.json',
        '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.09/seed_sweepv2/11.38.50/1/seed_sweepv2/cfg.json',
        '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.09/seed_sweepv2/11.38.50/2/seed_sweepv2/cfg.json',
    ]
    cfg_dicts = []
    for file in file_paths:
        with open(file, 'r') as file:
            cfg_dict = json.load(file)

        cfg_dict['cli_args'] = {}
        cfg_dict['fps'] = 0
        cfg_dict['render_action_repeat'] = None
        cfg_dict['no_render'] = None
        cfg_dict['policy_index'] = 0
        cfg_dict['record_to'] = os.path.join(os.getcwd(), '..', 'recs')
        cfg_dict['continuous_actions_sample'] = True
        cfg_dict['discrete_actions_sample'] = True
        cfg_dicts.append(cfg_dict)

    class Bunch(object):

        def __init__(self, adict):
            self.__dict__.update(adict)

    cfgs = []
    for cfg in cfg_dicts:
        cfg = Bunch(cfg_dict)
        cfgs.append(cfg)
    status, avg_reward = enjoy(cfgs)
    return status


if __name__ == '__main__':
    sys.exit(main())
