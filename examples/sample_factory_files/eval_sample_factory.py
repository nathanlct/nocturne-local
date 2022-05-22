"""Run a policy over the entire train set.

TODO(ev) refactor, this is wildly similar to visualize_sample_factory
"""

from copy import deepcopy
from collections import deque, defaultdict
import itertools
import json
import sys
import os

import numpy as np
from pyvirtualdisplay import Display
import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution, \
     CategoricalActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from run_sample_factory import register_custom_components

from cfgs.config import PROCESSED_VALID_NO_TL


def run_eval(cfgs):
    """Eval a stored agent over all files in validation set.

    Args:
        cfg (dict): configuration file for instantiating the agents and environment.

    Returns
    -------
        None: None
    """
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

    goal_array = np.zeros((len(actor_critics), len(actor_critics)))
    collision_array = np.zeros((len(actor_critics), len(actor_critics)))
    success_rate_by_num_agents = np.zeros(
        (len(actor_critics), len(actor_critics), cfg.max_num_vehicles, 3))
    # we bin the success rate into bins of 10 meters between 0 and 400
    # the second dimension is the counts
    distance_bins = np.linspace(0, 400, 40)
    success_rate_by_distance = np.zeros(
        (len(actor_critics), len(actor_critics), distance_bins.shape[-1], 3))
    f_path = PROCESSED_VALID_NO_TL
    files = os.listdir(PROCESSED_VALID_NO_TL)
    files = [file for file in files if 'tfrecord' in file]

    for (index_1, actor_1), (index_2, actor_2) in itertools.product(
            actor_critics, actor_critics):
        episode_rewards = [
            deque([], maxlen=100) for _ in range(env.num_agents)
        ]
        true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        goal_frac = 0
        collision_frac = 0
        for file_num, file in enumerate(files[0:200]):
            # for i, file in enumerate(os.listdir(files)[0:100]):

            num_frames = 0
            env.unwrapped.files = [os.path.join(f_path, file)]

            # step the env to its conclusion to generate the expert trajectories we compare against
            env.reset()
            trajectory_dict = defaultdict(lambda: np.zeros((80, 2)))
            env.unwrapped.make_all_vehicles_experts()
            for i in range(80):
                for veh in env.unwrapped.get_vehicles:
                    trajectory_dict[veh.id] = veh.position.numpy()
                env.step()
            obs = env.reset()
            # some key information for tracking statistics
            goal_dist = env.goal_dist_normalizers
            valid_indices = env.valid_indices
            agent_id_to_env_id_map = env.agent_id_to_env_id_map
            # pick which valid indices go to which policy
            val = np.random.uniform()
            if val < 0.5:
                num_choice = int(np.floor(len(valid_indices) / 2.0))
            else:
                num_choice = int(np.ceil(len(valid_indices) / 2.0))
            indices_1 = list(
                np.random.choice(valid_indices, num_choice, replace=False))
            indices_2 = [val for val in valid_indices if val not in indices_1]
            rnn_states = torch.zeros(
                [env.num_agents, get_hidden_size(cfg)],
                dtype=torch.float32,
                device=device)
            rnn_states_2 = torch.zeros(
                [env.num_agents, get_hidden_size(cfg)],
                dtype=torch.float32,
                device=device)
            episode_reward = np.zeros(env.num_agents)
            finished_episode = [False] * env.num_agents
            goal_achieved = [False] * len(valid_indices)
            collision_observed = [False] * len(valid_indices)

            while not all(finished_episode):
                with torch.no_grad():
                    obs_torch = AttrDict(transform_dict_observations(obs))
                    for key, x in obs_torch.items():
                        obs_torch[key] = torch.from_numpy(x).to(device).float()

                    obs_torch_2 = deepcopy(obs_torch)
                    policy_outputs = actor_1(obs_torch,
                                             rnn_states,
                                             with_action_distribution=True)
                    policy_outputs_2 = actor_2(obs_torch_2,
                                               rnn_states_2,
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
                    rnn_states_2 = policy_outputs_2.rnn_states

                    for _ in range(render_action_repeat):

                        obs, rew, done, infos = env.step(actions)
                        episode_reward += rew
                        num_frames += 1

                        for i, index in enumerate(valid_indices):
                            goal_achieved[i] = infos[index][
                                'goal_achieved'] or goal_achieved[i]
                            collision_observed[i] = infos[index][
                                'collided'] or collision_observed[i]

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
                            goal_frac = (file_num * goal_frac +
                                         avg_goal) / (file_num + 1)
                            collision_frac = (file_num * collision_frac +
                                              avg_collisions) / (file_num + 1)
                            success_rate_by_num_agents[index_1, index_2,
                                                       len(valid_indices) - 1,
                                                       0] += avg_goal
                            success_rate_by_num_agents[index_1, index_2,
                                                       len(valid_indices) - 1,
                                                       1] += avg_collisions
                            success_rate_by_num_agents[index_1, index_2,
                                                       len(valid_indices) - 1,
                                                       2] += 1
                            # track how well we do as a function of distance
                            for i, index in enumerate(valid_indices):
                                env_id = agent_id_to_env_id_map[index]
                                bin = np.searchsorted(distance_bins,
                                                      goal_dist[env_id])
                                success_rate_by_distance[index_1, index_2,
                                                         bin - 1,
                                                         0] += goal_achieved[i]
                                success_rate_by_distance[
                                    index_1, index_2, bin - 1,
                                    1] += collision_observed[i]
                                success_rate_by_distance[index_1, index_2,
                                                         bin - 1, 2] += 1
                            # do some logging
                            log.info(f'Avg goal achieved, {goal_frac}')
                            log.info(f'Avg num collisions, {collision_frac}')
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
        goal_array[index_1, index_2] = goal_frac
        collision_array[index_1, index_2] = collision_frac

    np.savetxt('results/zsc_goal.txt', goal_array, delimiter=',')
    np.savetxt('results/zsc_collision.txt', collision_array, delimiter=',')
    with open('results/success_by_veh_number.npy', 'wb') as f:
        success_ratio = np.nan_to_num(
            success_rate_by_num_agents[:, :, :, 0:2] /
            success_rate_by_num_agents[:, :, :, [2]])
        print(success_ratio)
        np.save(f, success_ratio)
    with open('results/success_by_dist.npy', 'wb') as f:
        dist_ratio = np.nan_to_num(success_rate_by_distance[:, :, :, 0:2] /
                                   success_rate_by_distance[:, :, :, [2]])
        print(dist_ratio)
        np.save(f, dist_ratio)

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    disp = Display()
    disp.start()
    register_custom_components()
    # file_paths = [
    #     # '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.09/seed_sweepv2/11.38.50/0/seed_sweepv2/cfg.json',
    #     '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.10/new_features/15.10.09/2/new_features/cfg.json',
    #     '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.10/new_features/15.10.09/4/new_features/cfg.json',
    #     # '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.09/seed_sweepv2/11.38.50/2/seed_sweepv2/cfg.json',
    # ]
    file_paths = [
        '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.20/new_road_sample/18.32.35/13/new_road_sample/cfg.json',
        '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.20/new_road_sample/18.32.35/14/new_road_sample/cfg.json'
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
    status, avg_reward = run_eval(cfgs)
    return status


if __name__ == '__main__':
    sys.exit(main())
