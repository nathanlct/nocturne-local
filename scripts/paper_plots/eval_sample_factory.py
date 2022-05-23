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
import pandas as pd
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
from examples.sample_factory_files.run_sample_factory import register_custom_components

from cfgs.config import PROCESSED_VALID_NO_TL, ERR_VAL


def run_eval(cfgs, test_zsc, output_path):
    """Eval a stored agent over all files in validation set.

    Args:
        cfg (dict): configuration file for instantiating the agents and environment.
        test_zsc (bool): if true, we play all agents against all agents

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

    # we bin the success rate into bins of 10 meters between 0 and 400
    # the second dimension is the counts
    distance_bins = np.linspace(0, 400, 40)
    if test_zsc:
        goal_array = np.zeros((len(actor_critics), len(actor_critics)))
        collision_array = np.zeros((len(actor_critics), len(actor_critics)))
        success_rate_by_num_agents = np.zeros(
            (len(actor_critics), len(actor_critics), cfg.max_num_vehicles, 3))
        success_rate_by_distance = np.zeros(
            (len(actor_critics), len(actor_critics), distance_bins.shape[-1],
             3))
    else:
        goal_array = np.zeros(len(actor_critics))
        collision_array = np.zeros(len(actor_critics))
        success_rate_by_num_agents = np.zeros(
            (len(actor_critics), cfg.max_num_vehicles, 3))
        success_rate_by_distance = np.zeros(
            (len(actor_critics), distance_bins.shape[-1], 3))
    ade_array = np.zeros(len(actor_critics))
    fde_array = np.zeros(len(actor_critics))
    f_path = PROCESSED_VALID_NO_TL
    files = os.listdir(PROCESSED_VALID_NO_TL)
    files = [file for file in files if 'tfrecord' in file]

    if test_zsc:
        output_generator = itertools.product(actor_critics, actor_critics)
    else:
        output_generator = actor_critics

    for output in output_generator:
        if test_zsc:
            (index_1, actor_1), (index_2, actor_2) = output
        else:
            (index_1, actor_1) = output
        episode_rewards = [
            deque([], maxlen=100) for _ in range(env.num_agents)
        ]
        true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        goal_frac = 0
        collision_frac = 0
        average_displacement_error = 0
        final_displacement_error = 0
        veh_counter = 0
        for file_num, file in enumerate(files[0:cfg['num_eval_files']]):

            num_frames = 0
            env.unwrapped.files = [os.path.join(f_path, file)]

            # step the env to its conclusion to generate the expert trajectories we compare against
            env.reset()
            expert_trajectory_dict = defaultdict(lambda: np.zeros((80, 2)))
            env.unwrapped.make_all_vehicles_experts()
            for i in range(80):
                for veh in env.unwrapped.get_objects_that_moved():
                    expert_trajectory_dict[veh.id][i] = veh.position.numpy()
                env.unwrapped.simulation.step(0.1)

            obs = env.reset()

            rollout_traj_dict = defaultdict(lambda: np.zeros((80, 2)))
            # some key information for tracking statistics
            goal_dist = env.goal_dist_normalizers
            valid_indices = env.valid_indices
            agent_id_to_env_id_map = env.agent_id_to_env_id_map
            if test_zsc:
                # pick which valid indices go to which policy
                val = np.random.uniform()
                if val < 0.5:
                    num_choice = int(np.floor(len(valid_indices) / 2.0))
                else:
                    num_choice = int(np.ceil(len(valid_indices) / 2.0))
                indices_1 = list(
                    np.random.choice(valid_indices, num_choice, replace=False))
                indices_2 = [
                    val for val in valid_indices if val not in indices_1
                ]
                rnn_states = torch.zeros(
                    [env.num_agents, get_hidden_size(cfg)],
                    dtype=torch.float32,
                    device=device)
                rnn_states_2 = torch.zeros(
                    [env.num_agents, get_hidden_size(cfg)],
                    dtype=torch.float32,
                    device=device)
            else:
                rnn_states = torch.zeros(
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

                    policy_outputs = actor_1(obs_torch,
                                             rnn_states,
                                             with_action_distribution=True)
                    if test_zsc:
                        obs_torch_2 = deepcopy(obs_torch)
                        policy_outputs_2 = actor_2(
                            obs_torch_2,
                            rnn_states_2,
                            with_action_distribution=True)

                    # sample actions from the distribution by default
                    # also update the indices that should be drawn from the second policy
                    # with its outputs
                    actions = policy_outputs.actions
                    if test_zsc:
                        actions[indices_2] = policy_outputs_2.actions[
                            indices_2]

                    action_distribution = policy_outputs.action_distribution
                    if isinstance(action_distribution,
                                  ContinuousActionDistribution):
                        if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                            actions = action_distribution.means
                            if test_zsc:
                                actions[
                                    indices_2] = policy_outputs_2.action_distribution.means[
                                        indices_2]
                    if isinstance(action_distribution,
                                  CategoricalActionDistribution):
                        if not cfg.discrete_actions_sample:
                            actions = policy_outputs['action_logits'].argmax(
                                axis=1)
                            if test_zsc:
                                actions[indices_2] = policy_outputs_2[
                                    'action_logits'].argmax(axis=1)[indices_2]

                    actions = actions.cpu().numpy()

                    for veh in env.unwrapped.get_objects_that_moved():
                        # only check vehicles we are actually controlling
                        if veh.expert_control == False:
                            rollout_traj_dict[veh.id][
                                env.step_num] = veh.position.numpy()

                    rnn_states = policy_outputs.rnn_states
                    if test_zsc:
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
                            if test_zsc:
                                success_rate_by_num_agents[index_1, index_2,
                                                           len(valid_indices) -
                                                           1, 0] += avg_goal
                                success_rate_by_num_agents[index_1, index_2,
                                                           len(valid_indices) -
                                                           1,
                                                           1] += avg_collisions
                                success_rate_by_num_agents[index_1, index_2,
                                                           len(valid_indices) -
                                                           1, 2] += 1
                            else:
                                success_rate_by_num_agents[index_1,
                                                           len(valid_indices) -
                                                           1, 0] += avg_goal
                                success_rate_by_num_agents[index_1,
                                                           len(valid_indices) -
                                                           1,
                                                           1] += avg_collisions
                                success_rate_by_num_agents[index_1,
                                                           len(valid_indices) -
                                                           1, 2] += 1
                            # track how well we do as a function of distance
                            for i, index in enumerate(valid_indices):
                                env_id = agent_id_to_env_id_map[index]
                                bin = np.searchsorted(distance_bins,
                                                      goal_dist[env_id])
                                if test_zsc:
                                    success_rate_by_distance[
                                        index_1, index_2, bin - 1,
                                        0] += goal_achieved[i]
                                    success_rate_by_distance[
                                        index_1, index_2, bin - 1,
                                        1] += collision_observed[i]
                                    success_rate_by_distance[index_1, index_2,
                                                             bin - 1, 2] += 1
                                else:
                                    success_rate_by_distance[
                                        index_1, bin - 1,
                                        0] += goal_achieved[i]
                                    success_rate_by_distance[
                                        index_1, bin - 1,
                                        1] += collision_observed[i]
                                    success_rate_by_distance[index_1, bin - 1,
                                                             2] += 1
                            # compute ADE
                            for agent_id, traj in rollout_traj_dict.items():
                                masking_arr = traj.sum(axis=1)
                                mask = (masking_arr != 0.0) * (
                                    masking_arr != traj.shape[1] * ERR_VAL)
                                expert_mask_arr = expert_trajectory_dict[
                                    agent_id].sum(axis=1)
                                expert_mask = (expert_mask_arr != 0.0) * (
                                    expert_mask_arr != traj.shape[1] * ERR_VAL)
                                ade = np.linalg.norm(
                                    traj - expert_trajectory_dict[agent_id],
                                    axis=-1)[mask * expert_mask]
                                average_displacement_error = (
                                    veh_counter * average_displacement_error +
                                    np.mean(ade)) / (veh_counter + 1)
                                fde = np.linalg.norm(
                                    traj - expert_trajectory_dict[agent_id],
                                    axis=-1)[np.max(
                                        np.argwhere(mask * expert_mask))]
                                final_displacement_error = (
                                    veh_counter * final_displacement_error +
                                    fde) / (veh_counter + 1)
                                veh_counter += 1

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
        if test_zsc:
            goal_array[index_1, index_2] = goal_frac
            collision_array[index_1, index_2] = collision_frac
            if index_1 == index_2:
                ade_array[index_1] = average_displacement_error
            if index_1 == index_2:
                fde_array[index_1] = final_displacement_error
        else:
            goal_array[index_1] = goal_frac
            collision_array[index_1] = collision_frac
            ade_array[index_1] = average_displacement_error
            fde_array[index_1] = final_displacement_error

    np.savetxt(os.path.join(output_path, 'zsc_goal.txt'),
               goal_array,
               delimiter=',')
    np.savetxt(os.path.join(output_path, 'zsc_collision.txt'),
               collision_array,
               delimiter=',')
    np.savetxt(os.path.join(output_path, 'ade.txt'), ade_array, delimiter=',')
    np.savetxt(os.path.join(output_path, 'fde.txt'), fde_array, delimiter=',')
    with open(os.path.join(output_path, 'success_by_veh_number.npy'),
              'wb') as f:
        if test_zsc:
            success_ratio = np.nan_to_num(
                success_rate_by_num_agents[:, :, :, 0:2] /
                success_rate_by_num_agents[:, :, :, [2]])
        else:
            success_ratio = np.nan_to_num(
                success_rate_by_num_agents[:, :, 0:2] /
                success_rate_by_num_agents[:, :, [2]])
        print(success_ratio)
        np.save(f, success_ratio)
    with open(os.path.join('success_by_dist.npy'), 'wb') as f:
        if test_zsc:
            dist_ratio = np.nan_to_num(success_rate_by_distance[:, :, :, 0:2] /
                                       success_rate_by_distance[:, :, :, [2]])
        else:
            dist_ratio = np.nan_to_num(success_rate_by_distance[:, :, 0:2] /
                                       success_rate_by_distance[:, :, [2]])
        print(dist_ratio)
        np.save(f, dist_ratio)

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    disp = Display()
    disp.start()
    register_custom_components()
    # output_folder = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.20/new_road_sample/18.32.35'
    output_folder = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.23/srt_v9/05.46.08'

    # class Bunch(object):

    #     def __init__(self, adict):
    #         self.__dict__.update(adict)

    # file_paths = []
    # cfg_dicts = []
    # for (dirpath, dirnames, filenames) in os.walk(output_folder):
    #     if 'cfg.json' in filenames:
    #         file_paths.append(dirpath)
    #         with open(os.path.join(dirpath, 'cfg.json'), 'r') as file:
    #             cfg_dict = json.load(file)

    #         cfg_dict['cli_args'] = {}
    #         cfg_dict['fps'] = 0
    #         cfg_dict['render_action_repeat'] = None
    #         cfg_dict['no_render'] = None
    #         cfg_dict['policy_index'] = 0
    #         cfg_dict['record_to'] = os.path.join(os.getcwd(), '..', 'recs')
    #         cfg_dict['continuous_actions_sample'] = True
    #         cfg_dict['discrete_actions_sample'] = True
    #         cfg_dict['num_eval_files'] = 2
    #         cfg_dicts.append(cfg_dict)

    # for file_path, cfg_dict in zip(file_paths, cfg_dicts):
    #     status, avg_reward = run_eval([Bunch(cfg_dict)],
    #                                   test_zsc=False,
    #                                   output_path=file_path)

    # okay, now build a pandas dataframe of the results that we will use for plotting
    # file_paths = []
    # data_dicts = []
    # for (dirpath, dirnames, filenames) in os.walk(output_folder):
    #     if 'cfg.json' in filenames:
    #         file_paths.append(dirpath)
    #         with open(os.path.join(dirpath, 'cfg.json'), 'r') as file:
    #             cfg_dict = json.load(file)
    #         goal = float(np.loadtxt(os.path.join(dirpath, 'zsc_goal.txt')))
    #         collide = float(
    #             np.loadtxt(os.path.join(dirpath, 'zsc_collision.txt')))
    #         data_dicts.append({
    #             'num_files': cfg_dict['num_files'],
    #             'goal_rate': goal,
    #             'collide_rate': collide
    #         })
    # df = pd.DataFrame(data_dicts)
    # means = df.groupby(['num_files'])['goal_rate'].mean()

    # load the wandb file
    # wandb_file = 'wandb_export_2022-05-23T15_44_41.104-04_00.csv'
    # with open(wandb_file, 'r') as f:
    #     wandb_df = pd.read_csv(f)
    # wandb_df['identifier'] = wandb_df['seed'].astype(
    #     str) + wandb_df['num_files'].astype(str)
    # import ipdb
    # ipdb.set_trace()
    # one_run = wandb_df[(wandb_df['seed'] == 0) & (wandb_df['num_files'] == 10)]
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # print(one_run)
    # sns.lineplot(data=one_run,
    #              x='global_step',
    #              y='0_aux/avg_goal_achieved',
    #              hue='num_files')
    # plt.savefig('test.png')
    #style='num_files')
    if not os.path.exists('wandb.csv'):
        import wandb

        api = wandb.Api()
        entity, project = "eugenevinitsky", "nocturne4"  # set to your entity and project
        runs = api.runs(entity + "/" + project)

        history_list = []
        for run in runs:
            if run.name == 'srt_v9':
                # # .summary contains the output keys/values for metrics like accuracy.
                # #  We call ._json_dict to omit large files
                # summary_list.append(run.summary._json_dict)

                # # .config contains the hyperparameters.
                # #  We remove special values that start with _.
                config = {
                    k: v
                    for k, v in run.config.items() if not k.startswith('_')
                }
                history_df = run.history()
                history_df['seed'] = config['seed']
                history_df['num_files'] = config['num_files']
                history_list.append(history_df)

            # # .name is the human-readable name of the run.
            # name_list.append(run.name)

        # runs_df = pd.DataFrame({
        #     "summary": summary_list,
        #     "config": config_list,
        #     "name": name_list
        # })
        runs_df = pd.concat(history_list)
        runs_df.to_csv('wandb.csv')

    wandb_df = pd.read_csv('wandb.csv')
    import seaborn as sns
    import matplotlib.pyplot as plt
    # print(one_run)
    sns.set_palette("PuBuGn_d")
    sns.lineplot(data=wandb_df,
                 x='global_step',
                 y='0_aux/avg_goal_achieved',
                 hue=wandb_df.num_files,
                 ci='sd',
                 palette=['r', 'g', 'b', 'm', 'k', 'c'])
    plt.savefig('test.png')


if __name__ == '__main__':
    sys.exit(main())
