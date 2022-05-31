"""Run a policy over the entire train set.

TODO(ev) refactor, this is wildly similar to visualize_sample_factory
"""

from copy import deepcopy
from collections import deque, defaultdict
import itertools
import json
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyvirtualdisplay import Display
import torch
import seaborn as sns

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

from cfgs.config import PROCESSED_VALID_NO_TL, PROCESSED_TRAIN_NO_TL, ERR_VAL


def run_eval(cfgs, test_zsc, output_path, scenario_dir, num_file_loops=1):
    """Eval a stored agent over all files in validation set.

    Args:
        cfg (dict): configuration file for instantiating the agents and environment.
        test_zsc (bool): if true, we play all agents against all agents
        num_file_loops (int): how many times to loop over the file set

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
    num_files = cfg['num_eval_files']
    if test_zsc:
        goal_array = np.zeros((len(actor_critics), len(actor_critics),
                               num_file_loops * num_files))
        collision_array = np.zeros((len(actor_critics), len(actor_critics),
                                    num_files * num_file_loops))
        success_rate_by_num_agents = np.zeros(
            (len(actor_critics), len(actor_critics), cfg.max_num_vehicles, 3))
        success_rate_by_distance = np.zeros(
            (len(actor_critics), len(actor_critics), distance_bins.shape[-1],
             3))
    else:
        goal_array = np.zeros((len(actor_critics), num_file_loops * num_files))
        collision_array = np.zeros(
            (len(actor_critics), num_file_loops * num_files))
        success_rate_by_num_agents = np.zeros(
            (len(actor_critics), cfg.max_num_vehicles, 3))
        success_rate_by_distance = np.zeros(
            (len(actor_critics), distance_bins.shape[-1], 3))
    ade_array = np.zeros(len(actor_critics))
    fde_array = np.zeros(len(actor_critics))

    with open(os.path.join(scenario_dir, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
        # sort the files so that we have a consistent order
        files = sorted(files)

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
        goal_frac = []
        collision_frac = []
        average_displacement_error = 0
        final_displacement_error = 0
        veh_counter = 0
        for loop_num in range(num_file_loops):
            for file_num, file in enumerate(files[0:cfg['num_eval_files']]):
                print('file is {}'.format(os.path.join(scenario_dir, file)))

                num_frames = 0
                env.unwrapped.files = [os.path.join(scenario_dir, file)]

                # step the env to its conclusion to generate the expert trajectories we compare against
                env.reset()
                expert_trajectory_dict = defaultdict(lambda: np.zeros((80, 2)))
                env.unwrapped.make_all_vehicles_experts()
                for i in range(80):
                    for veh in env.unwrapped.get_objects_that_moved():
                        expert_trajectory_dict[
                            veh.id][i] = veh.position.numpy()
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
                        np.random.choice(valid_indices,
                                         num_choice,
                                         replace=False))
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
                            obs_torch[key] = torch.from_numpy(x).to(
                                device).float()

                        # we have to make a copy before doing the pass
                        # because (for some reason), sample factory is making
                        # some changes to the obs in the forwards pass
                        # TBD what it is
                        if test_zsc:
                            obs_torch_2 = deepcopy(obs_torch)
                            policy_outputs_2 = actor_2(
                                obs_torch_2,
                                rnn_states_2,
                                with_action_distribution=True)

                        policy_outputs = actor_1(obs_torch,
                                                 rnn_states,
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
                                actions = policy_outputs[
                                    'action_logits'].argmax(axis=1)
                                if test_zsc:
                                    actions[indices_2] = policy_outputs_2[
                                        'action_logits'].argmax(
                                            axis=1)[indices_2]

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
                                    avg_true_rew = np.mean(
                                        true_rewards[agent_i])
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
                                avg_collisions = infos[0][
                                    'episode_extra_stats']['collided']
                                goal_frac.append(avg_goal)
                                collision_frac.append(avg_collisions)
                                if test_zsc:
                                    success_rate_by_num_agents[
                                        index_1, index_2,
                                        len(valid_indices) - 1, 0] += avg_goal
                                    success_rate_by_num_agents[
                                        index_1, index_2,
                                        len(valid_indices) - 1,
                                        1] += avg_collisions
                                    success_rate_by_num_agents[
                                        index_1, index_2,
                                        len(valid_indices) - 1, 2] += 1
                                else:
                                    success_rate_by_num_agents[
                                        index_1,
                                        len(valid_indices) - 1, 0] += avg_goal
                                    success_rate_by_num_agents[
                                        index_1,
                                        len(valid_indices) - 1,
                                        1] += avg_collisions
                                    success_rate_by_num_agents[
                                        index_1,
                                        len(valid_indices) - 1, 2] += 1
                                # track how well we do as a function of distance
                                for i, index in enumerate(valid_indices):
                                    env_id = agent_id_to_env_id_map[index]
                                    bin = np.searchsorted(
                                        distance_bins, goal_dist[env_id])
                                    if test_zsc:
                                        success_rate_by_distance[
                                            index_1, index_2, bin - 1,
                                            0] += goal_achieved[i]
                                        success_rate_by_distance[
                                            index_1, index_2, bin - 1,
                                            1] += collision_observed[i]
                                        success_rate_by_distance[index_1,
                                                                 index_2,
                                                                 bin - 1,
                                                                 2] += 1
                                    else:
                                        success_rate_by_distance[
                                            index_1, bin - 1,
                                            0] += goal_achieved[i]
                                        success_rate_by_distance[
                                            index_1, bin - 1,
                                            1] += collision_observed[i]
                                        success_rate_by_distance[index_1,
                                                                 bin - 1,
                                                                 2] += 1
                                # compute ADE and FDE
                                for agent_id, traj in rollout_traj_dict.items(
                                ):
                                    masking_arr = traj.sum(axis=1)
                                    mask = (masking_arr != 0.0) * (
                                        masking_arr != traj.shape[1] * ERR_VAL)
                                    expert_mask_arr = expert_trajectory_dict[
                                        agent_id].sum(axis=1)
                                    expert_mask = (expert_mask_arr != 0.0) * (
                                        expert_mask_arr !=
                                        traj.shape[1] * ERR_VAL)
                                    ade = np.linalg.norm(
                                        traj -
                                        expert_trajectory_dict[agent_id],
                                        axis=-1)[mask * expert_mask]
                                    average_displacement_error = (
                                        veh_counter *
                                        average_displacement_error +
                                        np.mean(ade)) / (veh_counter + 1)
                                    fde = np.linalg.norm(
                                        traj -
                                        expert_trajectory_dict[agent_id],
                                        axis=-1)[np.max(
                                            np.argwhere(mask * expert_mask))]
                                    final_displacement_error = (
                                        veh_counter * final_displacement_error
                                        + fde) / (veh_counter + 1)
                                    veh_counter += 1

                                # do some logging
                                log.info(
                                    f'Avg goal achieved {np.mean(goal_frac)}±{np.std(goal_frac)}'
                                )
                                log.info(
                                    f'Avg num collisions {np.mean(collision_frac)}±{np.std(collision_frac)}'
                                )
                                log.info(
                                    'Avg episode rewards: %s, true rewards: %s',
                                    avg_episode_rewards_str,
                                    avg_true_reward_str)
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

    np.save(os.path.join(output_path, 'zsc_goal.npy'), goal_array)
    np.save(os.path.join(output_path, 'zsc_collision.npy'), collision_array)
    np.save(os.path.join(output_path, 'ade.npy'), ade_array)
    np.save(os.path.join(output_path, 'fde.npy'), fde_array)
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


def load_wandb(experiment_name, force_reload=False):
    if not os.path.exists(
            'wandb_{}.csv'.format(experiment_name)) or force_reload:
        import wandb

        api = wandb.Api()
        entity, project = "eugenevinitsky", "nocturne4"  # set to your entity and project
        runs = api.runs(entity + "/" + project)

        history_list = []
        for run in runs:
            if run.name == experiment_name:

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

        runs_df = pd.concat(history_list)
        runs_df.to_csv('wandb_{}.csv'.format(experiment_name))


def plot_df(experiment_name):
    from matplotlib import pyplot as plt
    plt.figure()
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.figsize'] = (6, 4)

    df = pd.read_csv("wandb_{}.csv".format(experiment_name))
    df["timestamp"] = pd.to_datetime(df["_timestamp"] * 1e9)
    all_timestamps = np.sort(np.unique(df.timestamp.values))

    # technically not correct if the number of seeds varies by num_files
    num_seeds = len(np.unique(df.seed.values))

    values_num_files = np.unique(df.num_files.values)
    column = "0_aux/avg_goal_achieved"
    dfs = []
    stdevs = []
    for num_files in values_num_files:
        if num_files == 1:
            continue

        df_n = df[df.num_files == num_files].set_index('_step').sort_index()
        dfs.append(df_n[column].ewm(
            halflife=500,
            min_periods=10).mean().rename(f"num_files={num_files}"))
        stdevs.append(df_n[column].ewm(halflife=500, min_periods=10).std())

    ax = plt.gca()
    for i in range(len(dfs)):
        x = dfs[i].index.values
        y = dfs[i].values
        yerr = stdevs[i].replace(np.nan, 0) / np.sqrt(num_seeds)
        p = ax.plot(x, y, label=dfs[i].name)
        color = p[0].get_color()
        ax.fill_between(x, y - 2 * yerr, y + 2 * yerr, color=color, alpha=0.3)
    plt.grid(ls='--', color='#ccc')
    plt.legend()
    plt.xlabel("gradient step")
    plt.ylabel(column)
    plt.savefig('goal_achieved_v_gradient')


def eval_generalization(output_folder,
                        num_eval_files,
                        scenario_dir,
                        num_file_loops,
                        test_zsc=False,
                        cfg_filter=None):

    class Bunch(object):

        def __init__(self, adict):
            self.__dict__.update(adict)

    file_paths = []
    cfg_dicts = []
    for (dirpath, dirnames, filenames) in os.walk(output_folder):
        if 'cfg.json' in filenames:
            file_paths.append(dirpath)
            with open(os.path.join(dirpath, 'cfg.json'), 'r') as file:
                cfg_dict = json.load(file)

            if cfg_filter is not None and not cfg_filter(cfg_dict):
                continue

            cfg_dict['cli_args'] = {}
            cfg_dict['fps'] = 0
            cfg_dict['render_action_repeat'] = None
            cfg_dict['no_render'] = None
            cfg_dict['policy_index'] = 0
            cfg_dict['record_to'] = os.path.join(os.getcwd(), '..', 'recs')
            cfg_dict['continuous_actions_sample'] = False
            cfg_dict['discrete_actions_sample'] = False
            cfg_dict['num_eval_files'] = num_eval_files
            cfg_dicts.append(cfg_dict)
    if test_zsc:
        # TODO(eugenevinitsky) we're currently storing the ZSC result in a random
        # folder which seems bad.
        status, avg_reward = run_eval(
            [Bunch(cfg_dict) for cfg_dict in cfg_dicts],
            test_zsc=test_zsc,
            output_path=file_paths[0],
            scenario_dir=scenario_dir,
            num_file_loops=num_file_loops)
        print('stored ZSC result in {}'.format(file_paths[0]))
    else:
        for file_path, cfg_dict in zip(file_paths, cfg_dicts):
            status, avg_reward = run_eval([Bunch(cfg_dict)],
                                          test_zsc=test_zsc,
                                          output_path=file_path,
                                          scenario_dir=scenario_dir,
                                          num_file_loops=num_file_loops)


def main():
    """Script entry point."""
    disp = Display()
    disp.start()
    register_custom_components()
    RUN_EVAL = True
    TEST_ZSC = True
    PLOT_RESULTS = False
    RELOAD_WANDB = False
    NUM_EVAL_FILES = 10
    NUM_FILE_LOOPS = 3  # the number of times to loop over a fixed set of files
    experiment_names = ['srt_v12']
    # output_folder = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.20/new_road_sample/18.32.35'
    # output_folder = [
    #     '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.23/srt_v10/17.02.40/'
    # ]
    # 10 files
    output_folder = [
        '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/'
    ]
    # 100 files
    output_folder = [
        '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/'
    ]
    generalization_dfs = []

    def cfg_filter(cfg_dict):
        if cfg_dict['num_files'] == 10:
            return True
        else:
            return False

    '''
    ###############################################################################
    #########           Build the generalization dataframes ######################
    ##############################################################################
    '''
    if RUN_EVAL:
        for folder in output_folder:
            eval_generalization(folder,
                                NUM_EVAL_FILES,
                                FILES,
                                num_file_loops=NUM_FILE_LOOPS,
                                test_zsc=TEST_ZSC,
                                cfg_filter=cfg_filter)

    if PLOT_RESULTS:
        # okay, now build a pandas dataframe of the results that we will use for plotting
        # the generalization results
        for folder in output_folder:
            file_paths = []
            data_dicts = []
            for (dirpath, dirnames, filenames) in os.walk(folder):
                if 'cfg.json' in filenames:
                    file_paths.append(dirpath)
                    with open(os.path.join(dirpath, 'cfg.json'), 'r') as file:
                        cfg_dict = json.load(file)
                    goal = np.mean(
                        np.load(os.path.join(dirpath, 'zsc_goal.npy')))
                    collide = np.mean(
                        np.load(os.path.join(dirpath, 'zsc_collision.npy')))
                    ade = np.mean(np.load(os.path.join(dirpath, 'ade.npy')))
                    fde = np.mean(np.load(os.path.join(dirpath, 'fde.npy')))
                    num_files = cfg_dict['num_files']
                    if int(num_files) == -1:
                        num_files = 134453
                    if int(num_files) == 1:
                        continue
                    data_dicts.append({
                        'num_files': num_files,
                        'goal_rate': goal,
                        'collide_rate': collide,
                        'ade': ade,
                        'fde': fde
                    })
            df = pd.DataFrame(data_dicts)
            goals = df.groupby(['num_files'])['goal_rate'].mean().reset_index()
            ade = df.groupby(['num_files'])['ade'].mean().reset_index()
            fde = df.groupby(['num_files'])['fde'].mean().reset_index()
            collide = df.groupby(['num_files'
                                  ])['collide_rate'].mean().reset_index()
            goals = goals.merge(ade, how='inner', on='num_files')
            goals = goals.merge(fde, how='inner', on='num_files')
            goals = goals.merge(collide, how='inner', on='num_files')
            generalization_dfs.append(goals)
            '''
        ###############################################################################
        #########  load the training dataframes from wandb ######################
        ##############################################################################
        '''
        training_dfs = []
        for experiment_name in experiment_names:
            load_wandb(experiment_name, force_reload=RELOAD_WANDB)
            training_dfs.append(
                pd.read_csv('wandb_{}.csv'.format(experiment_name)))

        # create the goal plot
        plt.figure()
        for df in generalization_dfs:
            sns.lineplot(x=np.log(df.num_files), y=df.goal_rate)

        for df in training_dfs:
            values_num_files = np.unique(df.num_files.values)
            column = "0_aux/avg_goal_achieved"
            dfs = []
            y_vals = []
            x_vals = []
            for num_files in values_num_files:
                if num_files == 1:
                    continue
                df_n = df[df.num_files == num_files].set_index(
                    '_step').sort_index()
                dfs.append(df_n[column].ewm(
                    halflife=500,
                    min_periods=10).mean().rename(f"num_files={num_files}"))
                y_vals.append(dfs[-1].iloc[-1])
            values_num_files[np.argwhere(values_num_files == -1)] = 134453
            sns.lineplot(x=np.log(
                [value for value in values_num_files if value != 1]),
                         y=y_vals)
        plt.xlabel('log(number training files)')
        plt.ylabel('% goals achieved')
        plt.legend(['test', 'train'])
        plt.savefig('goal_achieved.png')

        # create the collide plot
        plt.figure()
        for df in generalization_dfs:
            sns.lineplot(x=np.log(df.num_files), y=df.collide_rate)

        for df in training_dfs:
            values_num_files = np.unique(df.num_files.values)
            column = "0_aux/avg_collided"
            dfs = []
            y_vals = []
            for num_files in values_num_files:
                df_n = df[df.num_files == num_files].set_index(
                    '_step').sort_index()
                dfs.append(df_n[column].ewm(
                    halflife=500,
                    min_periods=10).mean().rename(f"num_files={num_files}"))
                y_vals.append(dfs[-1].iloc[-1])
            values_num_files[np.argwhere(values_num_files == -1)] = 134453
            sns.lineplot(x=np.log(values_num_files), y=y_vals)
        plt.xlabel('log(number training files)')
        plt.ylabel('% vehicles collided')
        plt.legend(['test', 'train'])
        plt.savefig('collide_rate.png')

        # create ADE and FDE plots

        plt.figure()
        for df in generalization_dfs:
            sns.lineplot(x=np.log(df.num_files), y=df.ade)
        plt.xlabel('log(number training files)')
        plt.ylabel('average displacement error')
        # plt.legend(['test', 'train'])
        plt.savefig('ade.png')

        plt.figure()
        for df in generalization_dfs:
            sns.lineplot(x=np.log(df.num_files), y=df.ade)
        plt.xlabel('log(number training files)')
        plt.ylabel('final displacement error')
        # plt.legend(['test', 'train'])
        plt.savefig('fde.png')

        plot_df(experiment_names[0])


if __name__ == '__main__':
    sys.exit(main())
