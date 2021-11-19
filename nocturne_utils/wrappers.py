from collections import OrderedDict
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import index_tricks
import seaborn as sns
import wandb

from algos.gcsl.env_utils import ImageandProprio
from gym.spaces import Box, Discrete
from envs.base_env import BaseEnv
from nocturne_utils.density_estimators import RawKernelDensity

class PPOWrapper(object):
    def __init__(self, env):
        """Wrapper that adds the appropriate observation spaces

        Args:
            env ([type]): [description]
            no_img_concat (bool, optional): If true, we don't concat images into the 'state' key
        """
        self._env = env

        self.action_discretization = 5
        self.accel_grid = np.linspace(-1, 1, self.action_discretization)
        self.steering_grid = np.linspace(-.4, .4, self.action_discretization)

        # TODO(eugenevinitsky) this is a hack that assumes that we have a fixed number of agents
        self.n = len(self.vehicles)
        obs_dict = self.reset()
        # tracker used to match observations to actions
        self.agent_ids = []
        # TODO(eugenevinitsky this does not work if images are in the observation)

        self.feature_shape = obs_dict[0].shape[0]
        # TODO(eugenevinitsky) this is a hack that assumes that we have a fixed number of agents
        self.share_observation_space = [Box(
            low=-np.inf, high=+np.inf, shape=(self.feature_shape,), dtype=np.float32) for _ in range(self.n)]

    # TODO(eugenevinitsky this does not work if images are in the observation)
    @property
    def observation_space(self):
        return [Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.feature_shape,)) for _ in range(self.n)]
    @property
    # TODO(eugenevinitsky) put back the box once we figure out how to make it compatible with the code
    def action_space(self):
        # return [Box(low=np.array([-1, -0.4]), high=np.array([1, 0.4]))]
        return [Discrete(self.action_discretization ** 2) for _ in range(self.n)]

    def step(self, actions):
        agent_actions = {}
        for action, agent_id in zip(actions, self.agent_ids):
            one_hot = np.argmax(action)
            accel_action = self.accel_grid[int(one_hot // self.action_discretization)]
            steering_action = self.steering_grid[one_hot % self.action_discretization]
            agent_actions[agent_id] = {'accel': accel_action, 'turn': steering_action}
        next_obses, rew, done, info = self._env.step(agent_actions)
        obs_n = []
        rew_n = []
        done_n = []
        info_n = []
        self.agent_ids = []
        for key in next_obses.keys():
            self.agent_ids.append(key)
            obs_n.append(next_obses[key]['features'])
            rew_n.append([rew[key]])
            done_n.append(done[key])
            info = {'individual_reward': rew[key]}
            info_n.append(info)
        return obs_n, rew_n, done_n, info_n

    def reset(self):
        obses = self._env.reset()
        # TODO(eugenevinitsky) I'm a little worried that there's going to be a key mismatch here
        obs_n = []
        self.agent_ids = []
        for key in obses.keys():
            self.agent_ids.append(key)
            if not hasattr(self, 'agent_key'):
                self.agent_key = key
            obs_n.append(obses[key]['features'])
        return obs_n

    def render(self):
        return self._env.render()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionWrapper(object):
    '''Used to make sure actions conform to the expected format'''
    def __init__(self, env):
        self._env = env

    def step(self, action_dict):
        for key, action in action_dict.items():
            if isinstance(action, np.ndarray):
                new_action = {'accel': action[0], 'turn': action[1]}
                action_dict[key] = new_action
        obs_dict, rew_dict, done_dict, info_dict = self._env.step(action_dict)
        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        return self._env.reset()

    def render(self):
        return self._env.render()

    def __getattr__(self, name):
        return getattr(self._env, name)


class DictToVecWrapper(object):
    def __init__(self, env, use_images=False, normalize_value=400):
        """Takes a dictionary state space and returns a flattened vector in the 'state' key of the dict

        Args:
            env ([type]): [description]
            use_images (bool): If true, images are going to be used and need to be prepared in the dict
        """
        self._env = env
        self.normalize_value = normalize_value

        self.use_images = use_images
        if self.use_images:
            self.features_key = 'ego_image'
        else:
            self.features_key = 'features'

        # snag an obs that we will use to create the observation spaces
        obs_dict = self.transform_obs(self._env.reset())
        example_obs = next(iter(obs_dict.values()))
        self.feature_shape = example_obs['features'].shape[0]
        self.goal_space = Box(low=-np.inf,
                                high=np.inf,
                                shape=example_obs['goal_pos'].shape)

        if self.use_images:
            print('Using Images')

            self.state_space = ImageandProprio((84, 84, 4),
                                                example_obs['features'].shape)
        else:
            self.state_space = Box(low=-np.inf,
                                    high=np.inf,
                                    shape=example_obs['features'].shape)
        
        # TODO(eugenevinitsky) remove this once we actually vary the set of possible goals and sample instead of returning reset state
        # TODO(eugenevinitsky) this is handled bespoke in every env, make this more general or a wrapper
        if self.use_images:
            self.curr_goal = {key: self.state_space.to_flat(obs[self.features_key], obs['features']) for key, obs in obs_dict.items()}
        else:
            self.curr_goal = {key: obs[self.features_key] for key, obs in obs_dict.items()}

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        next_obses, rew, done, info = self._env.step(action)
        self.transform_obs(next_obses)
        return next_obses, rew, done, info

    def transform_obs(self, obs_dict):
        # next_obs is a dict since this is a multi-agent system
        for agent_id, next_obs in obs_dict.items():
            if isinstance(next_obs, dict):
                features = []
                for key, val in next_obs.items():
                    if len(val.shape) > 2 or key == 'goal_pos' or key == 'ego_pos':
                        continue
                    else:
                        features.append(val.ravel())
                # we want to make sure that the goal is at the end
                # and that the "achieved_goal" is between -4 and -2 from the end
                features.extend([next_obs['ego_pos'], next_obs['goal_pos']])
            agent_dict = obs_dict[agent_id]
            agent_dict['features'] = np.concatenate(
                features, axis=0).astype(np.float32) / self.normalize_value
            if 'image' in self.features_key:
                agent_dict[self.features_key] = (cv2.resize(agent_dict[self.features_key], 
                                                (84, 84), interpolation = cv2.INTER_AREA) - 0.5) / 255.0
        return obs_dict

    def reset(self):
        obses = self._env.reset()
        self.transform_obs(obses)
        return obses

    def render(self):
        return self._env.render()

    def __getattr__(self, name):
        return getattr(self._env, name)


class GoalEnvWrapper(BaseEnv):
    """
    A GoalEnv is a modified Gym environment designed for goal-reaching tasks.
    One of the main deviations from the standard Gym abstraction is the separation
    of state from observation. The step() method always returns states, and
    observations can be obtained using a separate observation() method. 
    This change makes it easy to check for goal status, because whether a state
    reaches a goal is not always computable from the observation.
    The API consists of the following:
        GoalEnv.state_space
        GoalEnv.goal_space
        GoalEnv.reset()
            Resets the environment and returns the *state*
        GoalEnv.step(action)
            Runs 1 step of simulation and returns (state, 0, done, infos)
        GoalEnv.observation(state)
            Returns the observation for a given state
        GoalEnv.extract_goal(state)
            Returns the goal representation for a given state
    """
    def __init__(self, env):
        self._env = env
        self.goal_metric = 'euclidean'

    @property
    def action_space(self):
        return Box(low=np.array([-1, -0.4]), high=np.array([1, 0.4]))

    @property
    def observation_space(self):
        # TODO(eugenevinitsky) remove hack
        if self.use_images:
            return ImageandProprio((4, 84, 84), (self.feature_shape - 2, ))
        else:
            return Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.feature_shape - 2, ))

    def reset(self):
        """
        Resets the environment and returns a state vector
        Returns:
            The initial state
        """
        obs_dict = self._env.reset()
        return self.get_obs(obs_dict)

    def step(self, action_dict):

        obs_dict, rew_dict, done_dict, info_dict = self._env.step(action_dict)
        return self.get_obs(obs_dict), rew_dict, done_dict, info_dict

    def get_obs(self, obs_dict):
        if self.use_images:
            return {key: self.state_space.to_flat(obs[self.features_key], obs['features']) for key, obs in obs_dict.items()}
        else:
            return {key: obs[self.features_key] for key, obs in obs_dict.items()}


    def observation(self, states):
        """
        Returns the observation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        if isinstance(states, dict):
            new_dict = {}
            for key, state in states.items():
                if len(state.shape) > 1:
                    new_dict[key] = state[..., :-2]
                else:
                    new_dict[key] = state[:-2]
            return new_dict
        else:
            return states[..., :-2]

    def _extract_sgoal(self, states):
        '''This method is used to get the goal shape and so does not return a dict but a vector.'''
        # if self.use_images:
        #     import ipdb; ipdb.set_trace()
        #     return self.state_space.from_flat(state)[1][-2:]
        # else:
        if isinstance(states, dict):
            return next(iter(states.values()))[..., -2:]
        else:
            return states[..., -2:]

    def extract_goal(self, state):
        """
        Returns the goal representation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        (TODO) no hardcoding
        """
        if isinstance(state, dict):
            return {key: state_val[..., -2:] for key, state_val in state.items()}
        else:
            return state[..., -2:]

    def extract_achieved_goal(self, states):
        # TODO(eugenevinitsky) hardcodiiiiiiiiing
        if isinstance(states, dict):
            return {key: state[..., -4:-2] for key, state in states.items()}
        else:
            return states[..., -4:-2]

    def goal_distance(self, state_dict, goal_state_dict):
        average_goal_dist = 0
        for key in state_dict.keys():
            state = state_dict[key][-1]
            goal_state = goal_state_dict[key]
            if self.goal_metric == 'euclidean':
                # TODO(eugenevinitsky) fix hardcoding
                if len(state.shape) > 1:
                    achieved_goal = state[:, -4:-2]
                else:
                    achieved_goal = state[-4:-2]
                desired_goal = self.extract_goal(goal_state)
                diff = achieved_goal - desired_goal
                average_goal_dist += np.linalg.norm(diff * self.normalize_value, axis=-1)
            else:
                raise ValueError('Unknown goal metric %s' % self.goal_metric)
        return average_goal_dist / len(state_dict)

    def sample_goal(self):
        # return self.goal_space.sample()
        # TODO(eugenevinitsky) remove this once we actually vary the set of possible goals
        return self.curr_goal

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Gets things to log
        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]
        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        distances = np.array([
            self.goal_distance(
                trajectories[i],
                np.tile(desired_goal_states[i], (trajectories.shape[1], 1)))
            for i in range(trajectories.shape[0])
        ])
        return OrderedDict([
            ('mean final l2 dist', np.mean(distances[:, -1])),
            ('median final l2 dist', np.median(distances[:, -1])),
        ])

    def render(self):
        return self._env.render()

    def __getattr__(self, name):
        return getattr(self._env, name)

class CurriculumGoalEnvWrapper(GoalEnvWrapper):
    '''Test a curriculum for a single agent to go to a goal that evolves to minimize log entropy of the goal distribution'''
    def __init__(self, env, density_optim_samples, num_goal_samples, log_figure, kernel, quartile_cutoff, wandb=False):
        """[summary]

        Args:
            env: the environment we are wrapping
            density_optim_samples (int): the number of samples that are drawn from updating the kernel density estimator
            num_goal_samples (int): how many samples we draw from the achieved goals to find a potential new goal
            log_figure (bool): whether the density estimators should be logging periodically 
            kernel (str): the kernel for the density estimator. Options such as gaussian, tophat, linear, etc.
            quartile_cutoff (float): if this is greater than zero, we will return the int(num_goal_samples * quartile_cutoff) 
                goal with lowest entropy instead of the lowest i.e. if this is .1 we will return the goal with the highest entropy
                from the set of goals in the 10% quartile of lowest entropy
        """
        super(CurriculumGoalEnvWrapper, self).__init__(env)
        self.achieved_goals = []
        self.sampled_goals = []
        self.num_episodes = 0
        self.num_goal_samples = num_goal_samples
        self.log_figure = log_figure
        self.wandb = wandb
        # discretization = 10

        # # construction of the goal space
        # y_goals = np.linspace(-180, -20, discretization)
        # vertical_goals = np.vstack((20 * np.ones(discretization), y_goals)).T
        # x_goals = np.linspace(20, 240, discretization)
        # horizontal_goals = np.vstack((x_goals, -20 * np.ones(discretization))).T
        # self.valid_goals = np.vstack((vertical_goals, horizontal_goals))
        # # the last column is a counter of how many times this goal has been achieved
        # self.valid_goals = np.hstack((self.valid_goals, np.zeros((self.valid_goals.shape[0], 1))))
        # self.current_goal_counter = 0
        # # how many times we have to achieve this goal before we increment the counter
        # self.goals_to_increment = 10
        self.achieved_during_episode = False
        self.rkd_is_ready = False

        # TODO(eugenevinitsky) make this not hardcoded
        vehicle_ids = [obj.getID() for obj in self._env.vehicles]
        self.density_estimators = {key: RawKernelDensity(samples=density_optim_samples, 
                                   log_figure=log_figure, kernel=kernel, quartile_cutoff=quartile_cutoff,
                                   wandb=self.cfg.wandb) for key in vehicle_ids}

    @property
    def action_space(self):
        return Box(low=np.array([-1, -0.4]), high=np.array([1, 0.4]))

    def step(self, action):
        obs, rew, done, info = super().step(action)
        for key, info_dict in info.items():
            if info_dict['goal_achieved']:
                if self.achieved_during_episode == False:
                    self.achieved_during_episode = True
                    print('goal achieved')
                # # # sample a new goal
                # vehicle_objs = self._env.vehicles
                # for vehicle_obj in vehicle_objs:
                #     if vehicle_obj.getID() == key:
                #         rkd = self.density_estimators[key]
                #         # TODO(eugenevinitsky) remove hardcoding
                #         if rkd.ready:
                #             new_goal = rkd.draw_min_sample(1000) * self.normalize_value
                #             vehicle_obj.setGoalPosition(new_goal[0], new_goal[1])

        for key, val in obs.items():
            if not done[key]:
                achieved_goal = self.extract_achieved_goal(val)
                rkd = self.density_estimators[key]
                rkd.add_sample(achieved_goal)
            # rkd._optimize()
        # print(f'time to optimize kernel {time.time() - t}')
        return obs, rew, done, info

    def reset(self):
        '''We sample a new goal position at each reset''' 
        # this is just some temp logging for debugging
        for vehicle in self._env.vehicles:
            achieved_goal = vehicle.getPosition()
            self.achieved_goals.append([achieved_goal.x, achieved_goal.y])
        if self.num_episodes % 200 == 0 and self.log_figure:
            fig = plt.figure()
            np_arr = np.array(self.achieved_goals)
            sns.scatterplot(np_arr[-500:, 0], np_arr[-500:, 1])
            plt.hlines(42, -400, 400)
            plt.hlines(-42, -400, 400)
            plt.vlines(42, -400, 400)
            plt.vlines(-42, -400, 400)
            plt.xlim([-400, 400])
            plt.ylim([-400, 400])
            if self.wandb:
                wandb.log({"final_goals": wandb.Image(fig)})
            else:
                plt.savefig('/private/home/eugenevinitsky/Code/nocturne/final_goals.png')
            plt.close(fig)

            if self.rkd_is_ready:
                fig = plt.figure()
                np_arr = np.array(self.sampled_goals)
                sns.scatterplot(np_arr[-500:, 0], np_arr[-500:, 1])
                plt.hlines(42, -400, 400)
                plt.hlines(-42, -400, 400)
                plt.vlines(42, -400, 400)
                plt.vlines(-42, -400, 400)
                plt.xlim([-400, 400])
                plt.ylim([-400, 400])
                if self.wandb:
                    wandb.log({"desired_goals": wandb.Image(fig)})
                else:
                    plt.savefig('/private/home/eugenevinitsky/Code/nocturne/desired_goals.png')
                plt.close(fig)

        obs = super().reset()
        self.achieved_during_episode = False
        # TODO(eugenevinitsky) this will not work for many agents
        # self.achieved_during_episode = False
        vehicle_objs = self._env.vehicles
        # sample over all past achieved goals + the newest goal
        # self.sampled_goal_index = np.random.randint(0, self.current_goal_counter + 1)
        # new_goal = self.valid_goals[np.random.randint(0, self.current_goal_counter + 1)]

        # sample either our current goal or the most recently achieved goal
        # self.sampled_goal_index = self.current_goal_counter
        # self.sampled_goal_index = max(np.random.choice([self.current_goal_counter, self.current_goal_counter - 1, 
        #                                                 self.current_goal_counter - 2]), 0)
        # new_goal = self.valid_goals[self.sampled_goal_index]
        # t = time.time()
        for vehicle_obj in vehicle_objs:
            veh_id = vehicle_obj.getID()
            rkd = self.density_estimators[veh_id]
            # TODO(eugenevinitsky) remove hardcoding
            rkd._optimize()
            if rkd.ready:
                self.rkd_is_ready = True
                new_goal = rkd.draw_min_sample(self.num_goal_samples) * self.normalize_value
                self.sampled_goals.append([new_goal[0], new_goal[1]])
                vehicle_obj.setGoalPosition(new_goal[0], new_goal[1])
                # print(f'new goal is {new_goal}')
                if (new_goal[0] < -42 and new_goal[1] > 42) or (new_goal[0] > 42 and new_goal[1] > 42) or \
                    (new_goal[0] < -42 and new_goal[1] < -42) or (new_goal[0] > 42 and new_goal[1] < -42):
                    print('the goal is not actually on the road, dangit.')
                    print(new_goal)
        # print(f'time to draw new goals {time.time() - t}')
        # TODO(eugenevinitsky) this is a hack since dict to vec wrapper expects a dict
        new_obs = {vehicle_obj.getID(): self.subscriber.get_obs(vehicle_obj) for vehicle_obj in vehicle_objs}
        features = self.transform_obs(new_obs)
        self.curr_goal = features
        
        self.num_episodes += 1
        return features

    def transform_obs(self, obs_dict):
         # TODO(eugenevinitsky) this is a hack since dict to vec wrapper expects a dict
        return self.get_obs(self._env.transform_obs(obs_dict))
    

def create_env(cfg):
    env = BaseEnv(cfg)
    return ActionWrapper(DictToVecWrapper(env))

def create_ppo_env(cfg):
    env = BaseEnv(cfg)
    env = DictToVecWrapper(env)
    return PPOWrapper(env)


def create_goal_env(cfg):
    env = BaseEnv(cfg)
    # return CurriculumGoalEnvWrapper(DictToVecWrapper(ActionWrapper(env), use_images=False))
    return CurriculumGoalEnvWrapper(DictToVecWrapper(ActionWrapper(env), use_images=False),
                                    density_optim_samples=cfg.algo.density_optim_samples,
                                    num_goal_samples=cfg.algo.num_goal_samples,
                                    log_figure=cfg.algo.log_figure,
                                    kernel=cfg.algo.kernel,
                                    quartile_cutoff=cfg.algo.quartile_cutoff,
                                    wandb=cfg.wandb)
