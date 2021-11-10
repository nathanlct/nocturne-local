from collections import OrderedDict

from algos.gcsl.env_utils import ImageandProprio
from gym.spaces import Box, Discrete

import numpy as np

from envs.base_env import BaseEnv

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



class DictToVecWrapper(object):
    def __init__(self, env, no_img_concat=True, normalize_value=400):
        """Takes a dictionary state space and returns a flattened vector in the 'state' key of the dict

        Args:
            env ([type]): [description]
            no_img_concat (bool, optional): If true, we don't concat images into the 'state' key
        """
        self._env = env
        self._no_img_concat = no_img_concat
        self.normalize_value = normalize_value

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
                for val in next_obs.values():
                    if self._no_img_concat and len(val.shape) > 2:
                        continue
                    else:
                        features.append(val.ravel())
            obs_dict[agent_id]['features'] = np.concatenate(
                features, axis=0).astype(np.float32) / self.normalize_value
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
    def __init__(self, env, use_images=False):
        self._env = env
        self.goal_metric = 'euclidean'
        self.use_images = use_images
        if self.use_images:
            self.features_key = 'ego_image'
        else:
            self.features_key = 'features'
        self.initialized_obs_spaces = False
        self.reset()

    @property
    def action_space(self):
        return Box(low=np.array([-1, -0.4]), high=np.array([1, 0.4]))

    @property
    def observation_space(self):
        # TODO(eugenevinitsky) remove hack
        if self.use_images:
            return ImageandProprio((3, 84, 84), (self.feature_shape - 2, ))
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
        self.agent_key = list(obs_dict.keys())[0]
        for value in obs_dict.values():
            obs = value
        if not self.initialized_obs_spaces:
            self.initialized_obs_spaces = True
            self.feature_shape = obs['features'].shape[0]
            if self.use_images:
                print('Using Images')
                self.state_space = ImageandProprio((300, 300, 4),
                                                   obs['features'].shape)
            else:
                # TODO(eugenevinitsky) remove this once we actually vary the set of possible goals and sample instead of returning reset state
                self.curr_goal = obs['features']
                self.state_space = Box(low=-np.inf,
                                       high=np.inf,
                                       shape=obs['features'].shape)
            self.goal_space = Box(low=-np.inf,
                                  high=np.inf,
                                  shape=obs['goal_pos'].shape)
        if self.use_images:
            return self.state_space.to_flat(obs[self.features_key],
                                            obs['features'])
        else:
            return obs[self.features_key]

    def step(self, a):
        """Temporarily a single agent env, so we extract all the things from the wrappers. Assumes only one agent in env."""
        accel = a[0]
        steer = a[1]
        action_dict = {self.agent_key: {'accel': accel, 'turn': steer}}
        obs_dict, rew_dict, done_dict, info_dict = self._env.step(action_dict)
        for key in obs_dict.keys():
            obs = obs_dict[key]
            rew = rew_dict[key]
            done = done_dict[key]
        if self.use_images:
            return self.state_space.to_flat(
                obs[self.features_key], obs['features']), rew, done, info_dict
        else:
            return obs[self.features_key], rew, done, info_dict

    def observation(self, state):
        """
        Returns the observation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        if len(state.shape) > 1:
            return state[..., :-2]
        else:
            return state[:-2]

    def _extract_sgoal(self, state):
        if self.use_images:
            return self.state_space.from_flat(state)[1][-2:]
        else:
            return state[..., -2:]

    def extract_goal(self, state):
        """
        Returns the goal representation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        (TODO) no hardcoding
        """
        return state[..., -2:]

    def extract_achieved_goal(self, state):
        return state[..., 2:4]

    def goal_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            # TODO(eugenevinitsky) fix hardcoding
            if len(state.shape) > 1:
                achieved_goal = state[:, 2:4]
            else:
                achieved_goal = state[2:4]
            desired_goal = self.extract_goal(goal_state)
            diff = achieved_goal - desired_goal
            return np.linalg.norm(diff, axis=-1)
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)

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



def create_env(cfg):
    env = BaseEnv(cfg)
    return DictToVecWrapper(env)

def create_ppo_env(cfg):
    env = BaseEnv(cfg)
    env = DictToVecWrapper(env)
    return PPOWrapper(env)


def create_goal_env(cfg):
    env = BaseEnv(cfg)
    return GoalEnvWrapper(DictToVecWrapper(env))
