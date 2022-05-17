from gym.spaces import Box, Discrete
import numpy as np

from nocturne.envs import BaseEnv


class PPOWrapper(object):

    def __init__(self, env, use_images=False):
        """Wrapper that adds the appropriate observation spaces

        Args:
            env ([type]): [description]
            no_img_concat (bool, optional): If true, we don't concat images into the 'state' key
        """
        self._env = env
        self.use_images = use_images

        self.action_discretization = 5
        self.accel_grid = np.linspace(-1, 1, self.action_discretization)
        self.steering_grid = np.linspace(-.4, .4, self.action_discretization)

        # TODO(eugenevinitsky) this is a hack that assumes that we have a fixed number of agents
        self.n = len(self.vehicles)
        obs_dict = self.reset()
        # tracker used to match observations to actions
        self.agent_ids = []
        self.feature_shape = obs_dict[0].shape
        # TODO(eugenevinitsky) this is a hack that assumes that we have a fixed number of agents
        self.share_observation_space = [
            Box(low=-np.inf,
                high=+np.inf,
                shape=self.feature_shape,
                dtype=np.float32) for _ in range(self.n)
        ]

    @property
    def observation_space(self):
        return [
            Box(low=-np.inf, high=np.inf, shape=self.feature_shape)
            for _ in range(self.n)
        ]

    @property
    # TODO(eugenevinitsky) put back the box once we figure out how to make it compatible with the code
    def action_space(self):
        # return [Box(low=np.array([-1, -0.4]), high=np.array([1, 0.4]))]
        return [Discrete(self.action_discretization**2) for _ in range(self.n)]

    def step(self, actions):
        agent_actions = {}
        for action_vec, agent_id in zip(actions, self.agent_ids):
            # during training this is a one-hot vector, during eval this is the argmax
            if action_vec.shape[0] != 1:
                action = np.argmax(action_vec)
            else:
                action = action_vec[0]
            accel_action = self.accel_grid[int(action //
                                               self.action_discretization)]
            steering_action = self.steering_grid[action %
                                                 self.action_discretization]
            agent_actions[agent_id] = {
                'accel': accel_action,
                'turn': steering_action
            }
        next_obses, rew, done, info = self._env.step(agent_actions)
        obs_n = []
        rew_n = []
        done_n = []
        info_n = []
        # TODO(eugenevinitsky) I'm a little worried that there's going to be an order mismatch here
        for key in self.agent_ids:
            if isinstance(next_obses[key], dict):
                obs_n.append(next_obses[key]['features'])
            else:
                obs_n.append(next_obses[key])
            rew_n.append([rew[key]])
            done_n.append(done[key])
            agent_info = info[key]
            agent_info['individual_reward'] = rew[key]
            info_n.append(agent_info)
        return obs_n, rew_n, done_n, info_n

    def reset(self):
        obses = self._env.reset()
        # TODO(eugenevinitsky) I'm a little worried that there's going to be a key mismatch here
        # TODO(eugenevinitsky) this will break if the number of agents changes
        obs_n = []
        self.agent_ids = []
        for key in obses.keys():
            self.agent_ids.append(key)
            if not hasattr(self, 'agent_key'):
                self.agent_key = key
            if isinstance(obses[key], dict):
                obs_n.append(obses[key]['features'])
            else:
                obs_n.append(obses[key])
        return obs_n

    def render(self, mode=None):
        return self._env.render(mode)

    def seed(self, seed=None):
        self._env.seed(seed)

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

    def render(self, mode=None):
        return self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)

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
            if self.use_images:
                agent_dict = obs_dict[agent_id]
                agent_dict['features'] = np.concatenate(
                    (agent_dict['ego_img'], agent_dict['goal_img']), axis=-1)
                agent_dict['features'] = np.transpose(agent_dict['features'],
                                                      (2, 0, 1))
            else:
                if isinstance(next_obs, dict):
                    features = []
                    for key, val in next_obs.items():
                        if len(val.shape
                               ) > 2 or key == 'goal_pos' or key == 'ego_pos':
                            continue
                        else:
                            features.append(val.ravel())
                    # we want to make sure that the goal is at the end
                    # and that the "achieved_goal" is between -4 and -2 from the end
                    # TODO(eugenevinitsky) local coordinates break the goal wrappers
                    if 'ego_pos' in next_obs.keys(
                    ) and 'goal_pos' in next_obs.keys():
                        features.extend(
                            [next_obs['ego_pos'], next_obs['goal_pos']])
                agent_dict = obs_dict[agent_id]
                agent_dict['features'] = np.concatenate(
                    features, axis=0).astype(np.float32) / self.normalize_value
        return obs_dict

    def reset(self):
        obses = self._env.reset()
        self.transform_obs(obses)
        return obses

    def render(self, mode=None):
        return self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)


def create_env(cfg):
    env = BaseEnv(cfg)
    return env


def create_wrapped_env(cfg):
    env = BaseEnv(cfg)
    return ActionWrapper(DictToVecWrapper(env))


def create_ppo_env(cfg, rank=0):
    env = BaseEnv(cfg, rank=rank)
    return PPOWrapper(env, use_images=cfg.img_as_state)
