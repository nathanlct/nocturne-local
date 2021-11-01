import numpy as np

from envs.base_env import BaseEnv

class DictToVecWrapper(object):
    def __init__(self, env, no_img_concat=True):
        """Takes a dictionary state space and returns a flattened vector in the 'state' key of the dict

        Args:
            env ([type]): [description]
            no_img_concat (bool, optional): If true, we don't concat images into the 'state' key
        """
        self._env = env
        self._no_img_concat = no_img_concat

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
            obs_dict[agent_id]['features'] = np.concatenate(features, axis=0).astype(np.float32)
        return obs_dict

    def reset(self):
        obses = self._env.reset()
        self.transform_obs(obses)
        return obses

def create_env(cfg):
    env = BaseEnv(cfg)
    return DictToVecWrapper(env)