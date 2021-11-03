"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from gym import spaces
import numpy as np

from envs.base_env import BaseEnv
from nocturne import Simulation
from utils.subscribers import Subscriber
from utils.waymo_scenario_construction import waymo_to_scenario, load_protobuf, get_actions_from_protobuf


class WaymoImitationEnv(BaseEnv):
    def __init__(self, protobuf_path, cfg):
        self.protobuf = load_protobuf(protobuf_path)
        waymo_to_scenario(cfg.scenario_path, self.protobuf)
        super(WaymoImitationEnv, self).__init__(cfg.scenario_path, cfg)

    def observation_space(self):
        pass

    def action_space(self):
        pass

    def step(self, action_dict):
        obs_dict = {}
        expert_actions = {}
        for veh_id, veh_obj in self.simulation.getVehicles():
            expert_actions[veh_id] = get_actions_from_protobuf(
                self.protobuf, veh_id, self.t)
        # either we use the expert or we roll out the actions returned by the imitator
        if self.cfg.dagger:
            self.apply_actions(action_dict)
        else:
            self.apply_actions(expert_actions)
        self.simulation.step(self.cfg.dt)
        self.t += self.cfg.dt
        for veh_id, veh_obj in self.simulation.getVehicles():
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
        return obs_dict, action_dict, expert_actions

    def reset(self):
        super().reset()
