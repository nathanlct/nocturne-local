"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from gym import spaces
import numpy as np

from nocturne import Simulation
from utils.subscribers import Subscriber


class GoalEnv(object):
    def __init__(self, scenario_path, cfg):
        self.simulation = Simulation(scenario_path)
        self.subscriber = Subscriber(cfg.subscriber)
        self.cfg = cfg

    def observation_space(self):
        pass

    def action_space(self):
        pass

    def step(self, action_dict):
        obs_dict = {}
        rew_dict = {}
        done_dict = {'__all__': False}
        info_dict = {}
        rew_cfg = self.cfg.rew_cfg
        for veh_id, veh_obj in self.simulation.getVehicles():
            if veh_id in action_dict.keys():
                action = action_dict[veh_id]
                if 'accel' in action.keys():
                    self.simulation.applyAccel(action['accel'])
                if 'turn' in action.keys():
                    self.simulation.applyTurn(action['turn'])
                if 'tilt_view' in action.keys():
                    self.simulation.applyViewTilt(action['tilt_view'])
        self.simulation.step(self.cfg.dt)
        for veh_id, veh_obj in self.simulation.getVehicles():
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            if self.simulation.hasCollided(veh_obj):
                rew_dict[veh_id] -= np.abs(rew_cfg.collision_penalty)
                done_dict[veh_id] = True
            if self.simulation.crossedLaneLines(veh_obj):
                rew_dict[veh_id] -= np.abs(rew_cfg.crossed_lanes_penalty)
            if self.simulation.goalAchieved(veh_obj):
                rew_dict[veh_id] += np.abs(rew_cfg.goal_achieved_bonus)

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        self.simulation.reset()
        obs_dict = {}
        for veh_id, veh_obj in self.simulation.getVehicles():
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
        return obs_dict
