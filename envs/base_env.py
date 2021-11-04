"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from gym import spaces
import numpy as np

from nocturne import Simulation
from utils.subscribers import Subscriber


class BaseEnv(object):
    def __init__(self, cfg):
        
        self.simulation = Simulation(cfg.scenario_path)
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.subscriber = Subscriber(cfg.subscriber, self.scenario, self.simulation)
        self.cfg = cfg
        self.t = 0

    @property
    def observation_space(self):
        pass

    @property
    def action_space(self):
        pass

    def apply_actions(self, action_dict):
        for veh_obj in self.scenario.getVehicles():
            veh_id = veh_obj.getID()
            if veh_id in action_dict.keys():
                action = action_dict[veh_id]
                if 'accel' in action.keys():
                    veh_obj.setAccel(action['accel'])
                if 'turn' in action.keys():
                    veh_obj.setSteeringAngle(action['turn'])
                # if 'tilt_view' in action.keys():
                #     self.simulation.applyViewTilt(action['tilt_view'])

    def step(self, action_dict):
        obs_dict = {}
        rew_dict = {}
        done_dict = {'__all__': False}
        info_dict = {}
        rew_cfg = self.cfg.rew_cfg
        self.apply_actions(action_dict)
        self.simulation.step(self.cfg.dt)
        self.t += self.cfg.dt
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            if veh_obj.getCollided():
                rew_dict[veh_id] -= np.abs(rew_cfg.collision_penalty)
                done_dict[veh_id] = True
            # if self.simulation.crossedLaneLines(veh_obj):
            #     rew_dict[veh_id] -= np.abs(rew_cfg.crossed_lanes_penalty)
            # TODO(eugenevinitsky) 
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            if np.linalg.norm(goal_pos - obj_pos) < rew_cfg.goal_tolerance:
                rew_dict[veh_id] += np.abs(rew_cfg.goal_achieved_bonus)

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        self.t = 0
        # TODO(eugenevinitsky) remove this once there is a scenario reset method
        self.simulation = Simulation(self.cfg.scenario_path)
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.subscriber = Subscriber(self.cfg.subscriber, self.scenario, self.simulation)
        obs_dict = {}
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
        return obs_dict
