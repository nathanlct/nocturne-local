"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from copy import copy
from collections import defaultdict

from gym.spaces import Box
import numpy as np

from nocturne import Simulation
from traitlets.traitlets import default
from nocturne_utils.subscribers import Subscriber


class BaseEnv(object):
    def __init__(self, cfg, should_terminate=True, rank=0):
        self.simulation = Simulation(cfg.scenario_path)
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.subscriber = Subscriber(cfg.subscriber, self.scenario, self.simulation)
        self.cfg = cfg
        self.t = 0
        self.rank=rank
        # If true, agents observations stop being sent back once they're dead
        # If false, a vector of zeros will persist till the episode ends
        self.should_terminate = should_terminate
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        self.all_vehicle_ids = [veh.getID() for veh in self.vehicles]
        self.dead_feat =  {key: -np.ones_like(value) for key, value in self.subscriber.get_obs(self.vehicles[0]).items()}

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
        done_dict = {}
        info_dict = defaultdict(dict)
        rew_cfg = self.cfg.rew_cfg
        self.apply_actions(action_dict)
        self.simulation.step(self.cfg.dt)
        self.t += self.cfg.dt
        objs_to_remove = []
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]['goal_achieved'] = False
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            # TODO(eugenevinitsky) this is never achieved because this is in meters but the goal tolerance is in pixels or something pixel-y
            if np.linalg.norm(goal_pos - obj_pos) < rew_cfg.goal_tolerance:
                rew_dict[veh_id] += np.abs(rew_cfg.goal_achieved_bonus)
                # TODO(eugenevinitsky) temporarily disabled the done condition as this crushes exploration around a goal
                # done_dict[veh_id] = True
                info_dict[veh_id]['goal_achieved'] = True
            if rew_cfg.shaped_goal_distance:
                # the minus one is to ensure that it's not beneficial to collide
                # rew_dict[veh_id] -= ((np.linalg.norm(goal_pos - obj_pos) / 2000) - 1)
                rew_dict[veh_id] -= ((np.linalg.norm(goal_pos - obj_pos) **2 / 801**2) - 1)
            ######################## Handle potential done conditions #######################
            # we have gone off-screen or off road!
            if not self.scenario.isVehicleOnRoad(veh_obj):
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                rew_dict[veh_id] -= np.abs(rew_cfg.collision_penalty)
                done_dict[veh_id] = True
            # remove the vehicle so that its trajectory doesn't continue. This is important
            # in the multi-agent setting.
            if done_dict[veh_id]:
                objs_to_remove.append(veh_obj)
        
        for veh_obj in objs_to_remove:
            self.scenario.removeObject(veh_obj)
        
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        if not self.should_terminate:
            for veh_id in self.all_vehicle_ids:
                if veh_id in obs_dict.keys():
                    continue
                else:
                    obs_dict[veh_id] = copy(self.dead_feat)
                    rew_dict[veh_id] = 0
                    done_dict[veh_id] = True
                    info_dict[veh_id] = {'goal_achieved': False}
        
        all_done = True
        for value in done_dict.values():
            all_done *= value
        done_dict['__all__'] = all_done

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        self.t = 0
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        self.simulation.reset()
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.all_vehicle_ids = [veh.getID() for veh in self.vehicles]
        self.subscriber = Subscriber(self.cfg.subscriber, self.scenario, self.simulation)
        obs_dict = {}
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
            veh_obj.setSpeed(self.cfg.initial_speed)
        return obs_dict

    def render(self, mode=None):
        # TODO(eugenevinitsky) this should eventually return a global image instead of this hack
        return np.array(self.scenario.getImage(object=None, renderGoals=True), copy=False)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)