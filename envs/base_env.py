"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from copy import copy
from collections import defaultdict
import os

import numpy as np
import torch

from nocturne import Simulation


class BaseEnv(object):

    def __init__(self, cfg, should_terminate=True, rank=0):
        """[summary]

        Args:
            cfg ([type]): configuration file describing the experiment
            should_terminate (bool, optional): if true, agents continue to receive a -1 vector as their observations
                even after their rollouts are terminated. This is used for algorithms (like some PPO implementations)
                that insist that the number of agents throughout an episode are consistent. 
            rank (int, optional): [description]. Defaults to 0.
        """
        self.files = os.listdir(cfg['scenario_path'])
        self.file = self.files[np.random.randint(len(self.files))]
        self.simulation = Simulation(os.path.join(cfg.scenario_path,
                                                  self.file),
                                     use_non_vehicles=False)
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.cfg = cfg
        self.episode_length = cfg.episode_length
        self.t = 0
        self.step_num = 0
        self.rank = rank
        # If true, agents observations stop being sent back once they're dead
        # If false, a vector of zeros will persist till the episode ends
        self.should_terminate = should_terminate
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        self.all_vehicle_ids = [veh.getID() for veh in self.vehicles]
        self.dead_feat = -np.ones_like(
            self.scenario.observation(
                self.vehicles[0],
                self.cfg.subscriber.view_dist,
                self.cfg.subscriber.view_angle,
            ))

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
        self.step_num += 1
        objs_to_remove = []
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = np.array(self.scenario.observation(
                veh_obj,
                self.cfg.subscriber.view_dist,
                self.cfg.subscriber.view_angle,
            ),
                                        copy=False)
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]['goal_achieved'] = False
            info_dict[veh_id]['collided'] = False
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            ######################## Compute rewards #######################
            # TODO(eugenevinitsky) this is never achieved because this is in meters but the goal tolerance is in pixels or something pixel-y
            if np.linalg.norm(goal_pos - obj_pos) < rew_cfg.goal_tolerance:
                rew_dict[veh_id] += np.abs(rew_cfg.goal_achieved_bonus)
                info_dict[veh_id]['goal_achieved'] = True
            if rew_cfg.shaped_goal_distance:
                # the minus one is to ensure that it's not beneficial to collide
                rew_dict[veh_id] -= ((np.linalg.norm(
                    (goal_pos - obj_pos) / (400 * 1.5), ord=2)**2) - 1)
            if rew_cfg.speed_rew:
                rew_dict[veh_id] += veh_obj.getSpeed() * rew_cfg.speed_rew_coef

            ######################## Handle potential done conditions #######################
            # achieved our goal
            if info_dict[veh_id]['goal_achieved']:
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                info_dict[veh_id]['collided'] = True
                rew_dict[veh_id] -= np.abs(rew_cfg.collision_penalty)
                done_dict[veh_id] = True
            # remove the vehicle so that its trajectory doesn't continue. This is important
            # in the multi-agent setting.
            if done_dict[veh_id]:
                objs_to_remove.append(veh_obj)

        for veh_obj in objs_to_remove:
            self.scenario.removeVehicle(veh_obj)

        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents and make sure they get something returned for PPO.
        if not self.should_terminate:
            for veh_id in self.all_vehicle_ids:
                if veh_id in obs_dict.keys():
                    continue
                else:
                    obs_dict[veh_id] = copy(self.dead_feat)
                    rew_dict[veh_id] = 0
                    done_dict[veh_id] = True
                    info_dict[veh_id] = {
                        'goal_achieved': False,
                        'collided': False
                    }

        if self.cfg.rew_cfg.shared_reward:
            total_reward = np.sum([rew_dict[key] for key in rew_dict.keys()])
            rew_dict = {key: total_reward for key in rew_dict.keys()}

        if self.step_num >= self.episode_length:
            done_dict = {key: True for key in done_dict.keys()}

        all_done = True
        for value in done_dict.values():
            all_done *= value
        done_dict['__all__'] = all_done

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        self.t = 0
        self.step_num = 0
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        self.file = self.files[np.random.randint(len(self.files))]
        self.simulation = Simulation(os.path.join(self.cfg.scenario_path,
                                                  self.file),
                                     use_non_vehicles=False)
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.all_vehicle_ids = [veh.getID() for veh in self.vehicles]

        obs_dict = {}
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = np.array(self.scenario.observation(
                veh_obj,
                self.cfg.subscriber.view_dist,
                self.cfg.subscriber.view_angle,
            ),
                                        copy=False)

        return obs_dict

    def render(self, mode=None):
        # TODO(eugenevinitsky) this should eventually return a global image instead of this hack
        return self.simulation.getScenario().getImage(None, render_goals=True)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
