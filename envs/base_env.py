"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from copy import copy
from collections import defaultdict
import os
import time

from gym.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import torch

from nocturne import Simulation


class BaseEnv(MultiAgentEnv):
    # class BaseEnv(object):
    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, cfg, should_terminate=True, rank=0):
        """[summary]

        Args:
            cfg ([type]): configuration file describing the experiment
            should_terminate (bool, optional): if true, agents continue to receive a -1 vector as their observations
                even after their rollouts are terminated. This is used for algorithms (like some PPO implementations)
                that insist that the number of agents throughout an episode are consistent. 
            rank (int, optional): [description]. Defaults to 0.
        """
        super().__init__()
        self._skip_env_checking = True  # temporary fix for rllib env checking issue
        self.files = os.listdir(cfg['scenario_path'])
        if cfg['num_files'] != -1:
            self.files = self.files[0:cfg['num_files']]
        self.file = self.files[np.random.randint(len(self.files))]
        self.simulation = Simulation(os.path.join(cfg['scenario_path'],
                                                  self.file),
                                     use_non_vehicles=False)
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getObjectsThatMoved()
        self.single_agent_mode = cfg['single_agent_mode']
        self.cfg = cfg
        self.episode_length = cfg['episode_length']
        self.t = 0
        self.step_num = 0
        self.rank = rank
        ego_obs = self.scenario.egoStateObservation(self.vehicles[0])
        if self.cfg['subscriber']['use_ego_state'] and self.cfg['subscriber'][
                'use_observations']:
            obs_shape = -np.ones_like(
                np.concatenate(
                    (ego_obs,
                     self.scenario.observation(
                         self.vehicles[0], self.cfg['subscriber']['view_dist'],
                         self.cfg['subscriber']['view_angle']))))
        elif self.cfg['subscriber']['use_ego_state'] and not self.cfg[
                'subscriber']['use_observations']:
            obs_shape = -np.ones_like(ego_obs)
        else:
            obs_shape = -np.ones_like(
                self.scenario.observation(
                    self.vehicles[0], self.cfg['subscriber']['view_dist'],
                    self.cfg['subscriber']['view_angle']))
        self.observation_space = Box(low=-np.infty,
                                     high=np.infty,
                                     shape=(obs_shape.shape[0], ))
        if self.cfg['discretize_actions']:
            self.accel_discretization = self.cfg['accel_discretization']
            self.steering_discretization = self.cfg['steering_discretization']
            self.action_space = Discrete(self.accel_discretization *
                                         self.steering_discretization)
            self.accel_grid = np.linspace(
                -np.abs(self.cfg['accel_lower_bound']),
                self.cfg['accel_upper_bound'], self.accel_discretization)
            self.steering_grid = np.linspace(
                -np.abs(self.cfg['steering_lower_bound']),
                self.cfg['steering_upper_bound'], self.steering_discretization)
        else:
            self.action_space = Box(low=-np.abs(self.cfg['accel_lower_bound']),
                                    high=self.cfg['accel_upper_bound'],
                                    shape=(2, ))

    def apply_actions(self, action_dict):
        for veh_obj in self.scenario.getObjectsThatMoved():
            veh_id = veh_obj.getID()
            if veh_id in action_dict.keys():
                action = action_dict[veh_id]
                if isinstance(action, dict):
                    if 'accel' in action.keys():
                        veh_obj.setAccel(action['accel'])
                    if 'turn' in action.keys():
                        veh_obj.setSteeringAngle(action['turn'])
                elif isinstance(action, list) or isinstance(
                        action, np.ndarray):
                    veh_obj.setAccel(action[0])
                    veh_obj.setSteeringAngle(action[1])
                else:
                    accel_action = self.accel_grid[int(
                        action // self.steering_discretization)]
                    steering_action = self.steering_grid[
                        action % self.accel_discretization]
                    veh_obj.setAccel(accel_action)
                    veh_obj.setSteeringAngle(steering_action)

    def step(self, action_dict):
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        info_dict = defaultdict(dict)
        rew_cfg = self.cfg['rew_cfg']
        self.apply_actions(action_dict)
        self.simulation.step(self.cfg['dt'])
        self.t += self.cfg['dt']
        self.step_num += 1
        objs_to_remove = []
        t = time.time()
        for veh_obj in self.simulation.getScenario().getObjectsThatMoved():
            veh_id = veh_obj.getID()
            if self.single_agent_mode and veh_id != self.single_agent_id:
                continue
            if self.cfg['subscriber']['use_ego_state'] and self.cfg[
                    'subscriber']['use_observations']:
                obs_dict[veh_id] = np.concatenate(
                    (self.scenario.egoStateObservation(veh_obj),
                     self.scenario.observation(
                         veh_obj, self.cfg['subscriber']['view_dist'],
                         self.cfg['subscriber']['view_angle'])))
            elif self.cfg['subscriber']['use_ego_state'] and not self.cfg[
                    'subscriber']['use_observations']:
                obs_dict[veh_id] = self.scenario.egoStateObservation(veh_obj)
            else:
                obs_dict[veh_id] = self.scenario.observation(
                    veh_obj, self.cfg['subscriber']['view_dist'],
                    self.cfg['subscriber']['view_angle'])
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]['goal_achieved'] = False
            info_dict[veh_id]['collided'] = False
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            ######################## Compute rewards #######################
            if np.linalg.norm(goal_pos - obj_pos) < rew_cfg['goal_tolerance']:
                info_dict[veh_id]['goal_achieved'] = True
                rew_dict[veh_id] += rew_cfg['goal_achieved_bonus']
            if rew_cfg['shaped_goal_distance']:
                # the minus one is to ensure that it's not beneficial to collide
                # we divide by goal_achieved_bonus / episode_length to ensure that
                # acquiring the maximum "get-close-to-goal" reward at every time-step is
                # always less than just acquiring the goal reward once
                # we also assume that vehicles are never more than 400 meters from their goal
                # which makes sense as the episodes are 9 seconds long i.e. we'd have to go more than
                # 40 m/s to get there
                rew_dict[veh_id] += (1 - np.linalg.norm(
                    (goal_pos - obj_pos), ord=2) /
                                     self.goal_dist_normalizers[veh_id])

            ######################## Handle potential done conditions #######################
            # achieved our goal
            if info_dict[veh_id]['goal_achieved']:
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                info_dict[veh_id]['collided'] = True
                rew_dict[veh_id] -= np.abs(rew_cfg['collision_penalty'])
                done_dict[veh_id] = True
            # remove the vehicle so that its trajectory doesn't continue. This is important
            # in the multi-agent setting.
            if done_dict[veh_id]:
                objs_to_remove.append(veh_obj)

        for veh_obj in objs_to_remove:
            self.scenario.removeVehicle(veh_obj)

        if self.cfg['rew_cfg']['shared_reward']:
            total_reward = np.sum([rew_dict[key] for key in rew_dict.keys()])
            rew_dict = {key: total_reward for key in rew_dict.keys()}

        if self.step_num >= self.episode_length:
            done_dict = {key: True for key in done_dict.keys()}

        all_done = True
        for value in done_dict.values():
            all_done *= value
        done_dict['__all__'] = all_done

        # for val in obs_dict.values():
        #     if np.any(np.isnan(val)):
        #         import ipdb
        #         ipdb.set_trace()

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        self.t = 0
        self.step_num = 0
        too_many_vehicles = True
        # we don't want to initialize scenes with more than N actors
        while too_many_vehicles:
            self.file = self.files[np.random.randint(len(self.files))]
            self.simulation = Simulation(os.path.join(
                self.cfg['scenario_path'], self.file),
                                         use_non_vehicles=False)
            self.scenario = self.simulation.getScenario()

            # remove all the objects that are in collision or are already in goal dist
            for veh_obj in self.simulation.getScenario().getObjectsThatMoved():
                obj_pos = veh_obj.getPosition()
                obj_pos = np.array([obj_pos.x, obj_pos.y])
                goal_pos = veh_obj.getGoalPosition()
                goal_pos = np.array([goal_pos.x, goal_pos.y])
                ######################## Compute rewards #######################
                if np.linalg.norm(goal_pos - obj_pos
                                  ) < self.cfg['rew_cfg']['goal_tolerance']:
                    self.scenario.removeVehicle(veh_obj)
                if veh_obj.getCollided():
                    self.scenario.removeVehicle(veh_obj)
            self.vehicles = self.scenario.getObjectsThatMoved()
            self.all_vehicle_ids = [veh.getID() for veh in self.vehicles]
            # check if we have less than the desired number of vehicles and have
            # at least one vehicle
            if len(self.all_vehicle_ids) <= self.cfg['max_num_vehicles'] \
                and len(self.all_vehicle_ids) > 0:
                too_many_vehicles = False

        obs_dict = {}
        self.goal_dist_normalizers = {}
        # in single agent mode, we always declare the same agent in each scene
        # the controlled agent to make the learning process simpler
        if self.single_agent_mode:
            objs_that_moved = self.simulation.getScenario(
            ).getObjectsThatMoved()
            self.single_agent_id = objs_that_moved[-1].getID()
            # tag all vehicles except for the one you control as controlled by the expert
            for veh in self.simulation.getScenario().getVehicles():
                if veh.getID() != self.single_agent_id:
                    veh.expert_control = True
        for veh_obj in self.simulation.getScenario().getObjectsThatMoved():
            veh_id = veh_obj.getID()
            if self.single_agent_mode and veh_id != self.single_agent_id:
                continue
            # store normalizers for each vehicle
            if self.cfg['rew_cfg']['shaped_goal_distance']:
                obj_pos = veh_obj.getPosition()
                obj_pos = np.array([obj_pos.x, obj_pos.y])
                goal_pos = veh_obj.getGoalPosition()
                goal_pos = np.array([goal_pos.x, goal_pos.y])
                dist = np.linalg.norm(obj_pos - goal_pos)
                self.goal_dist_normalizers[veh_id] = dist
            # compute the obs
            ego_obs = self.scenario.egoStateObservation(veh_obj)
            if self.cfg['subscriber']['use_ego_state'] and self.cfg[
                    'subscriber']['use_observations']:
                obs_dict[veh_id] = np.concatenate(
                    (ego_obs,
                     self.scenario.observation(
                         veh_obj, self.cfg['subscriber']['view_dist'],
                         self.cfg['subscriber']['view_angle'])))
            elif self.cfg['subscriber']['use_ego_state'] and not self.cfg[
                    'subscriber']['use_observations']:
                obs_dict[veh_id] = ego_obs
            else:
                obs_dict[veh_id] = self.scenario.observation(
                    veh_obj, self.cfg['subscriber']['view_dist'],
                    self.cfg['subscriber']['view_angle'])
        return obs_dict

    def render(self, mode=None):
        return self.simulation.getScenario().getImage(None, render_goals=True)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
