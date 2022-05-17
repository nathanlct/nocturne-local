"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from collections import defaultdict
import json
import os

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

    def __init__(self, cfg, rank=0):
        """[summary]

        Args:
            cfg ([type]): configuration file describing the experiment
            rank (int, optional): [description]. Defaults to 0.
        """
        super().__init__()
        self._skip_env_checking = True  # temporary fix for rllib env checking issue
        self.files = []
        with open(os.path.join(cfg['scenario_path'],
                               'valid_files.json')) as file:
            self.valid_veh_dict = json.load(file)
            self.files = list(self.valid_veh_dict.keys())
        if cfg['num_files'] != -1:
            self.files = self.files[0:cfg['num_files']]
        self.file = self.files[np.random.randint(len(self.files))]
        self.simulation = Simulation(os.path.join(cfg['scenario_path'],
                                                  self.file),
                                     allow_non_vehicles=False)
        self.scenario = self.simulation.getScenario()
        self.controlled_vehicles = self.scenario.getObjectsThatMoved()
        self.single_agent_mode = cfg['single_agent_mode']
        self.cfg = cfg
        self.seed(cfg['seed'])
        self.episode_length = cfg['episode_length']
        self.t = 0
        self.step_num = 0
        self.rank = rank
        self.seed(cfg['seed'])
        obs_dict = self.reset()
        self.observation_space = Box(low=-np.infty,
                                     high=np.infty,
                                     shape=(obs_dict[list(
                                         obs_dict.keys())[0]].shape[0], ))
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
                        veh_obj.acceleration = action['accel']
                    if 'turn' in action.keys():
                        veh_obj.steering = action['turn']
                elif isinstance(action, list) or isinstance(
                        action, np.ndarray):
                    veh_obj.acceleration = action[0]
                    veh_obj.steering = action[1]
                else:
                    accel_action = self.accel_grid[int(
                        action // self.steering_discretization)]
                    steering_action = self.steering_grid[
                        action % self.accel_discretization]
                    veh_obj.acceleration = accel_action
                    veh_obj.steering = steering_action

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
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            if self.single_agent_mode and veh_id != self.single_agent_obj.getID(
            ):
                continue
            if veh_id in self.done_ids:
                continue
            obs_dict[veh_id] = self.get_observation(veh_obj)
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]['goal_achieved'] = False
            info_dict[veh_id]['collided'] = False
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            '''############################################
                            Compute rewards
               ############################################'''
            if np.linalg.norm(goal_pos - obj_pos) < rew_cfg['goal_tolerance']:
                info_dict[veh_id]['goal_achieved'] = True
                rew_dict[veh_id] += rew_cfg['goal_achieved_bonus'] / rew_cfg[
                    'reward_scaling']
            if rew_cfg['shaped_goal_distance']:
                # penalize the agent for its distance from goal
                # we scale by goal_dist_normalizers to ensure that this value is always less than the penalty for
                # collision
                if rew_cfg['goal_distance_penalty']:
                    rew_dict[veh_id] -= (np.linalg.norm(
                        (goal_pos - obj_pos), ord=2) /
                                         self.goal_dist_normalizers[veh_id]
                                         ) / rew_cfg['reward_scaling']
                else:
                    # the minus one is to ensure that it's not beneficial to collide
                    # we divide by goal_achieved_bonus / episode_length to ensure that
                    # acquiring the maximum "get-close-to-goal" reward at every time-step is
                    # always less than just acquiring the goal reward once
                    # we also assume that vehicles are never more than 400 meters from their goal
                    # which makes sense as the episodes are 9 seconds long i.e. we'd have to go more than
                    # 40 m/s to get there
                    rew_dict[veh_id] += (1 - np.linalg.norm(
                        (goal_pos - obj_pos), ord=2) /
                                         self.goal_dist_normalizers[veh_id]
                                         ) / rew_cfg['reward_scaling']
            '''############################################
                    Handle potential done conditions
            ############################################'''
            # achieved our goal
            if info_dict[veh_id]['goal_achieved']:
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                info_dict[veh_id]['collided'] = True
                rew_dict[veh_id] -= np.abs(
                    rew_cfg['collision_penalty']) / rew_cfg['reward_scaling']
                done_dict[veh_id] = True
            # remove the vehicle so that its trajectory doesn't continue. This is important
            # in the multi-agent setting.
            if done_dict[veh_id]:
                self.done_ids.append(veh_id)
                objs_to_remove.append(veh_obj)

        for veh_obj in objs_to_remove:
            self.scenario.removeVehicle(veh_obj)

        if self.cfg['rew_cfg']['shared_reward']:
            total_reward = np.sum([rew_dict[key] for key in rew_dict.keys()])
            rew_dict = {key: total_reward for key in rew_dict.keys()}

        # fill in the missing observations if we should be doing so
        if self.cfg['subscriber']['keep_inactive_agents']:
            # force all vehicles done to be false since they should persist through the episode
            done_dict = {key: False for key in self.all_vehicle_ids}
            for key in self.all_vehicle_ids:
                if key not in obs_dict.keys():
                    obs_dict[key] = self.dead_feat
                    rew_dict[key] = 0.0
                    info_dict[key]['goal_achieved'] = False
                    info_dict[key]['collided'] = False

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
        enough_vehicles = False
        # we don't want to initialize scenes with more than N actors
        while not enough_vehicles:
            self.file = self.files[np.random.randint(len(self.files))]
            self.simulation = Simulation(os.path.join(
                self.cfg['scenario_path'], self.file),
                                         allow_non_vehicles=False)
            self.scenario = self.simulation.getScenario()

            # remove all the objects that are in collision or are already in goal dist
            for veh_obj in self.simulation.getScenario().getObjectsThatMoved():
                obj_pos = veh_obj.getPosition()
                obj_pos = np.array([obj_pos.x, obj_pos.y])
                goal_pos = veh_obj.getGoalPosition()
                goal_pos = np.array([goal_pos.x, goal_pos.y])
                '''############################################
                    Remove vehicles at goal
                ############################################'''
                norm = np.linalg.norm(goal_pos - obj_pos)
                if norm < self.cfg['rew_cfg'][
                        'goal_tolerance'] or veh_obj.getCollided():
                    self.scenario.removeVehicle(veh_obj)
                '''############################################
                    Set all vehicles with unachievable goals to be experts
                ############################################'''
                if self.file in self.valid_veh_dict and veh_obj.getID(
                ) in self.valid_veh_dict[self.file]:
                    veh_obj.expert_control = True
            '''############################################
                Pick out the vehicles that we are controlling
            ############################################'''
            # ensure that we have no more than max_num_vehicles are controlled
            temp_vehicles = self.scenario.getObjectsThatMoved()
            curr_index = 0
            self.controlled_vehicles = []
            for vehicle in temp_vehicles:
                # we don't want to include vehicles that had unachievable goals
                # as controlled vehicles
                if not vehicle.expert_control:
                    self.controlled_vehicles.append(vehicle)
                else:
                    curr_index += 1
                if curr_index > self.cfg['max_num_vehicles']:
                    break
            self.all_vehicle_ids = [
                veh.getID() for veh in self.controlled_vehicles
            ]
            # make all the vehicles that are in excess of max_num_vehicles controlled by an expert
            for veh in self.scenario.getObjectsThatMoved()[curr_index:]:
                veh.expert_control = True
            # check that we have at least one vehicle or if we have just one file, exit anyways
            # or else we might be stuck in an infinite loop
            if len(self.all_vehicle_ids) > 0 or len(self.files) == 1:
                enough_vehicles = True

        # for one reason or another (probably we had a file where all the agents achieved their goals)
        # we have no controlled vehicles
        if len(self.all_vehicle_ids) == 0:
            # just grab a vehicle even if it hasn't moved so that we have something
            # to return obs for even if it's not controlled
            self.controlled_vehicles = [self.scenario.getVehicles()[0]]
            self.all_vehicle_ids = [
                veh.getID() for veh in self.controlled_vehicles
            ]

        # step all the vehicles forward by one second and record their observations as context
        if self.single_agent_mode:
            self.context_dict = {self.single_agent_obj.getID(): []}
            self.single_agent_obj.expert_control = True
        else:
            self.context_dict = {
                veh.getID(): []
                for veh in self.scenario.getObjectsThatMoved()
            }
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = True
        for _ in range(10):
            if self.single_agent_mode:
                self.context_dict[self.single_agent_obj.getID()].append(
                    self.get_observation(self.single_agent_obj))
            else:
                for veh in self.scenario.getObjectsThatMoved():
                    self.context_dict[veh.getID()].append(
                        self.get_observation(veh))
            self.simulation.step(self.cfg['dt'])
        # now hand back control to our actual controllers
        if self.single_agent_mode:
            self.single_agent_obj.expert_control = False
        else:
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = False

        # construct the observations and goal normalizers
        obs_dict = {}
        self.goal_dist_normalizers = {}
        if self.single_agent_mode:
            objs_that_moved = self.simulation.getScenario(
            ).getObjectsThatMoved()
            self.single_agent_obj = objs_that_moved[np.random.randint(
                len(objs_that_moved))]
            # tag all vehicles except for the one you control as controlled by the expert
            for veh in self.scenario.getObjectsThatMoved():
                if veh.getID() != self.single_agent_obj.getID():
                    veh.expert_control = True
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            if self.single_agent_mode and veh_obj.getID(
            ) != self.single_agent_obj.getID():
                continue
            # store normalizers for each vehicle
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            dist = np.linalg.norm(obj_pos - goal_pos)
            self.goal_dist_normalizers[veh_id] = dist
            # compute the obs
            obs_dict[veh_id] = self.get_observation(veh_obj)

        self.done_ids = []
        self.dead_feat = -np.ones_like(obs_dict[list(obs_dict.keys())[0]])
        # we should return obs for the missing agents
        if self.cfg['subscriber']['keep_inactive_agents']:
            max_id = max([int(key) for key in obs_dict.keys()])
            num_missing_agents = max(
                0, self.cfg['max_num_vehicles'] - len(obs_dict))
            for i in range(num_missing_agents):
                obs_dict[max_id + i + 1] = self.dead_feat
            self.dead_agent_ids = [
                max_id + i + 1 for i in range(num_missing_agents)
            ]
            self.all_vehicle_ids = list(obs_dict.keys())
        else:
            self.dead_agent_ids = []
        return obs_dict

    def get_observation(self, veh_obj):
        ego_obs = self.scenario.ego_state(veh_obj)
        if self.cfg['subscriber']['use_ego_state'] and self.cfg['subscriber'][
                'use_observations']:
            obs = np.concatenate(
                (ego_obs,
                 self.scenario.flattened_visible_state(
                     veh_obj, self.cfg['subscriber']['view_dist'],
                     self.cfg['subscriber']['view_angle'])))
        elif self.cfg['subscriber']['use_ego_state'] and not self.cfg[
                'subscriber']['use_observations']:
            obs = ego_obs
        else:
            obs = self.scenario.flattened_visible_state(
                veh_obj, self.cfg['subscriber']['view_dist'],
                self.cfg['subscriber']['view_angle'])
        return obs

    def render(self, mode=None):
        return np.array(self.simulation.getScenario().getImage(
            None, render_goals=True),
                        copy=False)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
