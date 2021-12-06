import time
import logging
import os
from numpy.core.fromnumeric import _nonzero_dispatcher

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import hydra
import numpy as np
import imageio

from algos.box_ddp import env_dx, util
from algos.box_ddp import mpc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from envs.base_env import BaseEnv

os.environ["DISPLAY"] = ":0.0"

class CarDx(nn.Module):
    def __init__(self, cfg, timesteps):
        super().__init__()
        # we need the agent position in the state so we that we can reset things properly
        # TODO(eugenevinitsky) there's a better way to do this
        cfg.subscriber.use_local_coordinates = False
        self.env = BaseEnv(cfg, should_terminate=False)
        self.num_vehicles = len(self.env.vehicles)
        self.dt = cfg.dt
        # TODO(eugenevinitsky) batch size hardcoding
        self.lower = torch.Tensor([-2, -0.4]).tile((timesteps,1)).unsqueeze(1)
        self.higher = torch.Tensor([1, 0.4]).tile((timesteps,1)).unsqueeze(1)

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

        # get the keys in the base env that we care about
        veh_id = self.env.vehicles[0].getID()
        state_dict = self.env.reset()[veh_id]
        # TODO(eugenevinitsky) extremely hardcoded
        self.dict_key_order = ['goal_dist', 'ego_pos', 'goal_pos', 'ego_speed', 'heading', 'visible_objects', 'road_objects']
        self.dict_key_order = [key for key in self.dict_key_order if key in state_dict.keys()]
        self.dict_shape = [state_dict[key].shape[0] if isinstance(state_dict[key], np.ndarray) else 1 for key in self.dict_key_order]
        # horrid hardcoding, get the indices from the subscriber
        start_pos = np.sum(self.dict_shape[0:5])
        if 'visible_objects' in self.dict_key_order:
            self.vehicle_dist_indices = np.arange(start_pos + 3, start_pos + self.dict_shape[-2], 5)
            self.vehicle_pos_indices = np.concatenate([[i, i + 1] for i in range(start_pos + 1, start_pos + self.dict_shape[-2], 5)])
            self.vehicle_heading_indices = np.arange(start_pos + 4, start_pos + self.dict_shape[-2], 5)
        start_pos = np.sum(self.dict_shape[0:-1])
        self.road_dist_indices = np.arange(start_pos + 4, start_pos + self.dict_shape[-1], 5)
        self.road_pos_indices = np.concatenate([[i, i + 1] for i in range(start_pos, start_pos + self.dict_shape[-1], 5)])
        self.road_heading_indices = np.arange(start_pos + 2, start_pos + self.dict_shape[-1], 5)
        self.goal_dist_index = 0
        self.ego_pos_index = [1, 2]
        self.goal_pos_index = [3, 4]
        self.ego_speed_index = [5]
        self.ego_heading_index = [6]


    def forward(self, x, u):
        # TODO(eugenevinitsky) remove the hardcoding of the first vehicle
        # vehicle = self.env.vehicles[0]
        # veh_id = vehicle.getID()
        # state_dict = self.state_to_dict(x)
        # # set the vehicle position to the first element of x
        # veh_pos = state_dict['ego_pos'][0].detach().numpy()
        # vehicle.setPosition(veh_pos[0], veh_pos[1])
        # TODO(eugenevinitsky) batch size hardcoding
        # sync up the environment so that we get good rendering
        # _, _, _, _ = self.env.step({veh_id: {'accel': accel, 'turn': turn}})
        
        # okay now lets do all the updates for the environment in a differentiable way
        # first lets update speed and heading of the vehicle
        length = 20.0 # TODO(remove hardcoding)
        accel = u[:, 0]
        turn = u[:, 1]
        heading = x[:, self.ego_heading_index] * torch.pi / 180
        speed = x[:, self.ego_speed_index]
        slipAngle = torch.atan(torch.tan(turn) / 2.0)
        dHeading = speed * torch.sin(turn) / length
        dX = speed * torch.cos(heading + slipAngle)
        dY = speed * torch.sin(heading + slipAngle)
        # heading update
        heading_unit_vector = torch.zeros_like(x)
        heading_unit_vector[:, self.ego_heading_index] = 1
        x = x + heading_unit_vector * (dHeading * 180 / np.pi) * self.dt
        # speed updates
        speed_unit_vector = torch.zeros_like(x)
        speed_unit_vector[:, self.ego_speed_index] = 1
        x = x + speed_unit_vector * accel * self.dt
        # position update
        ego_pos_unit_vector = torch.zeros_like(x)
        # TODO(eugenevinitsky) lazy as heck don't do this lol
        ego_pos_unit_vector[:, self.ego_pos_index[0]] = dX * self.dt
        ego_pos_unit_vector[:, self.ego_pos_index[1]] = dY * self.dt
        x = x + ego_pos_unit_vector

        # now do the position updates for all the other vehicles
        # we assume that they are not going to change their trajectories
        # TODO(eugenevinitsky) vectorize
        if hasattr(self, 'vehicle_pos_indices'):
            for i in range(len(self.vehicle_pos_indices)):
                heading = x[:, self.vehicle_heading_indices[i]] * torch.pi / 180
                speed = x[:, self.vehicle_speed_indices[i]]
                dX = speed * torch.cos(heading + slipAngle)
                dY = speed * torch.sin(heading + slipAngle)
                veh_pos_unit_vector = torch.zeros_like(x)
                # TODO(eugenevinitsky) lazy as heck don't do this lol
                veh_pos_unit_vector[:, self.vehicle_pos_indices[2 * i]] = dX * self.dt
                veh_pos_unit_vector[:, self.vehicle_pos_indices[2 * i + 1]] = dY * self.dt
                x = x + veh_pos_unit_vector

        # now we recompute all the distances
        # goal dist
        new_goal_dist = torch.linalg.norm(x[:, self.ego_pos_index] - x[:, self.goal_pos_index])
        goal_dist_unit_vector = torch.zeros_like(x)
        goal_dist_unit_vector[:, self.goal_dist_index] = 1
        x = (1 - goal_dist_unit_vector) * x + goal_dist_unit_vector * new_goal_dist

        # dist to all the other vehicles
        if hasattr(self, 'vehicle_pos_indices'):
            for i in range(len(self.vehicle_pos_indices)):
                new_dist = torch.linalg.norm(x[:, self.ego_pos_index] - x[:, self.vehicle_pos_indices[2 * i: 2 * (i + 1)]])
                veh_dist_unit_vector = torch.zeros_like(x)
                veh_dist_unit_vector[:, self.vehicle_dist_indices[i]] = 1
                x = (1 - veh_dist_unit_vector) * x + veh_dist_unit_vector * new_dist
        
        # now do this for all the object distances, here we need to recompute the closest point
        # frequently
        # TODO(eugenevinitsky) there is a bug here, objects get further as we turn towards them
        for i in range(len(self.road_dist_indices)):
            # TODO(eugenevinitsky) remove the length hardcoding
            length = 360
            road_pos_center = x[0, self.road_pos_indices[2 * i: 2 * (i + 1)]]
            heading = x[:, self.road_heading_indices[i]] * np.pi / 180
            endpoint_1 = torch.tensor([road_pos_center[0] + length * torch.cos(heading) / 2,
                                       road_pos_center[1] + length * torch.sin(heading) / 2])
            endpoint_2 = torch.tensor([road_pos_center[0] - length * torch.cos(heading) / 2,
                                       road_pos_center[1] - length * torch.sin(heading) / 2])
            ego_pos =  x[0, self.ego_pos_index]
            # find the closest point on the line and project
            t = max(0, min(1, torch.dot(ego_pos - endpoint_1, endpoint_2 - endpoint_1) / (length **2)))
            projection = endpoint_1 + t * (endpoint_2 - endpoint_1)
            new_dist = torch.linalg.norm(projection - ego_pos)

            road_dist_unit_vector = torch.zeros_like(x)
            road_dist_unit_vector[:, self.road_dist_indices[i]] = 1
            # TODO(eugenevinitsky) fix to allow batch
            # new_dist = torch.abs(torch.dot(normal_vec, x[0, self.ego_pos_index]) + b)
            x = (1 - road_dist_unit_vector) * x + road_dist_unit_vector * new_dist

        # print(x[:, self.road_dist_indices])
        # import ipdb; ipdb.set_trace()
        return x

    def grad_input(self, x, u):
        '''Computed df / dx, df / du'''
        # we use non-analytic grads for now because prototyping
        pass

    def state_to_dict(self, x):
        # TODO(eugenevinitsky) this is extremely hardcoded
        step_index = 0
        state_dict = {}
        for i, key in enumerate(self.dict_key_order):
            state_dict[key] = x[:, step_index: step_index + self.dict_shape[i]]
            step_index = step_index + self.dict_shape[i]
        return state_dict

    def state_dict_to_vec(self, x_dict):
        # TODO(eugenevinitsky) hardcoding of batch 1
        return torch.cat([torch.Tensor(x_dict[key].ravel()) for key in self.dict_key_order]).unsqueeze(0)

    def np_state_dict_to_vec(self, x_dict):
        return np.concatenate([x_dict[key].ravel() for key in self.dict_key_order])

    @property
    def state(self):
        # TODO(eugenevinitsky) hardcoding
        veh_obj = self.env.vehicles[0]
        return self.np_state_dict_to_vec(self.env.subscriber.get_obs(veh_obj))

class VehicleCost(nn.Module):
    def __init__(self, road_dist_indices=None, veh_dist_indices=None):
        super().__init__()
        self.road_dist_indices = road_dist_indices
        self.veh_dist_indices = veh_dist_indices
        # TODO(no hardcoding)
        self.desired_distance = 13

    def forward(self, x):

        # add a quadratic cost around distance to goal
        # TODO(eugenevinitsky) handle batch sizes correctly
        dist_goal_vec = torch.zeros_like(x[0])
        dist_goal_vec[0] = 1
        Q = torch.diag(dist_goal_vec)
        cost = torch.matmul(x, torch.matmul(Q, x.T)) ** 0.5
        # now form the quadratic barrier around collisions with objects
        mask_vec = torch.zeros_like(x[0])
        if self.road_dist_indices is not None:
            mask_vec[self.road_dist_indices] = 1
        if self.veh_dist_indices is not None:
            mask_vec[self.veh_dist_indices] = 1
        # TODO(eugenevinitsky) actually shape the log barrier
        # TODO(eugenevinitsky) compute the second order approximation of this so I can use
        # quadcosts instead
        filtered_vec = torch.matmul(torch.diag(mask_vec), x.T)
        constraint_vec = -filtered_vec + self.desired_distance
        constraint_cost = torch.matmul(torch.diag(mask_vec), torch.exp(2 * constraint_vec))
        # print(filtered_vec[self.road_dist_indices], constraint_cost[self.road_dist_indices], constraint_cost.sum())
        cost = cost + constraint_cost.sum()

        # form a control cost
        control_cost_vec = torch.zeros_like(x[0])
        control_cost_vec[-2:] = 1.0
        Q = torch.diag(control_cost_vec)
        cost = cost + 0.001 * torch.matmul(x, torch.matmul(Q, x.T))
        # TODO(eugenevinitsky) remove hardcoding
        return cost[0]

    def grad_input(x):
        pass

@hydra.main(config_path='../../../cfgs/', config_name='config')
def main(cfg):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
    TIMESTEPS = 12  # T
    N_BATCH = 1
    LQR_ITER = 10
    dx = CarDx(cfg, TIMESTEPS)

    dx.env.reset()

    nx = np.sum(dx.dict_shape)
    nu = 2

    u_init = None
    render = True
    # retrain_after_iter = 50
    run_iter = 30
    imgs = []

    # TODO(eugenevinitsky) set up the goals here
    # we want to regulate goal dist to zero, while keeping distance from objects we could
    # crash into high

    # goal_weights = torch.tensor((1., 0.1))  # nx
    # goal_state = torch.tensor((0., 0.))  # nx
    # ctrl_penalty = 0.001
    # q = torch.cat((
    #     goal_weights,
    #     ctrl_penalty * torch.ones(nu)
    # ))  # nx + nu
    # px = -torch.sqrt(goal_weights) * goal_state
    # p = torch.cat((px, torch.zeros(nu)))
    # Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    # p = p.repeat(TIMESTEPS, N_BATCH, 1)
    # cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    if hasattr(dx, 'vehicle_dist_indices'):
        cost = VehicleCost(dx.road_dist_indices, dx.vehicle_dist_indices)
    else:
        cost = VehicleCost(dx.road_dist_indices)

    # run MPC
    total_reward = 0
    for i in range(run_iter):
        state = dx.state.copy()
        state = torch.tensor(state).view(1, -1)
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=dx.lower, u_upper=dx.higher, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False,  u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF,
                       verbose=1)

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, dx)
        # import ipdb; ipdb.set_trace()
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        elapsed = time.perf_counter() - command_start
        # TODO(eugenevinitsky) remove hardcoding
        action = action.detach().numpy()
        # print()
        # heads up, 0.4 turns you left, -0.4 turns you right
        s, r, _, _ = dx.env.step({8: {'accel': action[0, 0], 'turn': action[0, 1]}})
        total_reward += r[8]
        print(s[8]['road_objects'][dx.road_dist_indices - 7], r, action)
        if r[8] == 0:
            import ipdb; ipdb.set_trace()
            break
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r[8], elapsed)
        if render:
            img = dx.env.render()
            imgs.append(img)

    imageio.mimsave('/private/home/eugenevinitsky/Code/nocturne/algos/box_ddp/env_dx' + '/render.gif', imgs, duration=0.1)

    logger.info("Total reward %f", total_reward)

if __name__ == '__main__':
    main()