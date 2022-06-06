"""Average displacement error computation."""
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
import random

from nocturne import Simulation
from cfgs.config import ERR_VAL as INVALID_POSITION
from multiprocessing import Pool
from itertools import repeat

SIM_N_STEPS = 90  # number of steps per trajectory


def _average_displacement_impl(arg):
    trajectory_path, model, configs = arg
    print(trajectory_path)

    scenario_config = configs['scenario_cfg']
    dataloader_config = configs['dataloader_cfg']

    view_dist = configs['dataloader_cfg']['view_dist']
    view_angle = configs['dataloader_cfg']['view_angle']
    state_normalization = configs['dataloader_cfg']['state_normalization']
    dt = configs['dataloader_cfg']['dt']

    n_stacked_states = configs['dataloader_cfg']['n_stacked_states']
    state_size = configs['model_cfg']['n_inputs'] // n_stacked_states
    state_dict = defaultdict(lambda: np.zeros(state_size * n_stacked_states))

    # create expert simulation
    sim_expert = Simulation(str(trajectory_path), scenario_config)
    scenario_expert = sim_expert.getScenario()
    vehicles_expert = scenario_expert.getVehicles()
    objects_expert = scenario_expert.getObjectsThatMoved()
    id2veh_expert = {veh.id: veh for veh in vehicles_expert}

    # create model simulation
    sim_model = Simulation(str(trajectory_path), scenario_config)
    scenario_model = sim_model.getScenario()
    vehicles_model = scenario_model.getVehicles()
    objects_model = scenario_model.getObjectsThatMoved()

    # set all objects to be expert-controlled
    for obj in objects_expert:
        obj.expert_control = True
    for obj in objects_model:
        obj.expert_control = True

    # in model sim, model will control vehicles that moved
    controlled_vehicles = [
        veh for veh in vehicles_model if veh in objects_model
    ]
    random.shuffle(controlled_vehicles)
    controlled_vehicles = controlled_vehicles[:2]

    # warmup to build up state stacking
    for i in range(n_stacked_states - 1):
        for veh in controlled_vehicles:
            ego_state = scenario_model.ego_state(veh)
            visible_state = scenario_model.flattened_visible_state(
                veh, view_dist=view_dist, view_angle=view_angle)
            state = np.concatenate(
                (ego_state, visible_state)) / state_normalization
            state_dict[veh.getID()] = np.roll(state_dict[veh.getID()],
                                              len(state))
            state_dict[veh.getID()][:len(state)] = state
        sim_model.step(dt)
        sim_expert.step(dt)

    for veh in controlled_vehicles:
        veh.expert_control = False

    avg_displacements = []
    collisions = [False for _ in controlled_vehicles]
    for i in range(SIM_N_STEPS - n_stacked_states):
        for veh in controlled_vehicles:
            if np.isclose(veh.position.x, -10000.0):
                veh.expert_control = True
            else:
                veh.expert_control = False
        # set model actions
        for veh in controlled_vehicles:
            # get vehicle state
            state = np.concatenate(
                (scenario_model.ego_state(veh),
                 scenario_model.flattened_visible_state(
                     veh, view_dist=view_dist,
                     view_angle=view_angle))) / state_normalization
            # stack state
            state_dict[veh.getID()] = np.roll(state_dict[veh.getID()],
                                              len(state))
            state_dict[veh.getID()][:len(state)] = state
            # compute vehicle action
            action = model(torch.unsqueeze(
                torch.as_tensor(state_dict[veh.getID()], dtype=torch.float32),
                0),
                           deterministic=True)
            # set vehicle action
            veh.acceleration = action[0].cpu().numpy()[0]
            veh.steering = action[1].cpu().numpy()[0]

        # step simulations
        sim_expert.step(dt)
        sim_model.step(dt)

        # compute displacements over non-collided vehicles
        displacements = []
        for i, veh in enumerate(controlled_vehicles):
            # get corresponding vehicle in expert simulation
            expert_veh = id2veh_expert[veh.id]
            # make sure it is valid
            if np.isclose(expert_veh.position.x,
                          -10000) or expert_veh.collided:
                continue
            # print(expert_veh.position, veh.position)
            # compute displacement
            expert_pos = id2veh_expert[veh.id].position
            model_pos = veh.position
            pos_diff = (model_pos - expert_pos).norm()
            displacements.append(pos_diff)
            # a collision with another a vehicle
            if veh.collided and int(veh.collision_type) == 1:
                collisions[i] = True

        # average displacements over all vehicles
        if len(displacements) > 0:
            avg_displacements.append(np.mean(displacements))
            # print(displacements, np.mean(displacements))

    # average displacements over all time steps
    avg_displacement = np.mean(
        avg_displacements) if len(avg_displacements) > 0 else np.nan
    avg_collisions = np.mean(collisions) if len(collisions) > 0 else np.nan
    print('displacements', avg_displacement)
    print('collisions', avg_collisions)
    return avg_displacement, avg_collisions


def compute_average_displacement(trajectories_dir, model, configs):
    """Compute average displacement error between a model and the ground truth."""
    # get trajectories paths
    if isinstance(trajectories_dir, str):
        # if trajectories_dir is a string, treat it as the path to a directory of trajectories
        trajectories_dir = Path(trajectories_dir)
        trajectories_paths = list(trajectories_dir.glob('*tfrecord*.json'))
        trajectories_paths.sort()
        trajectories_paths = trajectories_paths[:]
    elif isinstance(trajectories_dir, list):
        # if trajectories_dir is a list, treat it as a list of paths to trajectory files
        trajectories_paths = [Path(path) for path in trajectories_dir]
    # compute average displacement over each individual trajectory file
    trajectories_paths = trajectories_paths[:140]
    with Pool(processes=14) as pool:
        result = list(
            pool.map(_average_displacement_impl,
                     zip(trajectories_paths, repeat(model), repeat(configs))))
        average_displacements = np.array(result)[:, 0]
        average_collisions = np.array(result)[:, 1]
        print(average_displacements, average_collisions)

    return np.mean(
        average_displacements[~np.isnan(average_displacements)]), np.mean(
            average_collisions[~np.isnan(average_collisions)])


if __name__ == '__main__':
    from nocturne.utils.imitation_learning.model import ImitationAgent  # noqa: F401
    import json
    model = torch.load(
        '/checkpoint/eugenevinitsky/nocturne/test/2022.06.05/test/14.23.17/++device=cuda,++file_limit=1000/train_logs/2022_06_05_14_23_23/model_390.pth'
    ).to('cpu')
    model.actions_grids = [x.to('cpu') for x in model.actions_grids]
    model.eval()
    with open(
            '/checkpoint/eugenevinitsky/nocturne/test/2022.06.05/test/14.23.17/++device=cuda,++file_limit=1000/train_logs/2022_06_05_14_23_23/configs.json',
            'r') as fp:
        configs = json.load(fp)
        configs['device'] = 'cpu'
    with torch.no_grad():
        ade, collisions = compute_average_displacement(
            '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json_v2_no_tl_valid',
            model=model,
            configs=configs)
    print(f'Average Displacement Error: {ade:.3f} meters')
    print(f'Average Collisions: {collisions:.3f}\%')