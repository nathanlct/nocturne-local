"""Average displacement error computation."""
from pathlib import Path
import numpy as np
import torch

from nocturne import Simulation
from cfgs.config import ERR_VAL as INVALID_POSITION


SIM_N_STEPS = 90  # number of steps per trajectory
SIM_STEP_TIME = 0.1  # dt (in seconds)


def _average_displacement_impl(trajectory_path, model, sim_allow_non_vehicles=True):
    print(trajectory_path)

    # create expert simulation
    sim_expert = Simulation(scenario_path=str(trajectory_path), start_time=0, allow_non_vehicles=sim_allow_non_vehicles)
    scenario_expert = sim_expert.getScenario()
    vehicles_expert = scenario_expert.getVehicles()
    objects_expert = scenario_expert.getObjectsThatMoved()
    id2veh_expert = {veh.id: veh for veh in vehicles_expert}

    # create model simulation
    sim_model = Simulation(scenario_path=str(trajectory_path), start_time=0, allow_non_vehicles=sim_allow_non_vehicles)
    scenario_model = sim_model.getScenario()
    vehicles_model = scenario_model.getVehicles()
    objects_model = scenario_model.getObjectsThatMoved()

    # set all objects to be expert-controlled
    for obj in objects_expert:
        obj.expert_control = True
    for obj in objects_model:
        obj.expert_control = True

    # in model sim, model will control vehicles that moved
    controlled_vehicles = [veh for veh in vehicles_model if veh in objects_model]
    for veh in controlled_vehicles:
        veh.expert_control = False

    avg_displacements = []
    for i in range(SIM_N_STEPS):
        # set model actions
        for veh in controlled_vehicles:
            # get vehicle state
            state = torch.as_tensor(np.expand_dims(np.concatenate(
                (scenario_model.ego_state(veh),
                 scenario_model.flattened_visible_state(veh, view_dist=120, view_angle=3.14))
            ), axis=0), dtype=torch.float32)
            # compute vehicle action
            action = model(state)[0]
            # set vehicle action
            veh.acceleration = action[0]
            veh.steering = action[1]

        # step simulations
        sim_expert.step(SIM_STEP_TIME)
        sim_model.step(SIM_STEP_TIME)

        # compute displacements over non-collided vehicles
        displacements = []
        for veh in filter(lambda veh: not veh.collided, controlled_vehicles):
            # get corresponding vehicle in expert simulation
            expert_veh = id2veh_expert[veh.id]
            # make sure it is valid
            if np.isclose(expert_veh.position.x, INVALID_POSITION):
                continue
            # compute displacement
            expert_pos = id2veh_expert[veh.id].position
            model_pos = veh.position
            pos_diff = (model_pos - expert_pos).norm()
            displacements.append(pos_diff)

        # average displacements over all vehicles
        if len(displacements) > 0:
            avg_displacements.append(np.mean(displacements))

    # average displacements over all time steps
    avg_displacement = np.mean(avg_displacements) if len(avg_displacements) > 0 else 0
    return avg_displacement


def compute_average_displacement(trajectories_dir, model, **kwargs):
    """Compute average displacement error between a model and the ground truth."""
    # get trajectories paths
    if isinstance(trajectories_dir, str):
        # if trajectories_dir is a string, treat it as the path to a directory of trajectories
        trajectories_dir = Path(trajectories_dir)
        trajectories_paths = list(trajectories_dir.glob('*tfrecord*.json'))
    elif isinstance(trajectories_dir, list):
        # if trajectories_dir is a list, treat it as a list of paths to trajectory files
        trajectories_paths = [Path(path) for path in trajectories_dir]
    # compute average displacement over each individual trajectory file
    average_displacements = list(map(
        lambda path: _average_displacement_impl(path, model, **kwargs),
        trajectories_paths
    ))

    return np.mean(average_displacements)


if __name__ == '__main__':
    from nocturne.utils.imitation_learning.waymo_data_loader import ImitationAgent  # noqa: F401
    model = torch.load('model.pth')
    ade = compute_average_displacement('dataset/json_files', model=model)
    print(f'Average Displacement Error: {ade:.3f} meters')
