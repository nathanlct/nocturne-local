"""Replay a video of a trained controller."""
import json
import os

import imageio
import numpy as np
from pathlib import Path
import subprocess
import torch

from cfgs.config import PROCESSED_VALID_NO_TL
from nocturne import Simulation
# from nocturne.nocturne_utils.imitation_learning.waymo_data_loader import ImitationAgent

OUTPUT_PATH = './vids'

MODEL_PATH = 'model.pth'
GOAL_TOLERANCE = 1.0

if __name__ == '__main__':
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True)

    with open(os.path.join(PROCESSED_VALID_NO_TL, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
    for traj_path in files:
        sim = Simulation(scenario_path=str(traj_path))

        model = torch.load(MODEL_PATH)
        model.eval()

        def policy(state):
            """Get model output."""
            state = torch.as_tensor(np.array([state]), dtype=torch.float32)
            return model(state, deterministic=True)

        frames = []

        with torch.no_grad():
            for expert_control_vehicles, mp4_name in [
                (False, f'{traj_path.stem}_policy_rollout.mp4'),
                (True, f'{traj_path.stem}_true_rollout.mp4')
            ]:

                sim.reset()
                scenario = sim.getScenario()

                for obj in scenario.getObjectsThatMoved():
                    obj.expert_control = True
                for veh in scenario.getVehicles():
                    veh.expert_control = expert_control_vehicles
                for i in range(90):
                    print(f'...{i+1}/90 ({traj_path} ; {mp4_name})')
                    img = np.array(scenario.getImage(None, render_goals=True),
                                copy=False)
                    frames.append(img)
                    for veh in scenario.getVehicles():
                        veh_state = np.concatenate(
                            (np.array(scenario.ego_state(veh), copy=False),
                            np.array(scenario.flattened_visible_state(
                                veh, view_dist=120, view_angle=3.14),
                                    copy=False)))
                        action = policy(veh_state)[0]
                        veh.acceleration = action[0]
                        veh.steering = action[1]
                    sim.step(0.1)
                    for veh in scenario.getObjectsThatMoved():
                        if (veh.position -
                                veh.destination).norm() < GOAL_TOLERANCE:
                            scenario.removeVehicle(veh)
                imageio.mimsave(mp4_name, np.stack(frames, axis=0), fps=30)
                print(f'> {mp4_name}')

        # stack the movies side by side
        output_path = f'{traj_path.stem}_output.mp4'
        subprocess.call(f'ffmpeg -y -i {traj_path.stem}_true_rollout.mp4 -i \
                {traj_path.stem}_policy_rollout.mp4 -filter_complex hstack {output_path}'
                        .split(' '))
        print(f'> {output_path}')
