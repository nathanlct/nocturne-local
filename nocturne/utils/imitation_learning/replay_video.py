"""Replay a video of a trained controller."""
import json
import os

import imageio
import numpy as np
from pathlib import Path
from pyvirtualdisplay import Display
import subprocess
import torch

from cfgs.config import PROCESSED_VALID_NO_TL
from nocturne import Simulation, Vector2D
from nocturne.utils.imitation_learning.waymo_data_loader import VIEW_DIST, VIEW_ANGLE

OUTPUT_PATH = './vids'

MODEL_PATH = 'model.pth'
GOAL_TOLERANCE = 1.0

if __name__ == '__main__':
    disp = Display()
    disp.start()
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True)

    with open(os.path.join(PROCESSED_VALID_NO_TL, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
    for traj_path in files:
        sim = Simulation(scenario_path=os.path.join(PROCESSED_VALID_NO_TL, str(traj_path)))

        model = torch.load(MODEL_PATH).to('cpu')
        model.eval()

        def policy(state):
            """Get model output."""
            state = torch.as_tensor(np.array([state]), dtype=torch.float32)
            return model(state, deterministic=True)

        frames = []

        with torch.no_grad():
            for expert_control_vehicles, mp4_name in [
                (False, f'{traj_path}_policy_rollout.mp4'),
                (True, f'{traj_path}_true_rollout.mp4')
            ]:

                sim.reset()
                scenario = sim.getScenario()

                for obj in scenario.getObjectsThatMoved():
                    obj.expert_control = True
                for veh in scenario.getVehicles():
                    veh.expert_control = expert_control_vehicles
                for i in range(90):
                    print(f'...{i+1}/90 ({traj_path} ; {mp4_name})')
                    img = scenario.getImage(
            img_width=1600,
            img_height=1600,
            draw_target_positions=True,
            padding=50.0,
        )
                    frames.append(img)
                    for veh in scenario.getVehicles():
                        veh_state = np.concatenate(
                            (np.array(scenario.ego_state(veh), copy=False),
                             np.array(scenario.flattened_visible_state(
                                veh, view_dist=VIEW_DIST, view_angle=VIEW_ANGLE),
                                    copy=False)))
                        action = policy(veh_state)[0]
                        # veh.acceleration = action[0]
                        # veh.steering = action[1]
                        veh.position = Vector2D.from_numpy(action.cpu().numpy() + veh.position.numpy())
                    sim.step(0.1)
                    for veh in scenario.getObjectsThatMoved():
                        if (veh.position -
                                veh.target_position).norm() < GOAL_TOLERANCE:
                            scenario.removeVehicle(veh)
                imageio.mimsave(mp4_name, np.stack(frames, axis=0), fps=30)
                print(f'> {mp4_name}')

        # stack the movies side by side
        import ipdb; ipdb.set_trace()
        output_path = f'{traj_path}_output.mp4'
        subprocess.call(f'ffmpeg -y -i {traj_path}_true_rollout.mp4 -i \
                {traj_path}_policy_rollout.mp4 -filter_complex hstack {output_path}'
                        .split(' '))
        print(f'> {output_path}')
