"""Replay a video of a trained controller."""
from collections import deque, defaultdict
import json
import os
from pathlib import Path
import sys

import imageio
import numpy as np
from pathlib import Path
from pyvirtualdisplay import Display
import subprocess
import torch

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROCESSED_VALID_NO_TL
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

    # with open(os.path.join(PROCESSED_VALID_NO_TL, 'valid_files.json')) as file:
    #     valid_veh_dict = json.load(file)
    #     files = list(valid_veh_dict.keys())
    data_path = PROCESSED_TRAIN_NO_TL
    files = [
            file for file in Path(data_path).iterdir()
            if 'tfrecord' in file.stem
        ]
    files = files[:100]
    np.random.shuffle(files)
    for traj_path in files:
        traj_path = str(traj_path)
        sim = Simulation(scenario_path=os.path.join(data_path, str(traj_path)))
        output_str = traj_path.split('.')[0].split('/')[-1]
        model = torch.load(MODEL_PATH).to('cpu')
        model.eval()

        def policy(state):
            """Get model output."""
            state = torch.as_tensor(np.array([state]), dtype=torch.float32)
            return model(state, deterministic=True)

        with torch.no_grad():
            for expert_control_vehicles, mp4_name in [
                (False, f'{output_str}_policy_rollout.mp4'),
                (True, f'{output_str}_true_rollout.mp4')
            ]:
                frames = []
                sim.reset()
                scenario = sim.getScenario()

                objects_of_interest = [obj for obj in scenario.getVehicles()
                        if obj in scenario.getObjectsThatMoved()]
                
                for obj in objects_of_interest:
                    obj.expert_control=True
                
                state_size = model.n_states // model.n_stack
                collections_dict = defaultdict(lambda: deque([np.zeros(state_size) for i in range(model.n_stack)], model.n_stack))
                for i in range(model.n_stack):
                    for veh in objects_of_interest:
                        # TODO(eugenevinitsky) remove the 100.0
                        collections_dict[veh.getID()].append(np.concatenate(
                            (np.array(scenario.ego_state(veh), copy=False),
                             np.array(scenario.flattened_visible_state(
                                veh, view_dist=VIEW_DIST, view_angle=VIEW_ANGLE),
                                    copy=False))) / 100.0)
                    sim.step(0.1)
                
                for obj in scenario.getObjectsThatMoved():
                    obj.expert_control = True
                for veh in objects_of_interest:
                    veh.expert_control = expert_control_vehicles

                for i in range(90 - model.n_stack):
                    print(f'...{i+1}/{90 - model.n_stack} ({traj_path} ; {mp4_name})')
                    img = scenario.getImage(
                            img_width=1600,
                            img_height=1600,
                            draw_target_positions=True,
                            padding=50.0,
                            )
                    frames.append(img)
                    for veh in objects_of_interest:
                        veh_state = np.concatenate(
                            (np.array(scenario.ego_state(veh), copy=False),
                             np.array(scenario.flattened_visible_state(
                                veh, view_dist=VIEW_DIST, view_angle=VIEW_ANGLE),
                                    copy=False)))
                        collections_dict[veh.getID()].append(veh_state)
                        action = policy(np.concatenate(collections_dict[veh.getID()]))[0]
                        # veh.acceleration = action[0]
                        # veh.steering = action[1]
                        action = action.cpu().numpy()
                        pos_diff = action[0:2]
                        heading = action[2:3]
                        veh.position = Vector2D.from_numpy(pos_diff + veh.position.numpy())
                        veh.heading += heading
                    sim.step(0.1)
                    for veh in scenario.getObjectsThatMoved():
                        if (veh.position -
                                veh.target_position).norm() < GOAL_TOLERANCE:
                            scenario.removeVehicle(veh)
                imageio.mimsave(mp4_name, np.stack(frames, axis=0), fps=30)
                print(f'> {mp4_name}')

        # stack the movies side by side
        output_name = traj_path.split('.')[0].split('/')[-1]
        output_path = f'{output_name}_output.mp4'
        ffmpeg_command = f'ffmpeg -y -i {output_str}_true_rollout.mp4 -i {output_str}_policy_rollout.mp4 -filter_complex hstack {output_path}'
        print(ffmpeg_command)
        subprocess.call(ffmpeg_command.split(' '))
        print(f'> {output_path}')
        sys.exit()
