'''Replay a video of a trained controller'''
from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
import torch

from nocturne import Simulation
from waymo_data_loader import ImitationAgent


OUTPUT_PATH = './vids'

EVAL_SCENARIO_PATHS = map(Path, [
    'dataset/json_files/tfrecord-00411-of-01000_160.json',
    'dataset/json_files/tfrecord-00411-of-01000_155.json',
    'dataset/json_files/tfrecord-00411-of-01000_188.json',
    'dataset/json_files/tfrecord-00411-of-01000_135.json',
    'dataset/json_files/tfrecord-00411-of-01000_195.json',
    'dataset/json_files/tfrecord-00411-of-01000_149.json',
    'dataset/json_files/tfrecord-00411-of-01000_174.json',
    'dataset/json_files/tfrecord-00411-of-01000_148.json',
    'dataset/json_files/tfrecord-00411-of-01000_179.json',
])

MODEL_PATH = 'model.pth'
GOAL_TOLERANCE = 1.0


if __name__ == '__main__':
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True)
    
    for traj_path in EVAL_SCENARIO_PATHS:
        sim = Simulation(scenario_path=str(traj_path))

        model = torch.load(MODEL_PATH)
        model.eval()
        model.deterministic = True

        def policy(state):
            with torch.no_grad():
                state = torch.as_tensor(np.array([state]), dtype=torch.float32)
                return model(state)
    
        for expert_control_vehicles, mp4_name in [
            (False, f'{traj_path.stem}_policy_rollout.mp4'), (True, f'{traj_path.stem}_true_rollout.mp4')
        ]:
            fig = plt.figure()
            cam = Camera(fig)

            sim.reset()
            scenario = sim.getScenario()

            for obj in scenario.getObjectsThatMoved():
                obj.expert_control = True
            for veh in scenario.getVehicles():
                veh.expert_control = expert_control_vehicles
            for i in range(90):
                print(f'...{i+1}/90 ({traj_path} ; {mp4_name})')
                img = np.array(scenario.getImage(None, render_goals=True), copy=False)
                plt.imshow(img)
                cam.snap()
                for veh in scenario.getVehicles():
                    veh_state = np.concatenate((
                        np.array(scenario.ego_state(veh), copy=False),
                        np.array(scenario.flattened_visible_state(veh, view_dist=120, view_angle=3.14), copy=False)
                    ))
                    action = policy(veh_state)[0]
                    veh.acceleration = action[0]
                    veh.steering = action[1]
                sim.step(0.1)
                for veh in scenario.getVehicles():
                    if (veh.position - veh.destination).norm() < GOAL_TOLERANCE:
                        scenario.removeVehicle(veh)
            animation = cam.animate(interval=50)
            animation.save(mp4_name)
            print(f'> {mp4_name}')

        # stack the movies side by side
        output_path = f'{traj_path.stem}_output.mp4'
        subprocess.call(f'ffmpeg -y -i {traj_path.stem}_true_rollout.mp4 -i {traj_path.stem}_policy_rollout.mp4 -filter_complex hstack {output_path}'.split(' '))
        print(f'> {output_path}')
