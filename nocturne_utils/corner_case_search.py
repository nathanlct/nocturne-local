# This file runs through the data to look for cases where there are undesirable corner cases
# the cases we currently check for are:
# 1) is a vehicle initialized in a colliding state with another vehicle
# 2) is a vehicle initialized in a colliding state with a road edge?
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH
from nocturne import Simulation

os.environ["DISPLAY"] = ":0.0"

if __name__ == '__main__':
    SAVE_IMAGES = False
    output_folder = 'corner_case_vis'
    output_path = Path(PROJECT_PATH) / f'nocturne_utils/{output_folder}'
    output_path.mkdir(exist_ok=True)
    files = list(os.listdir(PROCESSED_TRAIN_NO_TL))
    # track the number of collisions at each time-step
    collide_counter = np.zeros(90)
    file_has_collision_counter = 0
    for file_idx, file in enumerate(files):
        found_collision = False
        sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file), 0, False)
        vehs = sim.getScenario().getObjectsThatMoved()
        # this checks if the vehicles has actually moved any distance at all
        valid_vehs = []
        for veh in vehs:
            veh.expert_control = True
            obj_pos = veh.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            if np.linalg.norm(obj_pos - goal_pos) > 0.5:
                valid_vehs.append(veh)
        for time_index in range(90):
            for veh_index, veh in enumerate(valid_vehs):
                collided = veh.getCollided()
                if collided and not np.isclose(veh.getPosition().x, -10000.0):
                    collide_counter[time_index] += 1
                if np.isclose(veh.getPosition().x, -10000.0):
                    collided = False
                if time_index == 0 and not found_collision and collided and SAVE_IMAGES:
                    img = sim.getScenario().getImage(None, render_goals=True)
                    fig = plt.figure()
                    plt.imshow(img)
                    plt.savefig(f'{output_folder}/{file}.png')
                    plt.close(fig)
                if not found_collision and collided:
                    found_collision = True
                    file_has_collision_counter += 1
            sim.step(0.1)
        print(
            f'at file {file_idx} we have {collide_counter} collisions for a ratio of {collide_counter / (file_idx + 1)}'
        )
        print(
            f'the number of files that have a collision at all is {file_has_collision_counter / (file_idx + 1)}'
        )
        # if found_collision:
        #     import sys
        #     from celluloid import Camera
        #     fig = plt.figure()
        #     cam = Camera(fig)
        #     sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file), 0,
        #                      False)
        #     vehs = sim.getScenario().getObjectsThatMoved()
        #     for veh in vehs:
        #         veh.set_expert_controlled(True)
        #     for time_index in range(89):
        #         img = sim.getScenario().getImage(None, render_goals=True)
        #         plt.imshow(img)
        #         cam.snap()
        #         sim.step(0.1)
        #     animation = cam.animate(interval=50)
        #     animation.save(f'{output_path}/{os.path.basename(file)}.mp4')
        #     if file_has_collision_counter > 5:
        #         sys.exit()
