"""Example of how to make movies of Nocturne scenarios."""
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from pyvirtualdisplay import Display

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH
from nocturne import Simulation


def get_sim(scenario_file):
    """Initialize the scenario."""
    # load scenario, set vehicles to be expert-controlled
    sim = Simulation(scenario_path=str(scenario_file),
                     allow_non_vehicles=False)
    for obj in sim.getScenario().getObjectsThatMoved():
        obj.expert_control = True
    return sim


def make_movie(sim,
               scenario_fn,
               output_path='./vid.mp4',
               dt=0.1,
               steps=90,
               fps=10):
    """Make a movie from the scenario."""
    scenario = sim.getScenario()
    movie_frames = []
    timestep = 0
    movie_frames.append(scenario_fn(scenario, timestep))
    for i in range(steps):
        sim.step(dt)
        timestep += 1
        movie_frames.append(scenario_fn(scenario, timestep))
    movie_frames = np.array(movie_frames)
    imageio.mimwrite(output_path, movie_frames, fps=fps)
    print('>', output_path)
    del sim
    del movie_frames


def make_image(sim, scenario_file, scenario_fn, output_path='./img.png'):
    """Make a single image from the scenario."""
    scenario = sim.getScenario()
    img = scenario_fn(scenario)
    dpi = 100
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(output_path)
    print('>', output_path)


if __name__ == '__main__':
    disp = Display()
    disp.start()

    # files = ['tfrecord-00358-of-01000_{}.json'.format(i) for i in range(500)]

    files = [
        'tfrecord-00358-of-01000_60.json',  # unprotected turn
        'tfrecord-00358-of-01000_72.json',  # four way stop
        'tfrecord-00358-of-01000_257.json',  # crowded four way stop
        'tfrecord-00358-of-01000_332.json',  # crowded merge road
        'tfrecord-00358-of-01000_79.json',  # crowded parking lot
    ]
    for file in files:
        file = os.path.join(PROCESSED_TRAIN_NO_TL, file)
        sim = get_sim(file)
        if os.path.exists(file):
            # image of whole scenario
            # make_image(
            #     sim,
            #     file,
            #     scenario_fn=lambda scenario: scenario.getImage(
            #         img_width=2000,
            #         img_height=2000,
            #         padding=50.0,
            #         draw_destinations=True,
            #     ),
            #     output_path=PROJECT_PATH /
            #     'scripts/paper_plots/figs/scene_{}.png'.format(
            #         os.path.basename(file)),
            # )

            make_image(
                sim,
                file,
                scenario_fn=lambda scenario: scenario.getImage(
                    img_width=1600,
                    img_height=1600,
                    draw_destinations=True,
                    padding=50.0,
                    source=scenario.getVehicles()[-1],
                    view_width=120,
                    view_height=120,
                    rotate_with_source=True,
                ),
                output_path=PROJECT_PATH /
                'scripts/paper_plots/figs/cone_original_{}.png'.format(
                    os.path.basename(file)),
            )
            make_image(
                sim,
                file,
                scenario_fn=lambda scenario: scenario.getConeImage(
                    source=scenario.getVehicles()[-1],
                    view_dist=120.0,
                    view_angle=np.pi,
                    head_tilt=0.0,
                    img_width=1600,
                    img_height=1600,
                    padding=50.0,
                    draw_destination=True,
                ),
                output_path=PROJECT_PATH /
                'scripts/paper_plots/figs/cone_{}.png'.format(
                    os.path.basename(file)),
            )
            make_image(
                sim, 
                file,
                scenario_fn=lambda scenario: scenario.getFeaturesImage(
                    source=scenario.getVehicles()[-1],
                    view_dist=120.0,
                    view_angle=np.pi,
                    head_tilt=0.0,
                    img_width=1600,
                    img_height=1600,
                    padding=50.0,
                    draw_destination=True,
                ),
                output_path=PROJECT_PATH /
                'scripts/paper_plots/figs/feature_{}.png'.format(
                    os.path.basename(file)),
            )
