import os
import matplotlib.pyplot as plt
import numpy as np


def create_heat_map(file, title, save_path):
    np_arr = np.load(os.path.join(zsc_path, file))
    np_arr_mean = np.mean(np_arr, axis=-1)
    np_arr_mean_centered = np_arr_mean - np.diag(np_arr_mean)[:, np.newaxis]
    np_arr_std = np.std(np_arr - np.diag(np_arr_mean)[:, np.newaxis], axis=-1)

    agent_indices = [f'Agent {i}' for i in range(np_arr.shape[0])]

    fig, ax = plt.subplots()
    im = ax.imshow(np_arr_mean)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(agent_indices)), labels=agent_indices)
    ax.set_yticks(np.arange(len(agent_indices)), labels=agent_indices)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(agent_indices)):
        for j in range(len(agent_indices)):
            text = ax.text(
                j,
                i,
                f'{np.round(np_arr_mean_centered[i, j], decimals=2)}',  #±{np.round(np_arr_std[i, j], decimals=2)}',
                ha="center",
                va="center",
                color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_path)


def compute_average_change(file):
    np_arr = np.load(os.path.join(zsc_path, file))
    np_arr_mean = np.mean(np_arr, axis=-1)
    np_arr_std = np.std(np_arr, axis=-1)
    self_play = np.mean(np.diag(np_arr_mean))
    cross_play = np.mean(
        np_arr_mean[np.where(~np.eye(np_arr_mean.shape[0], dtype=bool))])
    self_play_std = np.mean(np.diag(np_arr_std))
    cross_play_std = np.mean(
        np_arr_std[np.where(~np.eye(np_arr_std.shape[0], dtype=bool))])
    print(
        f'self play: {self_play} ± {self_play_std}, cross play: {cross_play} ± {cross_play_std}'
    )


if __name__ == '__main__':
    zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.23/srt_v10/17.02.40/23/srt_v10'
    create_heat_map('zsc_goal.npy', "Cross-play Relative Goal Rate",
                    'cross_play_heat_map.png')
    create_heat_map('zsc_collision.npy', "Cross-play Relative Collision Rate",
                    'cross_play_collision_map.png')
    compute_average_change('zsc_goal.npy')
    compute_average_change('zsc_collision.npy')
