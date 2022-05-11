#!/usr/bin/env python3
import time
import os
import sys
import argparse
import pathlib, shutil
from datetime import datetime
from subprocess import Popen, DEVNULL

from cfgs.config import PROJECT_PATH


class Overrides(object):

    def __init__(self):
        self.kvs = dict()

    def add(self, key, values):
        value = ','.join(str(v) for v in values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd


def make_code_snap(experiment, code_path, str_time):
    if len(code_path) > 0:
        snap_dir = pathlib.Path(code_path)
    else:
        snap_dir = pathlib.Path.cwd()
    snap_dir /= str_time
    snap_dir /= f'{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(dir, pat):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            shutil.copy(f, dst_dir / f.name)

    dirs_to_copy = [
        '.', './cfgs/', './examples/', './examples/sample_factory_files',
        './cfgs/algorithm', './envs/', './nocturne_utils/', './python/',
        './scenarios/', './build'
    ]
    src_dir = pathlib.Path(PROJECT_PATH)
    for dir in dirs_to_copy:
        copy_dir(dir, '*.py')
        copy_dir(dir, '*.yaml')

    return snap_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument(
        '--code_path',
        default='/checkpoint/eugenevinitsky/nocturne/sample_factory_runs')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    now = datetime.now()
    str_time = now.strftime('%Y.%m.%d_%H%M%S')
    snap_dir = make_code_snap(args.experiment, args.code_path, str_time)
    overrides = Overrides()
    overrides.add('hydra/launcher', ['submitit_slurm'])
    overrides.add('hydra.launcher.partition', ['learnlab'])
    overrides.add('experiment', [args.experiment])
    # overrides.add('num_files', [10])
    overrides.add('single_agent_mode', [False])
    # overrides.add('algorithm.kl_loss_coeff', [0.0, 0.01])
    # overrides.add('algorithm.max_grad_norm', [1.0, 4.0, 20.0])
    # overrides.add('max_num_vehicles', [5])
    # overrides.add('rew_cfg.goal_achieved_bonus', [0])
    # overrides.add('rew_cfg.goal_distance_penalty', [True])
    # overrides.add('rew_cfg.collision_penalty', [-90.0])
    # overrides.add('discretize_actions', [True, False])
    # overrides.add('algorithm.kl_loss_coeff', [0.0, 0.01, 0.1, 1.0, 10.0])
    # overrides.add('algorithm.max_grad_norm', [4.0, 20.0])
    # exp
    # overrides.add('algorithm.ppo_clip_ratio', [0.02, 0.05, 0.1])
    # overrides.add('algorithm.ppo_clip_value', [1.0, 10.0])

    # overrides.add('algorithm.rollout', [10, 20, 30])
    # overrides.add('algorithm.recurrence', [10, 20, 30])
    # overrides.add('algorithm.learning_rate', [0.0001, 0.00005, 0.00001])
    # overrides.add('num_files', [1, 10, 100, 1000, -1])

    overrides.add('num_files', [100])
    overrides.add('algorithm.exploration_loss_coeff', [0.0, 0.0001, 0.001])
    overrides.add('algorithm.kl_loss_coeff', [0.0, 0.1, 1.0])
    overrides.add('algorithm.learning_rate', [0.001, 0.0001, 0.00001])

    cmd = [
        'python',
        str(snap_dir / 'code' / 'examples' / 'sample_factory_files' /
            'run_sample_factory.py'), '-m', 'algorithm=APPO'
    ]
    print(cmd)
    cmd += overrides.cmd()

    if args.dry:
        print(' '.join(cmd))
    else:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(snap_dir / 'code')
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()