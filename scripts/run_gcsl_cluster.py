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


def make_code_snap(experiment, code_path, slurm_dir='exp'):
    now = datetime.now()
    if len(code_path) > 0:
        snap_dir = pathlib.Path(code_path) / slurm_dir
    else:
        snap_dir = pathlib.Path.cwd() / slurm_dir
    snap_dir /= now.strftime('%Y.%m.%d')
    snap_dir /= now.strftime('%H%M%S') + f'_{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(dir, pat):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            shutil.copy(f, dst_dir / f.name)

    dirs_to_copy = [
        '.', './cfgs/', './cfgs/algo', './algos/', './algos/ppo/',
        './algos/gcsl/', './envs/', './nocturne_utils/', './python/',
        './scenarios/', './build'
    ]
    src_dir = pathlib.Path(os.path.dirname(os.getcwd()))
    for dir in dirs_to_copy:
        copy_dir(dir, '*.py')
        copy_dir(dir, '*.yaml')

    return snap_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--code_path',
                        default='/checkpoint/eugenevinitsky/nocturne')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    snap_dir = make_code_snap(args.experiment, args.code_path)
    print(str(snap_dir))
    overrides = Overrides()
    overrides.add('hydra/launcher', ['submitit_slurm'])
    overrides.add('hydra.launcher.partition', ['learnlab'])
    overrides.add('experiment', [args.experiment])
    # algo
    overrides.add('algo.max_trajectory_length', [50])
    overrides.add('algo.quartile_cutoff', [0])
    overrides.add('algo.batch_size', [256])
    # misc
    overrides.add('scenario_path',
                  [PROJECT_PATH / 'scenarios/four_car_intersection.json'])

    cmd = [
        'python',
        str(snap_dir / 'code' / 'algos' / 'gcsl' / 'gcsl_example.py'), '-m'
    ]
    print(cmd)
    cmd += overrides.cmd()

    if args.dry:
        print(' '.join(cmd))
    else:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(snap_dir / 'code')
        import ipdb; ipdb.set_trace()
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()