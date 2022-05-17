#!/usr/bin/env python3
import argparse
import os
import pathlib
import shutil
from datetime import datetime
from subprocess import Popen

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
        './cfgs/algorithm', './nocturne/envs/', './nocturne_utils/', './nocturne/python/',
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
    overrides.add('num_files', [100])
    overrides.add('seed', [0, 1, 2, 3, 4])

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
