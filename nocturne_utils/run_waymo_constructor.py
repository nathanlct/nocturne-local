import argparse
from pathlib import Path
import os

import waymo_scenario_construction as waymo

PATH = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/training/'

def main():
    parser = argparse.ArgumentParser(
        description="Load and show waymo scenario data.")
    parser.add_argument("--file", type=str, default=os.path.join(PATH, 'training.tfrecord-00995-of-01000'))
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--all_files", action='store_true')

    args = parser.parse_args()

    if args.all_files:
        files = list(Path(PATH).glob('*tfrecord*'))
        output_dir = os.path.join('/'.join(PATH.split('/')[0:-2]), 'formatted_json')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    else:
        output_dir = os.getcwd()
        files = [args.file]

    cnt = 0
    for file in files:
        for data in waymo.load_protobuf(str(file)):
            cnt += 1
            # this file is useful for debugging
            file_name = os.path.basename(file).split('.')[1] + '.json'
            if len(files) == 1:
                with open( os.path.basename(file).split('.')[1] + '.txt', 'w') as f:
                    f.write(str(data))
            waymo.waymo_to_scenario(os.path.join(output_dir, file_name), data)
            # TODO(ev) why does load_protobuf return an iterator??
            break
        if args.num > 0 and cnt >= args.num:
            break


if __name__ == "__main__":
    main()