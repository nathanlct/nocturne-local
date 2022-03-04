import argparse
from pathlib import Path
import os

import waymo_scenario_construction as waymo

PATH = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/training/'

def main():
    parser = argparse.ArgumentParser(
        description="Load and show waymo scenario data.")
    parser.add_argument("--file", type=str, default=os.path.join(PATH, 'training.tfrecord-00995-of-01000'))
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--output_txt", action='store_true', help='output a txt version of one of the protobufs')
    parser.add_argument("--all_files", action='store_true', 
                        help='If true, iterate through the whole dataset')

    args = parser.parse_args()

    if args.num > 1 or args.all_files:
        files = list(Path(PATH).glob('*tfrecord*'))
        output_dir = os.path.join('/'.join(PATH.split('/')[0:-2]), 'formatted_json')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not args.all_files:
            files = files[0:args.num]

    else:
        output_dir = os.getcwd()
        files = [args.file]

    cnt = 0
    for file in files:
        inner_count = 0
        for data in waymo.load_protobuf(str(file)):
            file_name = os.path.basename(file).split('.')[1] + f'_{inner_count}.json'
            # this file is useful for debugging
            if args.output_txt and cnt == 0:
                with open( os.path.basename(file).split('.')[1] + '.txt', 'w') as f:
                    f.write(str(data))
            waymo.waymo_to_scenario(os.path.join(output_dir, file_name), data)
            inner_count += 1
            cnt += 1
            if cnt >= args.num and not args.all_files:
                import ipdb; ipdb.set_trace()
                break
        print(inner_count)


if __name__ == "__main__":
    main()