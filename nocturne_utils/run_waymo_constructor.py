import argparse

import waymo_scenario_construction as waymo


def main():
    parser = argparse.ArgumentParser(
        description="Load and show waymo scenario data.")
    parser.add_argument("--file", type=str, default='/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/training/training.tfrecord-00998-of-01000')
    parser.add_argument("--num", type=int, default=1)

    args = parser.parse_args()

    cnt = 0
    for data in waymo.load_protobuf(args.file):
        cnt += 1
        with open('output.txt', 'w') as f:
            f.write(str(data))
        waymo.waymo_to_scenario('output.json', data)
        if args.num is not None and cnt >= args.num:
            break


if __name__ == "__main__":
    main()