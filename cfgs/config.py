"""Set path to all the Waymo data and the parsed Waymo files."""
import os
from pathlib import Path

VERSION_NUMBER = 2

PROJECT_PATH = Path.resolve(Path(__file__).parent.parent)
DATA_FOLDER = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/'
TRAIN_DATA_PATH = os.path.join(DATA_FOLDER, 'training')
VALID_DATA_PATH = os.path.join(DATA_FOLDER, 'validation')
TEST_DATA_PATH = os.path.join(DATA_FOLDER, 'testing')
PROCESSED_TRAIN_NO_TL = os.path.join(
    DATA_FOLDER, f'formatted_json_v{VERSION_NUMBER}_no_tl_train')
PROCESSED_VALID_NO_TL = os.path.join(
    DATA_FOLDER, f'formatted_json_v{VERSION_NUMBER}_no_tl_valid')
PROCESSED_TEST_NO_TL = os.path.join(
    DATA_FOLDER, f'formatted_json_v{VERSION_NUMBER}_no_tl_test')
PROCESSED_TRAIN = os.path.join(DATA_FOLDER,
                               f'formatted_json_v{VERSION_NUMBER}_train')
PROCESSED_VALID = os.path.join(DATA_FOLDER,
                               f'formatted_json_v{VERSION_NUMBER}_valid')
PROCESSED_TEST = os.path.join(DATA_FOLDER,
                              f'formatted_json_v{VERSION_NUMBER}_test')

ERR_VAL = -1e4

DEFAULT_SCENARIO_CONFIG = {
    # initial timestep of the scenario (which ranges from timesteps 0 to 90)
    'start_time': 0,
    # if set to True, non-vehicle objects (eg. cyclists, pedestrians...) will be spawned
    'allow_non_vehicles': True,
    # for an object to be included into moving_objects
    'moving_threshold': 0.2,  # its goal must be at least this distance from its initial position
    'speed_threshold': 0.05,  # its speed must be superior to this value at some point
    # maximum number of each objects visible in the object state
    # if there are more objects, the closest ones are prioritized
    # if there are less objects, the features vector is padded with zeros
    'max_visible_objects': 20,
    'max_visible_road_points': 200,
    'max_visible_traffic_lights': 20,
    'max_visible_stop_signs': 4,
}
