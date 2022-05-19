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
