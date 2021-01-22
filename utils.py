from enum import Enum
import json
import os
import random
import time

import nibabel as nib


def load_nib(path: str):
    return nib.load(path).get_fdata()


def load_normalizations():
    with open('normalization.json', 'r') as file:
        return json.load(file)


def load_data_dirs():
    with open('data_dirs.json', 'r') as file:
        return {k: os.path.abspath(v) for (k, v) in json.load(file).items()}


def load_configuration():
    with open('configuration.json', 'r') as file:
        return json.load(file)


NORMALIZATION = load_normalizations()
DATA_DIRS = load_data_dirs()
CONFIGURATION = load_configuration()


def create_train_valid():
    all_files = os.listdir(DATA_DIRS['datasets'])
    random.shuffle(all_files)
    train_files = all_files[:int(0.8*len(all_files))]
    valid_files = all_files[int(0.8*len(all_files)):]
    with open('train_valid.json', 'w') as json_file:
        json.dump({
            'train_files': train_files,
            'valid_files': valid_files,
        }, json_file)


class HilbertPlane(Enum):
    SAGITTAL = 0
    CORONAL = 1
    BLENDED = 2
