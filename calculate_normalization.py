import json
import os
from os.path import join as pjoin

import nibabel as nib
import numpy as np

from utils import DATA_DIRS


def calculate_normalization(ftype: str):
    assert ftype in ['image', 'hilbert']

    normfunction = (lambda x: np.percentile(x, 99)) if ftype == 'image' \
        else np.std

    data_path = DATA_DIRS['views_360' if ftype == 'hilbert' else 'datasets']
    all_normvalues = []
    for fname in os.listdir(data_path):
        img = nib.load(pjoin(data_path, fname)).get_fdata()
        all_normvalues.append(normfunction(img))
    return np.median(all_normvalues)


def calculate_and_save_normalizations():
    images_99 = calculate_normalization('image')
    hilbert_coronal_std = calculate_normalization('hilbert')

    with open('normalization.json', 'w') as json_file:
        json.dump({
            "images_99": images_99,
            'hilbert_coronal_std': hilbert_coronal_std,
        }, json_file)


if __name__ == "__main__":
    calculate_and_save_normalizations()
