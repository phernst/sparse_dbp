import csv
import json
from os.path import join as pjoin

import cv2
import nibabel as nib
import numpy as np
from skimage.transform import resize

from metrics import nmse, psnr, ssim, reduce_zdim
from utils import DATA_DIRS, load_nib


def create_wedge(size, blur_radius=None):
    wedge = np.zeros((8, 1))
    wedge[1::4] = 1
    wedge[2::4] = 1
    wedge = resize(wedge, size, order=0)
    if blur_radius is not None:
        wedge = cv2.blur(wedge, (1, int(blur_radius/360*size[1])))
    wedge = cv2.warpPolar(
        wedge,
        size,
        tuple([s/2 for s in size]),
        size[0],
        cv2.WARP_INVERSE_MAP,
    )
    wedge[tuple([int(s/2) for s in size])] = 0.5
    return wedge


def blend_method(method: str, blur: int = 90, save: bool = True):
    assert method in ['fdkconv', 's2f_inv', 's2f_inv3', 'inv_sp', 'inv_sp3']
    reco_coronal = load_nib(f'testing/{method}_coronal.nii.gz')
    reco_sagittal = load_nib(f'testing/{method}_sagittal.nii.gz')

    # non-negativity constraint
    reco_coronal[reco_coronal < 0] = 0
    reco_sagittal[reco_sagittal < 0] = 0

    # [x, y, z] -> [z, y, x]
    reco_coronal = reco_coronal.transpose()
    reco_sagittal = reco_sagittal.transpose()

    wedge = create_wedge((512, 512), blur)[None, ...]

    # spectral blending
    reco_coronal_fft = np.fft.fftshift(np.fft.fft2(reco_coronal))
    reco_sagittal_fft = np.fft.fftshift(np.fft.fft2(reco_sagittal))
    blended_fft = wedge*reco_sagittal_fft + (1-wedge)*reco_coronal_fft
    blended = np.real(np.fft.ifft2(np.fft.fftshift(blended_fft)))

    if save:
        img = nib.Nifti1Image(blended.transpose(), np.eye(4))
        nib.save(img, f'testing/{method}_blended.nii.gz')

    return blended


def blend_sweep_blur():
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']

    gt = load_nib(pjoin(DATA_DIRS['datasets'], test_files[0]))
    gt = reduce_zdim(gt).transpose()

    reco_coronal = load_nib('testing/inv_sp3_coronal.nii.gz')
    reco_sagittal = load_nib('testing/inv_sp3_sagittal.nii.gz')

    # non-negativity constraint
    reco_coronal[reco_coronal < 0] = 0
    reco_sagittal[reco_sagittal < 0] = 0

    # [x, y, z] -> [z, y, x]
    reco_coronal = reco_coronal.transpose()
    reco_sagittal = reco_sagittal.transpose()

    all_blurs = np.linspace(1, 90, 10)
    all_metrics = {'nmse': [], 'psnr': [], 'ssim': []}
    for blur in all_blurs:
        wedge = create_wedge((512, 512), blur)[None, ...]

        # spectral blending
        reco_coronal_fft = np.fft.fftshift(np.fft.fft2(reco_coronal))
        reco_sagittal_fft = np.fft.fftshift(np.fft.fft2(reco_sagittal))
        blended_fft = (wedge)*reco_sagittal_fft + (1-wedge)*reco_coronal_fft
        blended = np.real(np.fft.ifft2(np.fft.fftshift(blended_fft)))

        nmse_value = nmse(blended, gt)
        psnr_value = psnr(blended, gt)
        ssim_value = ssim(blended, gt)

        print(f'{blur} {nmse_value} {psnr_value} {ssim_value}')

        all_metrics['nmse'].append(nmse_value)
        all_metrics['psnr'].append(psnr_value)
        all_metrics['ssim'].append(ssim_value)

    with open('spectral_blur_sweep.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['blur'] + all_blurs)
        writer.writerow(['nmse'] + all_metrics['nmse'])
        writer.writerow(['psnr'] + all_metrics['psnr'])
        writer.writerow(['ssim'] + all_metrics['ssim'])
