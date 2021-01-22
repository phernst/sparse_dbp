import json
from os.path import join as pjoin

import numpy as np
from skimage.metrics import structural_similarity

from utils import DATA_DIRS, HilbertPlane, load_nib


# average over first dimension: [z, x, y]
def nmse(prediction, target, reduce=np.mean):
    return reduce(np.sum(
        (prediction - target)**2, axis=(1, 2)) /
        np.sum(target**2, axis=(1, 2)))


def psnr(prediction, target, reduce=np.mean):
    return reduce(20*np.log10(np.prod(target.shape[1:]) *
                   np.max(target, axis=(1, 2)) /
                   np.sqrt(np.sum((prediction-target)**2, axis=(1, 2)))))


def ssim(prediction, target, reduce=np.mean):
    assert prediction.shape == target.shape
    return reduce(np.array([
        structural_similarity(prediction[idx], target[idx])
        for idx in range(prediction.shape[0])]))


def reduce_zdim(volume):
    zdiff = volume.shape[-1] - 512
    if zdiff > 0:  # bigger -> cut
        front_cut = zdiff//2
        back_cut = zdiff - front_cut
        volume = volume[..., front_cut:-back_cut]
    return volume


def calculate_metrics(method: str, plane: HilbertPlane):
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']

    gt = load_nib(pjoin(DATA_DIRS['datasets'], test_files[0]))
    gt = reduce_zdim(gt).transpose()

    prediction = load_nib(f'testing/{method}_{plane.name.lower()}.nii.gz')
    prediction = prediction.transpose()

    return {
        'nmse': nmse(prediction, gt),
        'psnr': psnr(prediction, gt),
        'ssim': ssim(prediction, gt),
    }


def calculate_z_depending_metrics():
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']

    gt = load_nib(pjoin(DATA_DIRS['datasets'], test_files[0]))
    gt = reduce_zdim(gt).transpose()

    prediction = load_nib('testing/inv_sp3_blended.nii.gz')
    prediction = prediction.transpose()

    fdkconv = load_nib('testing/fdkconv_blended.nii.gz')
    fdkconv = reduce_zdim(fdkconv).transpose()

    pred_nmse = nmse(prediction, gt, lambda _: _)
    pred_psnr = psnr(prediction, gt, lambda _: _)
    pred_ssim = ssim(prediction, gt, lambda _: _)

    fdkconv_nmse = nmse(fdkconv, gt, lambda _: _)
    fdkconv_psnr = psnr(fdkconv, gt, lambda _: _)
    fdkconv_ssim = ssim(fdkconv, gt, lambda _: _)

    from matplotlib import pyplot as plt
    plt.plot(pred_nmse/fdkconv_nmse), plt.figure()
    plt.plot(pred_psnr/fdkconv_psnr), plt.figure()
    plt.plot(pred_ssim/fdkconv_ssim), plt.show()


def calculate_sparse_fdk_metrics():
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']

    gt = load_nib(pjoin(DATA_DIRS['datasets'], test_files[0]))
    gt = reduce_zdim(gt).transpose()

    sparse_fdk = load_nib('sparse_fdk.nii.gz')
    sparse_fdk = reduce_zdim(sparse_fdk).transpose()

    return {
        'nmse': nmse(sparse_fdk, gt),
        'psnr': psnr(sparse_fdk, gt),
        'ssim': ssim(sparse_fdk, gt),
    }


def main():
    all_methods = ['fdkconv', 's2f_inv', 's2f_inv3', 'inv_sp', 'inv_sp3']
    all_planes = [
        HilbertPlane.CORONAL,
        HilbertPlane.SAGITTAL,
        HilbertPlane.BLENDED
    ]
    for plane in all_planes:
        for method in all_methods:
            met = calculate_metrics(method, plane)
            print(f"{plane.name.lower()}, {method}, {met['nmse']}, {met['psnr']}, {met['ssim']}")


if __name__ == "__main__":
    main()
