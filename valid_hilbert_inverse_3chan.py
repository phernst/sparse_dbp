import json
import os
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import torch

from deep_hilbert_inverse_3chan import DeepHilbertInverse, MyDataset
from spectral_blending import blend_method
from utils import HilbertPlane, NORMALIZATION


def dataset_prediction(plane: HilbertPlane):
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']

    pre_dir = 'valid_hilbert_inverse_3chan'
    pre_path = [
        x for x in sorted(os.listdir(pre_dir))
        if x.endswith('.ckpt') and x.startswith('epoch')
    ][-1]
    print(f'loading: {pjoin(pre_dir, pre_path)}')
    model = DeepHilbertInverse.load_from_checkpoint(pjoin(pre_dir, pre_path))
    model = model.cuda()
    model.eval()

    dataset = nib.load(pjoin('testing', f'sp2full_{plane.name.lower()}.nii.gz')).get_fdata()  # [512, 512, 512, 2]
    sp_fdk = nib.load(pjoin('fdk36', test_files[0])).get_fdata()[..., None]
    sp_fdk = MyDataset._reduce_zdim(sp_fdk, small=False)
    reco = np.zeros((512, 512, 512))

    for idx in range(512):
        print(f'{idx+1}/512')
        sample = dataset[idx] if plane is HilbertPlane.SAGITTAL else dataset[:, idx]
        spa = sp_fdk[idx] if plane is HilbertPlane.SAGITTAL else sp_fdk[:, idx]
        sample_t = torch.from_numpy(sample).float().cuda()
        spa_t = torch.from_numpy(spa).float().cuda()
        sample_t = sample_t.permute(2, 0, 1).unsqueeze(0)
        spa_t = spa_t.permute(2, 0, 1).unsqueeze(0)
        sample_t /= NORMALIZATION['hilbert_coronal_std']
        spa_t /= NORMALIZATION['images_99']
        pred = model(torch.cat([
            sample_t,
            spa_t,
        ], dim=1))
        pred = pred.detach().cpu().numpy()[0, 0]
        pred *= NORMALIZATION['images_99']
        pred[pred < 0] = 0
        if plane is HilbertPlane.CORONAL:
            reco[:, idx] = pred
        else:
            reco[idx] = pred

    image = nib.Nifti1Image(reco, np.eye(4))
    nib.save(image, pjoin('testing', f's2f_inv3_{plane.name.lower()}.nii.gz'))


def main():
    dataset_prediction(HilbertPlane.CORONAL)
    dataset_prediction(HilbertPlane.SAGITTAL)
    blend_method('s2f_inv3')


if __name__ == "__main__":
    main()
