import json
import os
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import torch

from deep_hilbert_inverse_3chan_sparse import DeepHilbertInverse, MyDataset
from spectral_blending import blend_method
from utils import HilbertPlane, NORMALIZATION


def dataset_prediction(plane: HilbertPlane):
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']

    pre_dir = 'valid_hilbert_inverse_3chan_sparse'
    pre_path = [
        x for x in sorted(os.listdir(pre_dir))
        if x.endswith('.ckpt') and x.startswith('epoch')
    ][-1]
    print(f'loading: {pjoin(pre_dir, pre_path)}')
    model = DeepHilbertInverse.load_from_checkpoint(pjoin(pre_dir, pre_path))
    model = model.cuda()
    model.eval()

    dataset = MyDataset(
        filename=test_files[0],
        plane=plane,
        transform=model.trafo_valid,
    )
    reco = np.zeros((512, 512, 512))

    for idx in range(512):
        print(f'{idx+1}/512')
        sample = dataset[idx]
        hilbert, sparse = sample['hil'].cuda(), sample['sparse'].cuda()
        hilbert = torch.cat([hilbert, sparse], dim=0)[None, ...]
        pred = model(hilbert)
        pred = pred.detach().cpu().numpy()[0, 0]
        pred *= NORMALIZATION['images_99']
        pred[pred < 0] = 0
        if plane is HilbertPlane.CORONAL:
            reco[:, idx] = pred
        else:
            reco[idx] = pred

    image = nib.Nifti1Image(reco, np.eye(4))
    nib.save(image, pjoin('testing', f'inv_sp3_{plane.name.lower()}.nii.gz'))


def main():
    dataset_prediction(HilbertPlane.CORONAL)
    dataset_prediction(HilbertPlane.SAGITTAL)
    blend_method('inv_sp3')


if __name__ == "__main__":
    main()
