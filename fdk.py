import json
from os.path import join as pjoin

import ctl
import nibabel as nib
import numpy as np

from utils import DATA_DIRS


NUM_VIEWS = 360
SDD = 1000.
SID = 750.
NUM_DET_PIXELS = 1024
DET_PIXEL_DIM = 1.


def create_fdk(filename: str):
    nib_volume = nib.load(pjoin(DATA_DIRS['datasets'], filename))
    nib_shape = nib_volume.header.get_data_shape()
    nib_dims = tuple([float(f) for f in nib_volume.header['pixdim'][1:4]])
    nib_volume = nib_volume.get_fdata()
    print(nib_dims)

    system = ctl.CTSystem()
    system.add_component(ctl.FlatPanelDetector(
        (NUM_DET_PIXELS, NUM_DET_PIXELS),
        (DET_PIXEL_DIM, DET_PIXEL_DIM),
    ))
    system.add_component(ctl.TubularGantry(SDD, SID))
    system.add_component(ctl.XrayTube())

    setup = ctl.AcquisitionSetup(system, NUM_VIEWS)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())

    ctl_volume = ctl.VoxelVolumeF.from_numpy(nib_volume.transpose())
    ctl_volume.set_voxel_size(nib_dims)

    projector = ctl.ocl.RayCasterProjector()
    projections = projector.configure_and_project(setup, ctl_volume)

    rec = ctl.ocl.FDKReconstructor()
    reco = ctl.VoxelVolumeF(nib_shape, nib_dims)
    reco.fill(0)
    rec.configure_and_reconstruct_to(setup, projections, reco)

    img = nib.Nifti1Image(reco, np.eye(4))
    nib.save(img, f'fdk{NUM_VIEWS}/{filename}')


def main():
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        dataset_files = json_dict['train_files'] \
            + json_dict['valid_files'] \
            + json_dict['test_files']

    for filename in dataset_files:
        print(filename)
        create_fdk(filename)


if __name__ == "__main__":
    main()
