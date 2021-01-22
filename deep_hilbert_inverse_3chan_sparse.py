from argparse import ArgumentParser
from typing import Callable, Optional, List, Any
import json
import os
from os.path import join as pjoin
import random

import cv2
import nibabel as nib
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from concatdataset import ConcatDataset
from unet import UNet
from utils import load_nib, CONFIGURATION, DATA_DIRS, NORMALIZATION
from utils import HilbertPlane


class Normalize:
    def __call__(_, sample):
        slist = [sample['hil'], sample['sparse'], sample['img']]
        hil_norm = NORMALIZATION['hilbert_coronal_std']
        img_norm = NORMALIZATION['images_99']
        slist[0] = slist[0]/hil_norm
        slist[1] = slist[1]/img_norm
        slist[2] = slist[2]/img_norm
        return {
            'hil': slist[0],
            'sparse': slist[1],
            'img': slist[2],
        }


class ToTensor:
    def __call__(_, sample):
        slist = [sample['hil'], sample['sparse'], sample['img']]
        slist[0] = np.transpose(slist[0], (2, 0, 1))
        slist[1] = np.transpose(slist[1], (2, 0, 1))
        slist[2] = np.transpose(slist[2], (2, 0, 1))
        slist = [torch.from_numpy(si).float() for si in slist]
        return {
            'hil': slist[0],
            'sparse': slist[1],
            'img': slist[2],
        }


class RandomVerticalFlip:
    def __call__(_, sample):
        slist = [sample['hil'], sample['sparse'], sample['img']]
        if random.random() > 0.5:
            slist = [s[:, ::-1] for s in slist]
        return {
            'hil': np.ascontiguousarray(slist[0]),
            'sparse': np.ascontiguousarray(slist[1]),
            'img': np.ascontiguousarray(slist[2]),
        }


class MyDataset(torch.utils.data.Dataset):
    # filename of the image volume
    def __init__(self, filename, plane: HilbertPlane,
                 transform: Optional[Callable] = None):
        self.plane = plane
        self.hilbert_volume = None
        self.sparse_volume = None
        self.image_volume = None
        self.filename = filename
        data_path = DATA_DIRS['views_36']
        ds_shape = nib.load(pjoin(
            data_path,
            f'{filename[:-7]}_{plane.name.lower()}.nii.gz',
        ))
        ds_shape = ds_shape.header.get_data_shape()
        self.ds_len = ds_shape[plane.value]
        self.transform = transform

    def __len__(self):
        return self.ds_len

    @staticmethod
    def _reduce_zdim(volume):
        zdiff = volume.shape[-2] - 512
        if zdiff > 0:  # bigger -> cut
            front_cut = zdiff//2
            back_cut = zdiff - front_cut
            volume = volume[..., front_cut:-back_cut, :]
        return volume

    @staticmethod
    def _make_square(image):
        zdiff = image.shape[-2] - 512
        if zdiff < 0:  # smaller -> pad zeros
            front_pad = -zdiff//2
            back_pad = -zdiff - front_pad
            image = np.pad(
                image,
                ((0, 0),)*(len(image.shape)-2) +
                ((front_pad, back_pad),) + ((0, 0),),
            )
        return image

    def _load_ds_if_none(self):
        hilbert_volume = self.hilbert_volume
        if hilbert_volume is None:
            hilbert_volume = load_nib(pjoin(
                DATA_DIRS['views_36'],
                f'{self.filename[:-7]}_{self.plane.name.lower()}.nii.gz',
            ))
            hilbert_volume = MyDataset._reduce_zdim(hilbert_volume)

        sparse_volume = self.sparse_volume
        if sparse_volume is None:
            sparse_volume = load_nib(pjoin(
                DATA_DIRS['fdk_36'],
                self.filename,
            ))[..., None]
            sparse_volume = MyDataset._reduce_zdim(sparse_volume)

        image_volume = self.image_volume
        if image_volume is None:
            image_volume = load_nib(pjoin(
                DATA_DIRS['datasets'],
                self.filename,
            ))[..., None]
            image_volume = MyDataset._reduce_zdim(image_volume)

        return hilbert_volume, sparse_volume, image_volume

    def unload_datasets(self):
        self.hilbert_volume = None
        self.sparse_volume = None
        self.image_volume = None

    def __getitem__(self, idx):
        self.hilbert_volume, self.sparse_volume, self.image_volume = \
            self._load_ds_if_none()

        hilbert_slice = self.hilbert_volume[idx] \
            if self.plane is HilbertPlane.SAGITTAL \
            else self.hilbert_volume[:, idx]
        sparse_slice = self.sparse_volume[idx] \
            if self.plane is HilbertPlane.SAGITTAL \
            else self.sparse_volume[:, idx]
        image_slice = self.image_volume[idx] \
            if self.plane is HilbertPlane.SAGITTAL \
            else self.image_volume[:, idx]

        hilbert_slice = MyDataset._make_square(hilbert_slice)
        sparse_slice = MyDataset._make_square(sparse_slice)
        image_slice = MyDataset._make_square(image_slice)

        sample = {
            'hil': hilbert_slice,
            'sparse': sparse_slice,
            'img': image_slice,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class DeepHilbertInverse(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.current_lr = self.hparams.lr
        self.valid_dir = self.hparams.valid_dir
        self.unet = UNet(in_channels=3)
        self.loss = torch.nn.MSELoss()
        self.trafo_train = transforms.Compose([
            RandomVerticalFlip(), Normalize(), ToTensor()])
        self.trafo_valid = transforms.Compose([Normalize(), ToTensor()])
        self.example_input_array = torch.empty(1, 3, 512, 512)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.8,
            min_lr=self.hparams.end_lr,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, _):
        hilbert, sparse, image = batch['hil'], batch['sparse'], batch['img']
        hilbert = torch.cat([
            hilbert,
            sparse,
        ], dim=1)
        prediction = self(hilbert)
        loss = self.loss(prediction, image)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        hilbert, sparse, image = batch['hil'], batch['sparse'], batch['img']
        hilbert = torch.cat([
            hilbert,
            sparse,
        ], dim=1)
        prediction = self(hilbert)
        loss = self.loss(prediction, image)

        if self.current_epoch % 5 == 0 and \
           batch_idx < 500 and \
           batch_idx % 10 == 0:
            os.makedirs(
                pjoin(self.valid_dir, f'{self.current_epoch}'),
                exist_ok=True,
            )
            image = image.cpu().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir,
                      f'{self.current_epoch}/{batch_idx}_out_gt.png'),
                image/image.max()*255,
            )

            prediction = prediction.cpu().float().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir,
                      f'{self.current_epoch}/{batch_idx}_out_pred.png'),
                prediction/image.max()*255,
            )

        return {'val_loss': loss}

    def create_dataset(self, filename: str, plane: HilbertPlane,
                       validation: bool) -> MyDataset:
        return MyDataset(
            filename,
            plane,
            transform=self.trafo_valid if validation else self.trafo_train,
        )

    def train_dataloader(self) -> DataLoader:
        full_dataset = ConcatDataset(*([
            self.create_dataset(f, HilbertPlane.SAGITTAL, validation=False)
            for f in self.hparams.train_files
        ] + [
            self.create_dataset(f, HilbertPlane.CORONAL, validation=False)
            for f in self.hparams.train_files
        ]), randomize_subset_idx=True)

        return DataLoader(full_dataset,
                          shuffle=False,  # handled by ConcatDataset
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        full_dataset = ConcatDataset(*([
            self.create_dataset(f, HilbertPlane.CORONAL, validation=True)
            for f in self.hparams.valid_files
        ] + [
            self.create_dataset(f, HilbertPlane.SAGITTAL, validation=True)
            for f in self.hparams.valid_files
        ]))

        return DataLoader(full_dataset,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('training', avg_loss)
        self.log('lr', self.current_lr)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.current_lr = optimizer.param_groups[0]['lr']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--valid_dir', type=str, default='valid')
        parser.add_argument('--end_lr', type=float, default=1e-5)
        parser.add_argument('--train_files', type=list)
        parser.add_argument('--valid_files', type=list)
        return parser


def main():
    parser = ArgumentParser()
    parser = DeepHilbertInverse.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.lr = 5e-2
    hparams.end_lr = 1e-2
    hparams.max_epochs = 300
    hparams.batch_size = CONFIGURATION['batch_size']
    hparams.data_dir = DATA_DIRS['datasets']
    hparams.valid_dir = 'valid_hilbert_inverse_3chan_sparse' \
        + ('_pre' if hparams.pretrained else '')
    with open('train_valid.json') as json_file:
        json_dict = json.load(json_file)
        hparams.train_files = json_dict['train_files']
        hparams.valid_files = json_dict['valid_files']

    model = DeepHilbertInverse(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.valid_dir,
        monitor='val_loss',
        save_last=True,
    )
    trainer = Trainer(
        precision=CONFIGURATION['precision'],
        progress_bar_refresh_rate=CONFIGURATION['progress_bar_refresh_rate'],
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=hparams.max_epochs,
        terminate_on_nan=True,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
