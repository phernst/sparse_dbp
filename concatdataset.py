import random

import numpy as np
import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, randomize_subset_idx=False):
        self.datasets = datasets
        self.cslen = np.concatenate([[0], np.cumsum([len(d) for d in datasets])])
        self.subset_idx = [list(range(len(d))) for d in datasets]
        self.last_ds = 0
        self.randomize_subset_idx = randomize_subset_idx

    def _randomize_subset_idx(self):
        for idx_list in self.subset_idx:
            random.shuffle(idx_list)

    def __len__(self):
        return self.cslen[-1]

    def __getitem__(self, idx):
        if idx == 0 and self.randomize_subset_idx:
            self._randomize_subset_idx()

        ds_idx = np.searchsorted(self.cslen - 1, idx) - 1

        if ds_idx != self.last_ds and hasattr(self.datasets[self.last_ds], 'unload_datasets'):
            self.datasets[self.last_ds].unload_datasets()
            self.last_ds = ds_idx

        pos_idx = idx - self.cslen[ds_idx]
        return self.datasets[ds_idx][self.subset_idx[ds_idx][pos_idx]]
