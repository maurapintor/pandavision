import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

SHARED_DATA_FOLDER = os.getenv('SHARED_DATA_FOLDER', 'data')


class H5Dataset(TorchDataset):

    def __init__(self, path=None, use_case=None, num_samples=None,
                 indexes=None):
        super(H5Dataset, self).__init__()

        self.transf_composed = transforms.ToTensor()

        self._use_case = use_case
        if self._use_case not in ['classification', 'detection',
                                  'segmentation']:
            raise ValueError("Use case not known: {}".format(self._use_case))

        self._file_path = path  # TODO check if not defined, return error in case

        self._file = h5py.File(self._file_path, 'r')

        if indexes is not None:  # check first if indexes are passed
            sample_indexes = indexes
        elif num_samples is not None:
            sample_indexes = range(min(num_samples, self._file['samples'].shape[0]))
            # shuffle
            sample_indexes = np.array(sample_indexes)
            np.random.shuffle(sample_indexes)
        else:
            sample_indexes = None

        if sample_indexes is None:
            if len(self._file['samples'].shape) > 2:
                self._samples = self._file['samples'][:]
            else:
                self._samples = self._file['samples'][:]
        else:
            self._samples = self._file['samples'][sorted(sample_indexes), ...]
        y_data = self._file['labels'][:]

        if self._use_case == 'classification':
            if sample_indexes is not None:
                self._labels = y_data[sorted(sample_indexes)]
            else:
                self._labels = y_data
        else:
            raise ValueError("Use case not found.")

        self.classes = np.unique(self._labels)
        self._file.close()

    def __getitem__(self, index):
        input = self.transf_composed(self._samples[index, ...])
        return input, self._labels[index]

    def __len__(self):
        return self._samples.shape[0]


class CustomDatasetLoader:

    def __init__(self, batch_size, shuffle=False, path=None, use_case=None,
                 num_samples=None, indexes=None):
        self._validation_dataset = H5Dataset(path=path,
                                             use_case=use_case,
                                             num_samples=num_samples,
                                             indexes=indexes)

        self._batch_size = batch_size
        self._shuffle = shuffle

    def get_data(self):
        validation_loader = torch.utils.data.DataLoader(self._validation_dataset,
                                                        batch_size=self._batch_size,
                                                        shuffle=self._shuffle)
        return validation_loader

    @property
    def validation_dataset(self):
        return self._validation_dataset
