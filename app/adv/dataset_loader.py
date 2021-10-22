import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from models.models import Dataset

try:
    from plugins.utils.pipeline import Pipeline, Compose
except:
    pass

SHARED_DATA_FOLDER = os.getenv('SHARED_DATA_FOLDER', 'data')


class H5Dataset(TorchDataset):

    def __init__(self, dataset_id=None, path=None, use_case=None, num_samples=None,
                 indexes=None, pipeline_path=None):
        super(H5Dataset, self).__init__()
        if pipeline_path is not None:
            with open(pipeline_path) as f:
                pre_pipeline_config = json.load(f)
            preprocessing_pipeline_instance = Pipeline(pre_pipeline_config)
            batch_transforms = preprocessing_pipeline_instance.get_transforms('batch_transforms', validation=True)
            self.transf_composed = Compose(batch_transforms)
        else:
            self.transf_composed = transforms.ToTensor()

        self._use_case = use_case
        if self._use_case not in ['classification', 'detection',
                                  'segmentation']:
            raise ValueError("Use case not known: {}".format(self._use_case))

        if dataset_id is not None:
            # Retrieve metadata from mongodb
            self._dataset_id = dataset_id
            self._dataset = Dataset.objects.get(id=str(self._dataset_id))
            self._file_path = os.path.join(SHARED_DATA_FOLDER, self._dataset.path)
        elif path is not None:
            self._file_path = path
        else:
            raise ValueError("Either dataset id or path should be specified.")

        self._file = h5py.File(self._file_path, 'r')

        if indexes is not None:  # check first if indexes are passed
            sample_indexes = indexes
        elif num_samples is not None:
            sample_indexes = range(min(num_samples, self._file['X_data'].shape[0]))
            # shuffle
            sample_indexes = np.array(sample_indexes)
            np.random.shuffle(sample_indexes)
        else:
            sample_indexes = None

        if sample_indexes is None:
            if len(self._file['X_data'].shape) > 2:
                self._samples = self._file['X_data'][:]
            else:
                self._samples = self._file['X_data'][:]
        else:
            get_samples = np.zeros(self._file['X_data'].shape[0], dtype=bool)
            get_samples[sample_indexes] = True
            if len(self._file['X_data'].shape) > 2:
                self._samples = self._file['X_data'][get_samples, ...]
            else:
                self._samples = self._file['X_data'][get_samples, ...]
        y_data = self._file['y_data']

        if self._use_case == 'classification':
            if sample_indexes is None:
                self._labels = torch.argmax(torch.from_numpy(y_data[:]), dim=1).flatten().long()
            else:
                self._labels = torch.argmax(torch.from_numpy(y_data[:num_samples]), dim=1).flatten().long()
        elif self._use_case == 'detection':
            if sample_indexes is None:
                self._labels = torch.from_numpy(y_data[:])
            else:
                self._labels = torch.from_numpy(y_data[get_samples, ...])
            self._labels = np.delete(self._labels, np.where(self._labels.sum(axis=2) == 0)[1], axis=1)
        elif self._use_case == 'segmentation':
            if sample_indexes is None:
                self._labels = torch.from_numpy(y_data[:])
            else:
                self._labels = torch.from_numpy(y_data[get_samples, ...])
            self._labels = self._labels.permute(0, 2, 1)
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
                 num_samples=None, indexes=None, pipeline_path=None):
        self._validation_dataset = H5Dataset(path=path,
                                             use_case=use_case,
                                             num_samples=num_samples,
                                             indexes=indexes,
                                             pipeline_path=pipeline_path)

        self._batch_size = batch_size
        self._shuffle = shuffle

    def get_data(self):
        validation_loader = torch.utils.data.DataLoader(self._validation_dataset,
                                                        batch_size=self._batch_size,
                                                        shuffle=self._shuffle)
        return validation_loader
