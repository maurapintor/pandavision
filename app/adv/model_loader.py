import os

import numpy as np
from secml.array import CArray
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerMeanStd
from .pytorch_importer import TorchONNXLoader

DEFAULT_PREPROCESSING_MEAN = (0.485, 0.456, 0.406,)
DEFAULT_PREPROCESSING_STD = (0.229, 0.224, 0.225,)


class ModelLoader:

    def __init__(self, model_path=None, input_shape=None, preprocessing=None):

        self._model_path = model_path  # todo check if defined, return error otherwise
        self._input_shape = input_shape
        if isinstance(preprocessing, dict):
            mean = preprocessing.get('mean', None)
            std = preprocessing.get('std', None)
            if all([mean, std]):
                normalizer = CNormalizerMeanStd(mean=mean, std=std)
                normalizer._mean = mean
                normalizer._std = std
                self._preprocessor = normalizer
            else:
                self._preprocessor = None
        elif preprocessing is None:
            self._preprocessor = CNormalizerMeanStd(mean=DEFAULT_PREPROCESSING_MEAN, std=DEFAULT_PREPROCESSING_STD)
            self._preprocessor._mean = DEFAULT_PREPROCESSING_MEAN
            self._preprocessor._std = DEFAULT_PREPROCESSING_STD

    @property
    def model(self):
        return self._model

    @property
    def input_shape(self):
        return self._input_shape

    def load_model(self):
        self.onnx_to_pytorch()
        self.pytorch_to_secml()
        return self._model

    def onnx_to_pytorch(self):
        """
        Extracts the trained model from the provided onnx file, with UNICA's parser.
        :return: the instantiated model obj
        """
        self._model = TorchONNXLoader(self._model_path).model

    def pytorch_to_secml(self):
        # todo handle gracefully pretrained_classes
        self._model = CClassifierPyTorch(model=self._model,
                                         pretrained=True,
                                         input_shape=self.input_shape,
                                         batch_size=1,
                                         preprocess=self._preprocessor)
