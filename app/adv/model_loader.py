import os

from secml.ml.classifiers import CClassifierPyTorch

from adv.pytorch_importer import TorchONNXLoader

SHARED_DATA_FOLDER = os.getenv('SHARED_DATA_FOLDER', 'data')

class ModelLoader:

    def __init__(self, model_path=None, input_shape=None):

        self._model_path = model_path  # todo check if defined, return error otherwise
        self._input_shape = input_shape

    @property
    def model(self):
        return self._model

    @property
    def input_shape(self):
        return self._input_shape

    def load_model(self, secml=False):
        self.onnx_to_pytorch()
        if secml:
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
        self._model = CClassifierPyTorch(model=self._model, pretrained=True, input_shape=self.input_shape)

