import os

from secml.ml.classifiers import CClassifierPyTorch

from models.models import AlgorithmConfiguration
from onnx_pytorch.pytorch_importer import TorchONNXLoader

SHARED_DATA_FOLDER = os.getenv('SHARED_DATA_FOLDER', 'data')

class ModelLoader:

    def __init__(self,
                 algorithm_id=None,
                 model_path=None,
                 input_shape=None,
                 output_shape=None):

        if algorithm_id is not None:
            algorithm = AlgorithmConfiguration.objects.get(id=algorithm_id)
            self._model_path = os.path.join(SHARED_DATA_FOLDER, algorithm.onnx_trained)
        elif model_path is not None:
            self._model_path = model_path
        else:
            raise ValueError("Either model path or id should be passed.")

        self._input_shape = input_shape
        self._output_shape = output_shape

    @property
    def model(self):
        return self._model

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
        model = TorchONNXLoader(self._model_path)
        model.load_model()
        self._model = model.load_model()

    def pytorch_to_secml(self):
        self._model = CClassifierPyTorch(model=self._model, pretrained=True, input_shape=self._input_shape)
