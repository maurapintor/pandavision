import importlib
import os

import onnx
import torch
from onnx2pytorch.convert.model import ConvertModel

class TorchONNXLoader:
    """Loads a ONNX model as pytorch model."""

    def __init__(self, onnx_path, device=None, use_case='classification'):
        """
        Loads an onnx model for inference and for performing security evaluation. Wraps a pytorch model
        with methods predict  for inference and loss_gradient for computing gradients with respect to
        input (used for performing the adversarial attacks).

        :param onnx_path: path of the onnx file to use
        :param device: use 'cuda' for using GPU. Default is 'cpu'
        """
        self.onnx = onnx_path
        onnx_model = onnx.load(self.onnx)
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ConvertModel(onnx_model)
        self.model.eval()



    def _validate_input(self):
        """Checks format and size of input."""
        # TODO implement this check and return warning if it fails
        #   eventually, apply the _transform method for fixing input
        pass

    def _transform(self):
        """Transforms the input in order to be compliant with the model input shape."""
        # TODO (maybe) move to data_loader section
        pass
