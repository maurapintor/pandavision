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
        self.model = ConvertModel(onnx_model, experimental=True)

    def torch_tensor(self, x):
        """
        Converts an input numpy array to a pytorch tensor.

        :param x: input numpy array
        :return: torch tensor, loaded in the device specified during initialization
        """
        return torch.from_numpy(x).to(self.device)

    def predict(self, x, return_decision_function=False):
        """
        Returns the prediction for the batch of samples x.

        :param return_decision_function: if set to True, the method returns the output scores
            along with the prediction results
        :param x: input batch. Should be a numpy array of size (batch_size, *input_shape)

        :return y: predictions
        :return scores: logits (returned only if return_decision_function is set to True)
        """
        x = self.torch_tensor(x)
        outputs = self.model(x)
        pred_label = torch.argmax(outputs, dim=1)
        if return_decision_function is True:
            return outputs.cpu().detach().numpy(), pred_label.cpu().numpy()
        # else return only the labels
        return pred_label.cpu().numpy()


    def _validate_input(self):
        """Checks format and size of input."""
        # TODO implement this check and return warning if it fails
        #   eventually, apply the _transform method for fixing input
        pass

    def _transform(self):
        """Transforms the input in order to be compliant with the model input shape."""
        # TODO (maybe) move to data_loader section
        pass
