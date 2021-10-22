from abc import abstractmethod, ABC

import numpy as np
import torch


class AttackBase(ABC):
    def __init__(self, model, lb, ub):
        self.model = model
        self.lb = lb
        self.ub = ub

    def add_noise(self, x, eps):
        """
        Generate noisy inputs and return them in an array.

        :param x: An array with the original inputs.
        :param eps: Maximum amount of perturbation to be added to the images

        :return: An array holding the noisy examples.
        """
        if eps == 0:
            return x
        else:
            eps *= (x.max() - x.min())
        x_noise = x + torch.randn(x.shape).sign() * eps
        x_noise = np.clip(x_noise, x.min(), x.max())
        return x_noise

    @abstractmethod
    def run(self, x, y, eps):
        ...

    @abstractmethod
    def evaluate_perf(self, x, labels):
        ...

    @abstractmethod
    def generate_figure(self, x, x_adv, y, figure_path, figure_name):
        ...
