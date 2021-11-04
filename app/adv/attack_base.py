from abc import abstractmethod, ABC

import numpy as np
import torch


class AttackBase(ABC):
    def __init__(self, model, lb, ub):
        self.model = model
        self.lb = lb
        self.ub = ub

    @abstractmethod
    def run(self, x, y, attack, attack_params, eps):
        ...

    @abstractmethod
    def evaluate_perf(self, x, labels):
        ...

    @abstractmethod
    def generate_figure(self, x, x_adv, y, figure_path, figure_name):
        ...
