"""This class is used whenever a security evaluation job is requested. It creates the security
evaluation job starting from the parameters of the request."""
import bisect
import json
import os
from typing import List, Union

import numpy as np

from .classification.attack_classification import AttackClassification
from .dataset_loader import CustomDatasetLoader
from .model_loader import ModelLoader


class EvaluationManager:
    def __init__(self, dataset_id: str,
                 model_id: str,
                 metric: str = None,
                 perturbation_type: str = None,
                 perturbation_values: List[Union[int, float]] = None,
                 evaluation_mode: str = 'complete',
                 task: str = 'classification',
                 indexes: List[int] = None,
                 preprocessing: dict = None):
        """Performs security evaluation for a given model and dataset.

        :param dataset_id: Path of the dataset.
        :param model_id: Path of the model.
        :param metric: Metric to use for the evaluation. Currently,
            only `classification-accuracy` is available as metric.
        :param perturbation_type: Type of perturbation to add to the
            input images. One of `max-norm` (Infinity-norm
            perturbation-type, worst case, gradient-based),
            `random` (Infinity-norm random perturbation).
        :param perturbation_values: List of integers containing the
            x-values for the security evaluation curve. For each point,
            a perturbation of type `perturbation-type` with constrained
            norm equal to the x-value will be applied.
        :param evaluation_mode: Indicates a particular configuration
            for the experiment. One of `fast`, `complete`. The `fast`
            evaluation will run the experiment on a small set of samples
            (100), while the complete will run it either on the complete
            dataset, or in all the samples indicated in the `indices`
            field.
        :param task: Task performed by the classifier. This can be one
            of `classification` or `detection`, and will determine the
            attack model/scenario to use for the evaluation.
        :param indexes: List of indexes for specifying which samples to 
            use and in what order for the evaluation. It might be
            useful for reproducing particular results or tests on a
            specific subset of samples.
        :param config_file: Path of json file to use as configuration
            for the experiment. It could contain anchors, additional
            parameters, task information.
        :param preprocessing: Dictionary containing the key `mean` and
            `std` for defining a preprocessing standardizer block.
        """

        self._dataset_id = dataset_id
        self._model_id = model_id
        if not os.path.isfile(self._dataset_id):
            raise ValueError("Dataset {} is not a valid path."
                             "".format(self._dataset_id))
        if not os.path.isfile(self._model_id):
            raise ValueError("Model {} is not a valid path."
                             "".format(self._model_id))
        self._task = task
        self._indexes = indexes
        self._preprocessing = preprocessing
        self._evaluation_mode = evaluation_mode
        if self._evaluation_mode == 'fast':
            if self._task == 'classification':
                self._num_samples = 10
        else:
            self._num_samples = None

        # load dataset and model
        if dataset_id is not None:
            self._load_dataset_by_id()
        if model_id is not None:
            self._load_model_by_id()

        if perturbation_type in ["max-norm", "random"]:
            self._perturbation_type = perturbation_type
        else:
            raise ValueError("Perturbation type {} not understood. "
                             "It should be one of: 'max-norm', 'random'."
                             "".format(perturbation_type))

        self._metric = metric
        if self._metric not in ['classification-accuracy', 'map', 'iou']:
            raise ValueError("Evaluation metric {} not understood. "
                             "It should be one of: 'classification-accuracy', 'map' ... ."
                             "".format(self._metric))

        if self._task == 'classification' and self._metric != 'classification-accuracy':
            raise ValueError("Please, use 'classification-accuracy' as detection metric")


        if perturbation_values is not None:
            self._perturbation_values = perturbation_values
        else:
            # default value
            self._perturbation_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05]

    def _load_dataset_by_id(self):
        # Dataset can be loaded from a local file path
        data_loader = CustomDatasetLoader(path=self._dataset_id,
                                          use_case=self._task,
                                          batch_size=1,
                                          shuffle=False,
                                          num_samples=self._num_samples,
                                          indexes=self._indexes)

        self._validation_loader = data_loader.get_data()

        self.data_max, self.data_min = data_loader.validation_dataset._samples.max(), \
                                       data_loader.validation_dataset._samples.min()

        self.input_scale = self.data_max - self.data_min
        self.input_shape = self._validation_loader.dataset._samples[0].shape
        self.n_output_classes = len(self._validation_loader.dataset.classes)

    def _load_model_by_id(self):
        self._model = ModelLoader(model_path=self._model_id, input_shape=self.input_shape,
                                  preprocessing=self._preprocessing).load_model()

    def prepare_attack(self):
        if self._task == 'classification':
            self.attack = AttackClassification(model=self._model, lb=self.data_min, ub=self.data_max)
        else:
            raise ValueError("Attack for task {} is not supported yet!".format(self._task))

    def sec_eval_curve(self):
        self.prepare_attack()

        if not isinstance(self._perturbation_values, list):
            raise ValueError("Perturbation values should "
                             "be a list of floats. Received {}"
                             "".format(self._perturbation_values))

        results = []
        for eps in self._perturbation_values:
            acc = []
            for samples, labels in self._validation_loader:
                if self._perturbation_type == 'max-norm':
                    adv_points = self.attack.run(samples, labels, eps)
                else:
                    adv_points = self.attack.add_noise(samples, eps)
                perf = self.attack.evaluate_perf(adv_points, labels)
                acc.append(perf)
            avg_acc = np.array(acc).mean()
            results.append(avg_acc)
        results = np.array(results)
        response = self.prepare_response(results)
        return response

    def generate_advx(self, samples, labels, eps):
        self.prepare_attack()
        if self._perturbation_type == 'max-norm':
            adv_points = self.attack.run(samples, labels, eps)
        else:
            adv_points = self.attack.add_noise(samples, eps)
        return adv_points

    def prepare_response(self, performances: np.ndarray):
        """
        Returns the response object for a security evaluation.
        :param performances: array containing a perf value for
            each of the perturbation values
        """
        sec_value = np.mean(performances) / performances[0]
        sec_levels = ((0.33, 0.66, 1.5), ("low", "medium", "high"))

        # compute sec-level
        sec_level = sec_levels[1][bisect.bisect_left(sec_levels[0], sec_value)]

        eval_results = {"sec-level": sec_level,
                        "sec-value": sec_value.item(),
                        "sec-curve": {
                            "x-values": self._perturbation_values,
                            "y-values": performances}}
        return eval_results
