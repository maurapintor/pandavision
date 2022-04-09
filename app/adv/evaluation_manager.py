"""This class is used whenever a security evaluation job is requested. It creates the security
evaluation job starting from the parameters of the request."""
import bisect
import os
from typing import List, Union

import numpy as np
import torch
from secml.array import CArray

from .classification.attack_classification import AttackClassification, SUPPORTED_ATTACKS
from .dataset_loader import CustomDatasetLoader
from .model_loader import ModelLoader

ATTACK_CHOICES = {
    'linf': [('PGD', 'pgd-linf'), ('Random', 'noise-linf')],
    'l2': [('PGD', 'pgd-l2'), ('CW', 'cw'), ('Random', 'noise-l2')],
}

PERT_SIZES = {
    'linf': [("1/255", 1 / 255), ("2/255", 2 / 255), ("4/255", 4 / 255), ("8/255", 8 / 255), ("16/255", 16 / 255)],
    'l2': [("0.01", 0.01), ("0.02", 0.02), ("0.05", 0.05), ("0.1", 0.1), ("0.2", 0.2), ("0.5", 0.5)],
}

NUM_CLASSES_TO_SHOW = 10


class EvaluationManager:
    def __init__(self, dataset_id: str,
                 model_id: str,
                 metric: str = None,
                 attack: str = None,
                 perturbation_values: List[Union[int, float]] = None,
                 evaluation_mode: str = 'complete',
                 task: str = 'classification',
                 indexes: List[int] = None,
                 preprocessing: dict = None,
                 attack_params: dict = None):
        """Performs security evaluation for a given model and dataset.

        :param dataset_id: Path of the dataset.
        :param model_id: Path of the model.
        :param metric: Metric to use for the evaluation. Currently,
            only `classification-accuracy` is available as metric.
        :param attack: Algorithm to use for attacking the model.
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
                self._num_samples = 3
        else:
            self._num_samples = None

        # load dataset and model
        if dataset_id is not None:
            self._load_dataset_by_id()
        if model_id is not None:
            self._load_model_by_id()

        if attack in SUPPORTED_ATTACKS:
            self._attack_cls = attack
        else:
            raise ValueError(f"Attack type {attack} not understood. "
                             f"It should be one of: {list(SUPPORTED_ATTACKS.keys())}.")

        self._attack_params = attack_params if attack_params is not None else dict()

        self._metric = metric
        if self._metric not in ['classification-accuracy']:
            raise ValueError("Evaluation metric {} not understood. "
                             "It should be one of: 'classification-accuracy' ... ."
                             "".format(self._metric))

        if self._task == 'classification' and self._metric != 'classification-accuracy':
            raise ValueError("Please, use 'classification-accuracy' as detection metric")

        if perturbation_values is not None:
            self._perturbation_values = perturbation_values
        else:
            # default value
            self._perturbation_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05]

        # initialize caches
        self.cached_is_adv = None
        self.cached_min_distance = None
        self._batch_is_cached = None
        self.attack_losses = None
        self.attack_distances = None
        self.attack_scores = None
        self.adv_examples = None

        if self._num_samples is None:
            self._num_samples = self._validation_loader.dataset._samples.shape[0]

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

        acc_results = []
        batch_size = self._validation_loader.batch_size
        self._batch_is_cached = [False for _ in range(len(self._validation_loader))]
        self.cached_is_adv = torch.full(size=(self._num_samples,), fill_value=False)
        self.cached_min_distance = torch.full(size=(self._num_samples,), fill_value=np.inf, dtype=torch.float64)

        steps = self._attack_params.get('steps', 0)

        self.attack_losses = torch.empty(
            size=(self._num_samples, steps))
        self.attack_distances = torch.empty(
            size=(self._num_samples, steps))
        self.attack_scores = torch.empty(
            size=(self._num_samples, steps, NUM_CLASSES_TO_SHOW * 2))
        self.orig_samples = torch.empty(
            size=(self._num_samples, *self.input_shape))
        self.adv_examples = torch.empty(
            size=(self._num_samples, *self.input_shape))

        for eps_idx, eps in enumerate(self._perturbation_values):
            acc = []
            for batch_idx, (samples, labels) in enumerate(self._validation_loader):
                min_batch_index = batch_idx * batch_size
                max_batch_index = min((batch_idx + 1) * batch_size, self._num_samples)
                to_replace = np.arange(min_batch_index, max_batch_index)

                if self.attack.is_min_distance(self._attack_cls):
                    if self._batch_is_cached[batch_idx] is False:
                        is_adv, adv_points = self.attack.run(samples, labels, self._attack_cls, self._attack_params,
                                                             eps)
                        adv_points = torch.from_numpy(adv_points)
                        is_adv = torch.from_numpy(is_adv)
                        distances = (adv_points - samples).view(adv_points.shape[0], -1).norm(dim=1,
                                                                                              p=self.attack.attack_norm(
                                                                                                  self._attack_cls))
                        distances[torch.logical_not(is_adv)] = np.inf
                        self.cached_is_adv[min_batch_index:
                                           max_batch_index] = is_adv
                        self.cached_min_distance[min_batch_index:
                                                 max_batch_index] = distances
                        if eps > 0:
                            self._batch_is_cached[batch_idx] = True

                        if self.attack.debug_possible is True:
                            if eps_idx > 0:
                                x_seq = self.attack.attack.x_seq
                                self.attack_losses[to_replace, :] = \
                                    to_torch(self.attack.attack.objective_function(
                                        x_seq)).type(torch.FloatTensor)
                                self.attack_distances[to_replace, :] = \
                                    to_torch((x_seq[0, :] - x_seq).norm_2d(
                                        axis=1,
                                        order=self.attack.attack_norm(self._attack_cls)).ravel()).type(torch.FloatTensor)
                                all_attack_scores = to_torch(self._model.decision_function(x_seq)).type(torch.FloatTensor)
                                max_beginning, indices_max_beginning = all_attack_scores[0, :].topk(NUM_CLASSES_TO_SHOW)
                                max_final, indices_max_final = all_attack_scores[-1, :].topk(NUM_CLASSES_TO_SHOW)
                                take_indexes = torch.cat([indices_max_beginning, indices_max_final])
                                self.attack_scores[to_replace, :, :] = all_attack_scores[:, take_indexes]
                                self.orig_samples[to_replace, ...] = samples.type(self.orig_samples.dtype)
                                self.adv_examples[to_replace, ...] = adv_points.type(self.adv_examples.dtype)
                    else:
                        pass
                else:
                    is_adv_batch = self.cached_is_adv[min_batch_index:
                                                      max_batch_index]
                    distances_batch = self.cached_min_distance[min_batch_index:
                                                               max_batch_index]
                    not_yet_adv = torch.nonzero(torch.logical_not(is_adv_batch))
                    if len(not_yet_adv) > 0:
                        is_adv, adv_points = self.attack.run(samples[not_yet_adv, ...],
                                                             labels[not_yet_adv, ...],
                                                             self._attack_cls, self._attack_params, eps)
                        is_adv = torch.from_numpy(is_adv)
                        is_adv_batch[not_yet_adv[is_adv]] = is_adv
                        adv_points = torch.from_numpy(adv_points)
                        self.cached_is_adv[min_batch_index:
                                           max_batch_index] = is_adv_batch
                        distances = (adv_points - samples).view(adv_points.shape[0], -1).norm(dim=1,
                                                                                              p=self.attack.attack_norm(
                                                                                                  self._attack_cls))
                        distances_batch[is_adv_batch] = distances
                        self.cached_min_distance[min_batch_index:
                                                 max_batch_index] = distances_batch
                        to_replace = to_replace[not_yet_adv]

                        if self.attack.debug_possible is True:
                            if eps_idx > 0:
                                x_seq = self.attack.attack.x_seq
                                self.attack_losses[to_replace, :] = \
                                    to_torch(self.attack.attack.objective_function(
                                        x_seq))
                                self.attack_distances[to_replace, :] = \
                                    to_torch((x_seq[0, :] - x_seq).norm_2d(
                                        axis=1,
                                        order=self.attack.attack_norm(self._attack_cls)).ravel())
                                all_attack_scores = to_torch(self._model.decision_function(x_seq))
                                max_beginning, indices_max_beginning = all_attack_scores[0, :].topk(NUM_CLASSES_TO_SHOW)
                                max_final, indices_max_final = all_attack_scores[-1, :].topk(NUM_CLASSES_TO_SHOW)
                                take_indexes = torch.cat([indices_max_beginning, indices_max_final])
                                self.attack_scores[to_replace, :, :] = all_attack_scores[:, take_indexes]
                                self.orig_samples[to_replace, ...] = samples
                                self.adv_examples[to_replace, ...] = adv_points

                perf = torch.logical_or(torch.logical_not(self.cached_is_adv),
                                        torch.logical_and(self.cached_is_adv, self.cached_min_distance > eps)) \
                    .type(torch.FloatTensor).mean()
                acc.append(perf)
            avg_acc = np.array(acc).mean()
            acc_results.append(avg_acc)
        acc_results = np.array(acc_results)
        results = {'acc_results': acc_results}

        # add cached values, if present
        if self.attack.debug_possible is True:
            results.update({
                'attack_losses': self.attack_losses,
                'attack_scores': self.attack_scores,
                'attack_distances': self.attack_distances,
                'orig_samples': self.orig_samples,
                'adv_examples': self.adv_examples,
            })

        response = self.prepare_response(results)
        return response

    def prepare_response(self, results: dict):
        """
        Returns the response object for a security evaluation.
        :param performances: array containing a perf value for
            each of the perturbation values
        """

        performances = results['acc_results']
        if performances[0] == 0:
            sec_value = np.array(-1)
        else:
            sec_value = np.mean(performances) / performances[0]
        sec_levels = ((0.33, 0.66, 1.5), ("low", "medium", "high"))

        # compute sec-level
        sec_level = sec_levels[1][bisect.bisect_left(sec_levels[0], sec_value)]

        eval_results = {"sec-level": sec_level,
                        "sec-value": sec_value.item(),
                        "sec-curve": {
                            "x-values": ["{:.3f}".format(v) for v in self._perturbation_values],
                            "y-values": performances.tolist()}}

        for k in ['attack_distances', 'attack_losses', 'attack_scores', 'adv_examples', 'orig_samples']:
            if k in results:
                eval_results.update({k: results[k].tolist()})

        eval_results['debug_possible'] = self.attack.debug_possible
        return eval_results


def to_torch(x):
    if isinstance(x, CArray):
        x = x.tondarray()
    x = torch.from_numpy(x)
    return x
