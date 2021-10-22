import os

import numpy as np
from secml.adv.attacks.evasion import CFoolboxPGDLinf
from secml.array import CArray
from secml.data import CDataset
from secml.ml.peval.metrics import CMetricAccuracy

from adv.attack_base import AttackBase


class AttackClassification(AttackBase):
    def __init__(self, model, lb, ub):
        self.classes = range(1000)
        super(AttackClassification, self).__init__(model, lb, ub)

    def run(self, x, y, eps):
        if eps == 0:
            return x
        self.prepare_attack(x, eps)
        x, y = x.numpy().astype(np.float32), y.numpy()
        orig_shape = x.shape
        data = CArray(x.reshape(x.shape[0], -1))
        labels = CArray(y)

        ts = CDataset(data, labels)

        y_pred, _, adv_ds, _ = self.attack.run(ts.X, ts.Y)
        adv_samples = adv_ds.X.tondarray().reshape(orig_shape)

        return adv_samples

    def prepare_attack(self, x, eps):
        max_ = x.max()
        min_ = x.min()
        data_range = max_ - min_
        attack_params = {'epsilons': eps * data_range.item(),
                         'lb': min_,
                         'ub': max_}
        self.attack = CFoolboxPGDLinf(
            classifier=self.model,
            y_target=None,
            **attack_params)

    def evaluate_perf(self, x, labels):
        metric = CMetricAccuracy()
        data = CArray(x.reshape(x.shape[0], -1))
        labels = CArray(labels)
        ts = CDataset(data, labels)
        preds = self.model.predict(ts.X)
        acc = metric.performance_score(ts.Y, preds)
        return acc

    def generate_figure(self, x, x_adv, y, figure_path, figure_name):
        import matplotlib.pyplot as plt
        perturbation = x - x_adv
        perturbation /= abs(perturbation).max()
        perturbation = perturbation.squeeze().detach().numpy()

        preds, scores = self.model.predict(x.reshape(x.shape[0], -1), return_decision_function=True)
        adv_preds, adv_scores = self.model.predict(x_adv.reshape(x_adv.shape[0], -1), return_decision_function=True)

        preds = preds.item()
        adv_preds = adv_preds.item()

        scores = scores.tondarray().ravel()
        adv_scores = adv_scores.tondarray().ravel()
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.title("original image\n"
                  "label: {}\n"
                  "pred: {}"
                  "".format(self.classes[y], self.classes[preds]))
        plt.imshow(x.squeeze(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 4, 2)
        plt.imshow(x_adv.squeeze(), cmap='gray')
        plt.title("perturbed image\n"
                  "label: {}\n"
                  "pred: {}"
                  "".format(self.classes[y], self.classes[adv_preds]))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 4, 3)
        plt.imshow(perturbation, vmin=-1, vmax=1, cmap='seismic')
        plt.title("perturb")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 4, 4)
        plt.title("predictions")
        r = np.arange(len(self.classes))
        w = 0.35  # width of the bars
        plt.bar(r - w / 2, scores, w, label='original', color='tab:green')
        plt.bar(r + w / 2, adv_scores, w, label='perturbed', color='tab:red')
        plt.xlabel('scores')
        plt.xticks(r, self.classes)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, "{}.pdf").format(figure_name), format='pdf')
        plt.savefig(os.path.join(figure_path, "{}.png").format(figure_name), format='png')
        plt.close()
