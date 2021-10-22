import matplotlib.pyplot as plt
import numpy as np
from cleverhans.attacks import FastGradientMethod
from secml.adv.attacks import CAttackEvasionCleverhans


def prepare_attack_classification(perturbation_type, data_max, data_min,
                                  model, ts):
    if perturbation_type == 'max-norm':
        attack_params = {'eps': 0.3,
                         'eps_iter': 0.05,
                         'nb_iter': 20,
                         'clip_max': data_max,
                         'clip_min': data_min,
                         'ord': np.inf,
                         'rand_init': False}
        attack = CAttackEvasionCleverhans(
            classifier=model,
            surrogate_classifier=model,
            y_target=None,
            surrogate_data=ts,
            clvh_attack_class=FastGradientMethod,
            **attack_params)
        param_name = 'attack_params.eps'
    elif perturbation_type == 'random':
        raise NotImplementedError("Sorry, still fixing this.")
    else:
        raise NotImplementedError

def view_classify(img_list, ps_list, eps_list, bounds=(0, 1)):
    """ Function for viewing an image and it's predicted classes."""
    data_min, data_max = bounds
    with plt.style.context("seaborn"):
        fig = plt.figure(figsize=(9, 3 * len(img_list)))

        for i, (img, ps, eps) in enumerate(zip(img_list, ps_list, eps_list)):
            img = img.squeeze()
            ps = ps.squeeze()
            noise = np.transpose((img - img_list[0][0]), (1, 2, 0))
            ps = np.exp(ps) / sum(np.exp(ps))
            img = normalize(img, data_min, data_max)
            ax1 = plt.subplot2grid((len(img_list), 3), (i, 0))
            ax2 = plt.subplot2grid((len(img_list), 3), (i, 1))
            ax3 = plt.subplot2grid((len(img_list), 3), (i, 2))
            ax1.set_title(r"$\epsilon$ = {}%".format((eps * 100)))
            ax2.set_title('Class Probability')
            ax3.set_title("Noise")
            if len(img.shape) == 2:
                ax1.imshow(img)
            else:
                ax1.imshow(np.transpose(img, (1, 2, 0)))
            ax1.axis('off')
            ax2.barh(np.arange(len(ps)), ps)
            ax2.set_aspect(0.1)
            ax2.set_yticks(np.arange(len(ps)))
            ax2.set_yticklabels(np.arange(len(ps)))

            ax2.set_xlim(0, 1.1)

            noise = normalize(noise, 0, 1) / max(eps_list) * eps
            if noise.shape[2] == 1:
                noise = np.squeeze(noise, 2)
            ax3.imshow(noise, cmap='seismic',
                       vmin=-1.0, vmax=1.0)
            ax3.yaxis.tick_right()
            ax3.axis('off')

        fig.tight_layout()
        return fig


def normalize(x, data_min, data_max):
    x -= data_min
    return x / data_max