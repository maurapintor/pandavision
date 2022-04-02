import os

import flask

from app.adv.evaluation_manager import EvaluationManager
from app.config import config
from rq.registry import FinishedJobRegistry, StartedJobRegistry, DeferredJobRegistry
from rq import Queue

DATA_FOLDER = os.path.join(config.PROJECT_ROOT, config.DATA_DIR)

status_handling_dict = {
    "started": (lambda: StartedJobRegistry().get_job_ids(), lambda: 1),
    "finished": (lambda: FinishedJobRegistry().get_job_ids(), lambda: FinishedJobRegistry().cleanup()),
    "queued": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='default').empty()),
    "failed": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='failed').empty()),
    "deferred": (lambda: DeferredJobRegistry().get_job_ids(), lambda: DeferredJobRegistry().cleanup()),
}


def create_evaluation_manager(**kwargs):
    em = EvaluationManager(
        dataset_id=os.path.join(DATA_FOLDER, 'data.h5'),
        attack=kwargs.get("attack", None),
        attack_params=kwargs.get("attack-params", None),
        model_id=os.path.join(DATA_FOLDER, 'model.onnx'),
        metric=kwargs.get("performance-metric", "classification-accuracy"),
        perturbation_values=kwargs.get("perturbation-values", None),
        evaluation_mode=kwargs.get("evaluation-mode", "complete"),
        task=kwargs.get("task", "classification"),
        indexes=kwargs.get("indexes", None),
        preprocessing=kwargs.get("preprocessing", None),
    )
    return em


def get_args(form):
    model = form.data['model']
    dataset = form.data['dataset']
    preprocessing = form.data['addpreprocessing']
    if preprocessing == 'default':
        preprocessing = None
    elif preprocessing == 'none':
        preprocessing = dict()
    elif preprocessing == 'custom':
        preprocessing = {
            'mean': (
                form.data['preprocess_mean_R'],
                form.data['preprocess_mean_G'],
                form.data['preprocess_mean_B'],
            ),
            'std': (
                form.data['preprocess_std_R'],
                form.data['preprocess_std_G'],
                form.data['preprocess_std_B'],
            )
        }

    attack_type = form.data['attack']
    if attack_type in ['pgd-linf', 'pgd-l2']:
        attack_params = form.pgd_form.data
    elif attack_type in ['cw']:
        attack_params = form.cw_form.data
    else:
        attack_params = {}
    attack_params = {k: v for k, v in attack_params.items()
                     if v is not None}
    if 'csrf_token' in attack_params:
        del attack_params['csrf_token']
    evaluation_mode = form.data['eval_mode']
    perturbation_values = flask.request.form.getlist('pertpicker')
    perturbation_values.insert(0, '0')
    perturbation_values = list(map(float, perturbation_values))
    args = {'trained-model': model,
            'dataset': dataset,
            'metric': 'classification-accuracy',
            'attack': attack_type,
            'attack-params': attack_params,
            'task': 'classification',
            'preprocessing': preprocessing,
            'evaluation-mode': evaluation_mode,
            'perturbation-values': perturbation_values,
            }
    return args
