import logging
import os

import flask
import numpy as np
from flask import abort, render_template, make_response
from flask_restful import Resource
from rq import Connection
from rq import Queue
from rq.registry import FinishedJobRegistry, StartedJobRegistry, DeferredJobRegistry

from app.adv.evaluation_manager import EvaluationManager
from app.config import config
from app.worker import conn
from forms.sec_eval_form import SecEvalForm

DATA_FOLDER = os.path.join(config.PROJECT_ROOT, config.DATA_DIR)

status_handling_dict = {
    "started": (lambda: StartedJobRegistry().get_job_ids(), lambda: 1),
    "finished": (lambda: FinishedJobRegistry().get_job_ids(), lambda: FinishedJobRegistry().cleanup()),
    "queued": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='default').empty()),
    "failed": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='failed').empty()),
    "deferred": (lambda: DeferredJobRegistry().get_job_ids(), lambda: DeferredJobRegistry().cleanup()),
}


def attack(**kwargs):
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
    eval = em.sec_eval_curve()
    return eval


def fake_attack(**kwargs):
    # DO NOT USE -- used for testing frontend
    eval_results = {"sec-level": "low",
                    "sec-value": 0.4,
                    "sec-curve": {
                        "x-values": [0.0, 0.05, 0.1, 0.2, 0.3, 0.6],
                        "y-values": [1.0, 0.9, 0.8, 0.5, 0.1, 0.0]}}
    return eval_results


class SecurityEvaluations(Resource):

    def get(self):

        form = SecEvalForm()
        return make_response(render_template('sec_eval_select.html', form=form))

    def post(self):
        args = flask.request.json
        if args is None:
            form = SecEvalForm()
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
            stop_value = 4 / 255 if form.data['pert_type'] == 'linf' else 0.2
            pert_values = np.linspace(start=0, stop=stop_value, num=4).tolist()
            args = {'trained-model': model,
                    'dataset': dataset,
                    'metric': 'classification-accuracy',
                    'attack': attack_type,
                    'attack-params': attack_params,
                    'task': 'classification',
                    'preprocessing': preprocessing,
                    'evaluation-mode': evaluation_mode,
                    'perturbation-values': pert_values,
                    }
        with Connection(conn):
            q = Queue(connection=conn, name="sec-evals")
            try:
                job = q.enqueue_call(func=attack, result_ttl=int(config.RESULT_TTL), timeout=int(config.JOB_TIMEOUT),
                                     kwargs=args)
            except Exception as e:
                print(str(e))
                logging.log(logging.WARNING, "Unable to queue the requested process. Redis service unavailable.")
                abort(422, "Unprocessable entry. The server understands the request "
                           "entity, but was unable to process the instructions.")
                return
        return make_response(render_template('sec_eval_output.html', jobID=job.get_id()))

    def delete(self):
        s = flask.request.json.get("status", None)
        with Connection(conn):
            if s is not None:
                if s in status_handling_dict:
                    status_handling_dict[s][1]()
                else:
                    logging.log(level=logging.INFO, msg="Invalid input.")
                    abort(400, "Filtering parameter not understood: status={}. "
                               "Possible statuses are: {}.".format(s, ", ".join(status_handling_dict.keys())))
                    return
            else:
                logging.log(level=logging.INFO, msg="Deleting all processes and queue register information.")
                for s in status_handling_dict:
                    status_handling_dict[s][1]()

        return None, 200, None
