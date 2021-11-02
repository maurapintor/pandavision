import logging
import os

import flask
from flask import abort, render_template
from flask_restful import Resource
from rq import Connection
from rq import Queue
from rq.registry import FinishedJobRegistry, StartedJobRegistry, DeferredJobRegistry

from app.adv.evaluation_manager import EvaluationManager, ATTACK_CHOICES
from app.config import config
from app.worker import conn
from forms.sec_eval_form import SecEvalForm

DATA_FOLDER = os.getenv('DATA_DIR', os.path.join(config.PROJECT_ROOT, config.DATA_DIR))

status_handling_dict = {
    "started": (lambda: StartedJobRegistry().get_job_ids(), lambda: 1),
    "finished": (lambda: FinishedJobRegistry().get_job_ids(), lambda: FinishedJobRegistry().cleanup()),
    "queued": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='default').empty()),
    "failed": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='failed').empty()),
    "deferred": (lambda: DeferredJobRegistry().get_job_ids(), lambda: DeferredJobRegistry().cleanup()),
}


def attack(**kwargs):
    em = EvaluationManager(
        dataset_id=os.path.join(DATA_FOLDER, kwargs.get("dataset", None)),
        attack=kwargs.get("attack", None),
        attack_params=kwargs.get("attack-params", None),
        model_id=os.path.join(DATA_FOLDER, kwargs.get("trained-model", None)),
        metric=kwargs.get("performance-metric", "classification-accuracy"),
        perturbation_values=kwargs.get("perturbation-values", None),
        evaluation_mode=kwargs.get("evaluation-mode", "complete"),
        task=kwargs.get("task", "classification"),
        indexes=kwargs.get("indexes", None),
        preprocessing=kwargs.get("preprocessing", None),
    )
    eval = em.sec_eval_curve()
    return eval


class SecurityEvaluations(Resource):

    def get(self):

        form = SecEvalForm()
        # form.attack = ATTACK_CHOICES[form.pert_type]
        return render_template('sec_eval_select.html', form=form)


    def post(self):
        args = flask.request.json
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
        return job.get_id(), 202, {}

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
