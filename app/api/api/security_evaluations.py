# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import logging

from flask import g, abort

from rq.registry import FinishedJobRegistry, StartedJobRegistry, DeferredJobRegistry
from rq import Queue
from rq import Connection

# todo manage to stop a started process
# todo change cleanup into some deletion


import os

from app.adv.evaluation_manager import EvaluationManager
from app.api.api import Resource
from app.worker import conn

SHARED_DATA_FOLDER = os.getenv('SHARED_DATA_FOLDER', 'appdata/')


status_handling_dict = {
    "started": (lambda: StartedJobRegistry().get_job_ids(), lambda: 1),
    "finished": (lambda: FinishedJobRegistry().get_job_ids(), lambda: FinishedJobRegistry().cleanup()),
    "queued": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='default').empty()),
    "failed": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='failed').empty()),
    "deferred": (lambda: DeferredJobRegistry().get_job_ids(), lambda: DeferredJobRegistry().cleanup()),
}


def attack(**kwargs):
    em = EvaluationManager(
        dataset_id=os.path.join(SHARED_DATA_FOLDER, kwargs.get("dataset", None)),
        perturbation_type=kwargs.get("perturbation-type", None),
        model_id=os.path.join(SHARED_DATA_FOLDER, kwargs.get("trained-model", None)),
        metric=kwargs.get("performance-metric", "classification-accuracy"),
        perturbation_values=kwargs.get("perturbation-values", None),
        evaluation_mode=kwargs.get("evaluation-mode", "complete"),
        task=kwargs.get("task", None),
        indexes=kwargs.get("indexes", None),
        preprocessing=kwargs.get("preprocessing", None),
    )
    eval = em.sec_eval_curve()
    return eval


class SecurityEvaluations(Resource):

    def get(self):
        with Connection(conn):
            s = g.args.get("status", None)
            default_queue = Queue(name='sec-evals')
            if s in status_handling_dict:
                jobs = [default_queue.fetch_job(job_id) for job_id in status_handling_dict[s][0]()]
            elif s is None:
                jobs = [default_queue.fetch_job(job_id) for status in status_handling_dict for job_id in
                        status_handling_dict[status][0]()]
            else:
                logging.log(level=logging.WARNING, msg="Invalid input.")
                abort(400, "Filtering parameter not understood: status={}. "
                           "Possible statuses are: {}.".format(s, ", ".join(status_handling_dict.keys())))
                return
            job_list = [{"id": j.id, "status": j.status} for j in jobs]
        return job_list, 200, None

    def post(self):
        args = {**g.json}
        with Connection(conn):
            q = Queue(connection=conn, name="sec-evals")
            try:
                job = q.enqueue_call(func=attack, result_ttl=5000, timeout=5000, kwargs=args)
            except Exception as e:
                logging.log(logging.WARNING, "Unable to queue the requested process. Redis service unavailable.")
                abort(422, "Unprocessable entry. The server understands the request "
                           "entity, but was unable to process the instructions.")
                return
        return job.get_id(), 202, {}

    def delete(self):
        s = g.args.get("status", None)
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
