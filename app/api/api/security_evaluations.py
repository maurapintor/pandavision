# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import logging

from flask import g, abort

# queue jobs handling
from rq.registry import FinishedJobRegistry, StartedJobRegistry, DeferredJobRegistry
from rq import Queue
from rq import Connection

# config = ConfigurationManager(configuration_id="5bc9ec73b520d91019a84fa5")

# todo manage to stop a started process
# todo change cleanup into some deletion

# status_handling_dict is used for queue handling
#   keys:       status name
#   values:     tuples containing (1) method for collecting the list of jobs
#               for the given status and (2) method for removing all
#               jobs corresponding to the given status from the queue
from api.api import Resource
from worker import conn

import traceback

import os
SHARED_DATA_FOLDER = os.getenv('SHARED_DATA_FOLDER', 'data')

status_handling_dict = {
    "started": (lambda: StartedJobRegistry().get_job_ids(), lambda: 1),
    "finished": (lambda: FinishedJobRegistry().get_job_ids(), lambda: FinishedJobRegistry().cleanup()),
    "queued": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='default').empty()),
    "failed": (lambda: Queue(name='sec-evals').get_job_ids(), lambda: Queue(name='failed').empty()),
    "deferred": (lambda: DeferredJobRegistry().get_job_ids(), lambda: DeferredJobRegistry().cleanup()),
}


def attack(**kwargs):
    from security_evaluations.evaluation_manager import EvaluationManager

    try:
        em = EvaluationManager(
            dataset_id= os.path.join(SHARED_DATA_FOLDER, kwargs.get("dataset", None)),
            perturbation_type=kwargs.get("perturbation-type", None),
            model_id= os.path.join(SHARED_DATA_FOLDER, kwargs.get("trained-model", None)),
            metric=kwargs.get("performance-metric", "classification-accuracy"),
            perturbation_values=kwargs.get("perturbation-values", None),
            evaluation_mode=kwargs.get("evaluation-mode", "complete"),
            task=kwargs.get("task", None),
            indexes=kwargs.get("indexes", None),
            config_file=kwargs.get("config-path", None),
            preprocessing_pipeline=kwargs.get("pipeline-path", None),
        )

        eval = em.sec_eval_curve()
        sec_curve = eval['sec-curve']
        print(sec_curve)
    except Exception as _:
        eval = None
        logging.log(logging.WARNING, "Unable to perform security evaluation.")
        traceback.print_exc()

    try:
        # callback to the DSE Engine
        callback_id = kwargs.get("callback_id", "")

        import requests
        url = 'http://%s/api/dse_engine/callback?callback_id=%s' % ('dse_engine:5000', callback_id)
        data = {"some_data": 0}

        _ = requests.post(url, data=data)
    except:
        logging.log(logging.WARNING, "Unable to send data to DSE server. SKIPPING post.")

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
        # if "dataset" in g.json and "trained-model" in g.json:
        #     try:
        #         model = AlgorithmConfiguration.objects.get(id=g.json["trained-model"])
        #     except:
        #         logging.log(logging.INFO, msg="Trained-model not found.")
        #         abort(404, "Model not found")
        #         return
        #     try:
        #         dataset = Dataset.objects.get(id=g.json["dataset"])
        #     except:
        #         logging.log(logging.INFO, msg="Dataset not found.")
        #         abort(404, "Dataset not found")
        #         return
        # else:
        #     logging.log(logging.INFO, msg="Missing parameters.")
        #     abort(400, "Bad request. Missing parameters.")
        #     return

        callback_id = g.args.get("callback_id")

        args = {**g.json, **{"callback_id": callback_id}}

        with Connection(conn):
            q = Queue(connection=conn, name="sec-evals")
            try:
                job = q.enqueue_call(func=attack, result_ttl=5000, timeout=5000, kwargs=args)
                # job = q.enqueue_call(func=attack, result_ttl=config.result_ttl, timeout=config.job_timeout,
                # kwargs=g.json)
            except:
                logging.log(logging.WARNING, "Unable to queue the requested process.")
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
