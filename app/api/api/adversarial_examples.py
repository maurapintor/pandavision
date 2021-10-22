# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import logging

from flask import g, abort
from rq import Connection
from rq import Queue
from rq.registry import StartedJobRegistry, FinishedJobRegistry, DeferredJobRegistry

from api.api import Resource
from worker import conn

status_handling_dict = {
    "started": (lambda: StartedJobRegistry().get_job_ids(), lambda: 1),
    "finished": (lambda: FinishedJobRegistry().get_job_ids(), lambda: FinishedJobRegistry().cleanup()),
    "queued": (lambda: Queue(name='adv-gen').get_job_ids(), lambda: Queue(name='default').empty()),
    "failed": (lambda: Queue(name='adv-gen').get_job_ids(), lambda: Queue(name='failed').empty()),
    "deferred": (lambda: DeferredJobRegistry().get_job_ids(), lambda: DeferredJobRegistry().cleanup()),
}


def create_adv_sample(**kwargs):
    from security_evaluations.evaluation_manager import EvaluationManager

    em = EvaluationManager(
        dataset_id=kwargs.get("dataset", None),
        perturbation_type=kwargs.get("perturbation-type", None),
        model_id=kwargs.get("trained-model", None),
        metric=kwargs.get("performance-metric", "scores"),
        perturbation_values=kwargs.get("perturbation-values", None),
        config_file=kwargs.get("config-path", None),
        preprocessing_pipeline=kwargs.get("pipeline-path", None),
    )
    return em.adv_image()


class AdversarialExamples(Resource):

    def get(self):

        with Connection(conn):
            s = g.args.get("status", None)
            default_queue = Queue(name='adv-gen')
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

        with Connection(conn):
            q = Queue(connection=conn, name="adv-gen")
            try:
                job = q.enqueue_call(func=create_adv_sample, result_ttl=5000, timeout=5000, kwargs=g.json)
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
