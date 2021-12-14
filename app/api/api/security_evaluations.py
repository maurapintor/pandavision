import logging

import flask
from flask import abort, render_template, make_response
from flask_restful import Resource
from rq import Connection, Queue

from app.api.api.api_utils import get_args, create_evaluation_manager, status_handling_dict
from app.config import config
from app.forms.sec_eval_form import SecEvalForm
from app.worker import conn


def attack(**kwargs):
    em = create_evaluation_manager(**kwargs)
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
            args = get_args(form)
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
