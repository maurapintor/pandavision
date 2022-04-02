import logging
import flask
from flask import abort, make_response, render_template
from flask_restful import Resource
from rq import Connection
from rq import Queue

from app.api.api.api_utils import create_evaluation_manager, status_handling_dict
from app.forms.adv_example_form import AdvExampleForm

from app.worker import conn


def create_adv_example(**kwargs):
    em = create_evaluation_manager(**kwargs)
    return em.adv_image()


class AdversarialExamples(Resource):

    def get(self):

        form = AdvExampleForm()
        return make_response(render_template('adv_examples_select.html', form=form))  # TODO

    def post(self):

        with Connection(conn):
            q = Queue(connection=conn, name="adv-gen")
            try:
                job = q.enqueue_call(func=create_adv_example, result_ttl=5000, timeout=5000, kwargs=flask.request.json)
            except:
                logging.log(logging.WARNING, "Unable to queue the requested process.")
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
