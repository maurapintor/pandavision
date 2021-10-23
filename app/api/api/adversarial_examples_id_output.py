# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import logging

from flask import make_response, abort
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rq.job import Job

from app.api.api import Resource
from worker import conn


class AdversarialExamplesIdOutput(Resource):

    def get(self, id):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "GET /api/adversarial_samples/{}/output failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        # TODO implement 410 GONE status code, should be returned if the job finished
        # todo but the result ttl has expired
        if job.is_finished:
            adv_img = job.result
            output = io.BytesIO()
            FigureCanvas(adv_img).print_png(output)
            response = make_response(output.getvalue())
            response.mimetype = 'image/png'
            return response

        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'api/adversarial_samples/{}'.format(id)}

    def delete(self, id):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "DELETE /api/adversarial_samples/{}/output failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        if not job.is_finished:
            return "Temporary redirect. Job not finished yet.", 307, \
                   {'Location': '/api/adversarial_samples/{}'.format(id)}
        try:
            job.cleanup(ttl=0)
        except:
            logging.log(logging.INFO,
                        "DELETE /api/adversarial_samples/{}/output failed. Unable to process "
                        "the job cleanup.".format(id))
            abort(422, "Unprocessable entry. The server understands the request "
                       "entity, but was unable to process the instructions.")
            return

        return None, 200, None
