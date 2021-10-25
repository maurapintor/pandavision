# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import logging

from flask import abort
from rq.job import Job

from app.api.api import Resource
from app.worker import conn


class SecurityEvaluationsIdOutput(Resource):

    def get(self, id):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "GET /api/security_evaluations/{}/output failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        # TODO implement 410 GONE status code, should be returned if the job finished
        # todo but the result ttl has expired
        if job.is_finished:
            result = job.result
            return result, 200, {}
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'api/security_evaluations/{}'.format(id)}

    def delete(self, id):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "DELETE /api/security_evaluations/{}/output failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        if not job.is_finished:
            return "Temporary redirect. Job not finished yet.", 307, \
                   {'Location': '/api/security_evaluation/{}'.format(id)}
        try:
            job.cleanup(ttl=0)
        except:
            logging.log(logging.INFO,
                        "DELETE /api/security_evaluations/{}/output failed. Unable to process "
                        "the job cleanup.".format(id))
            abort(422, "Unprocessable entry. The server understands the request "
                       "entity, but was unable to process the instructions.")
            return

        return None, 200, None
