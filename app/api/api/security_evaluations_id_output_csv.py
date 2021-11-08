import csv
import logging
from io import StringIO

from flask import abort, Response
from flask_restful import Resource
from rq.job import Job

from app.worker import conn


class SecurityEvaluationsIdOutputCsv(Resource):

    def get(self, id):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "GET /security_evaluations/{}/output failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        # TODO implement 410 GONE status code, should be returned if the job finished
        # todo but the result ttl has expired
        if job.is_finished:
            result = job.result
            f = StringIO()
            w = csv.writer(f)
            x, y = result['sec-curve']['x-values'], result['sec-curve']['y-values']
            for xi, yi in zip(x, y):
                w.writerow([xi, yi])
            return Response(f.getvalue(), mimetype='text/csv',
                            headers={"Content-disposition":
                                         "attachment; filename=sec_eval_results.csv"})
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}
