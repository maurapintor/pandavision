import logging

import pandas as pd
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
            x, y = result['sec-curve']['x-values'], result['sec-curve']['y-values']
            result_df = pd.DataFrame(data={'pert_value': x, 'accuracy': y})
            return Response(result_df.to_csv(), mimetype='text/csv',
                            headers={"Content-disposition":
                                         "attachment; filename=sec_eval_results.csv"})
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}
