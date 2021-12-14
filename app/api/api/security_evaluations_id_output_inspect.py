import logging

from flask import abort
from flask_restful import Resource
from rq.job import Job

from app.worker import conn


class SecurityEvaluationsIdOutputInspectEpsSample(Resource):

    def get(self, id, eps_idx, sample_idx):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "GET /security_evaluations/{}/output/inspect failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        # TODO implement 410 GONE status code, should be returned if the job finished
        # todo but the result ttl has expired
        if job.is_finished:
            result = job.result
            loss_curve = result.get('attack_losses')
            distance_curve = result.get('attack_distances')
            eps_idx, sample_idx = int(eps_idx), int(sample_idx)
            return {
                'attack_losses': loss_curve[eps_idx][sample_idx],
                'attack_distances': distance_curve[eps_idx][sample_idx],
                'epsilon_vals': result.get('sec-curve').get('x-values'),
                'num_samples': len(loss_curve[eps_idx])
            }
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}


class SecurityEvaluationsIdOutputInspectEps(Resource):

    def get(self, id, eps_idx):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "GET /security_evaluations/{}/output/inspect failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        # TODO implement 410 GONE status code, should be returned if the job finished
        # todo but the result ttl has expired
        if job.is_finished:
            result = job.result
            loss_curve = result.get('attack_losses')
            distance_curve = result.get('attack_distances')
            eps_idx = int(eps_idx)
            return {
                'attack_losses': loss_curve[eps_idx],
                'attack_distances': distance_curve[eps_idx],
                'epsilon_vals': result.get('sec-curve').get('x-values'),
                'num_samples': len(loss_curve[eps_idx])
            }
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}


class SecurityEvaluationsIdOutputInspect(Resource):

    def get(self, id):
        try:
            job = Job.fetch(id, connection=conn)
        except:
            logging.log(logging.INFO,
                        "GET /security_evaluations/{}/output/inspect failed. Job ID not found.".format(id))
            abort(404, "Job ID not found.")
            return

        # TODO implement 410 GONE status code, should be returned if the job finished
        # todo but the result ttl has expired
        if job.is_finished:
            result = job.result
            epsilon_values = result.get('sec-curve').get('x-values')
            num_samples = len(result.get('attack_losses')[0])
            return {'epsilon_values': epsilon_values[1:],
                    'num_samples': list(range(num_samples))}
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}
