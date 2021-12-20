import io
import logging
from base64 import b64encode

import numpy as np
from PIL import Image
from flask import abort
from flask_restful import Resource
from rq.job import Job

from app.worker import conn


class SecurityEvaluationsIdOutputInspectSample(Resource):

    def get(self, id, sample_idx):
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
            sample_idx = int(sample_idx)
            return {
                'attack_losses': loss_curve[sample_idx],
                'attack_distances': distance_curve[sample_idx],
                'epsilon_vals': result.get('sec-curve').get('x-values'),
                'num_samples': len(loss_curve)
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
            num_samples = len(result.get('attack_losses'))
            return {'epsilon_values': epsilon_values[1:],
                    'num_samples': list(range(num_samples))}
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}

class SecurityEvaluationsIdOutputInspectSampleImgType(Resource):

    def get(self, id, sample_idx, img_type):
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
            if img_type == 'diff':
                img = np.array(result.get('adv_examples')) - np.array(result.get('orig_samples'))
                img -= img.min()
                img /= img.max()
            else:
                img = np.array(result.get(img_type))
            file_object = io.BytesIO()
            sample_idx = int(sample_idx)
            image = img[sample_idx, ...].transpose(1, 2, 0)
            img = Image.fromarray((image*255).astype(np.uint8))
            img.save(file_object, 'PNG')
            base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
            return base64img
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}

class SecurityEvaluationsIdOutputInspectSampleOrig(Resource):

    def get(self, id, sample_idx):
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
            file_object = io.BytesIO()
            sample_idx = int(sample_idx)
            image = np.array(result.get('orig_samples'))[sample_idx, ...].transpose(1, 2, 0)
            img = Image.fromarray((image*255).astype(np.uint8))
            img.save(file_object, 'PNG')
            base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
            return base64img
        else:
            # redirect to job status API
            return "Temporary redirect. Job not finished yet.", \
                   307, {'Location': 'security_evaluations/{}'.format(id)}