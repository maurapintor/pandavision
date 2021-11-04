import os

from flask import request, redirect, abort
from flask_restful import Resource

from app.config import config

DATA_FOLDER = os.path.join(config.PROJECT_ROOT, config.DATA_DIR)
DATA_TYPES = {'model': 'onnx',
              'data': 'h5'}


def allowed_file(filename, data_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == DATA_TYPES[data_type]


class UploadFiles(Resource):

    def post(self, data_type):
        if not os.path.isdir(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename, data_type):
            file.save(os.path.join(DATA_FOLDER, data_type + '.' + DATA_TYPES[data_type]))
            return 200
        else:
            abort(400)

