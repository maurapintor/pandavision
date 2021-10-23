# -*- coding: utf-8 -*-

import six

base_path = '/api'


DefinitionsSecurityevaluationresults = {'type': 'object', 'properties': {'sec-level': {'type': 'string', 'description': 'Resulting security level of the security evaluation.', 'example': 'high'}, 'sec-value': {'type': 'number', 'example': '0.2', 'description': 'Resulting score of the security evaluation. The range depends on the chosen metric.'}, 'sec-curve': {'type': 'object', 'description': 'Complete security evaluation curve resulting from the security evaluation.', 'properties': {'x-values': {'type': 'array', 'items': {'type': 'number'}, 'example': '[0, 1, 2, 3]'}, 'y-values': {'type': 'array', 'items': {'type': 'number'}, 'example': '[10, 15, 18, 20]'}}}}}
DefinitionsAdversarialexampleparameters = {'type': 'object', 'required': ['dataset', 'trained-model'], 'properties': {'dataset': {'type': 'string', 'description': 'Path of the dataset source of the sample image. It should be a file in format: hdf5', 'example': '/data/cifar.onnx'}, 'trained-model': {'type': 'string', 'description': 'Path of the pre-trained model to evaluate. It should be a file in format: onnx', 'example': 'data/c78as6cdasjh3ds.hdf5'}, 'performance-metric': {'type': 'string', 'description': 'Name of the performance metric to use. (available: scores)', 'example': 'scores'}, 'perturbation-type': {'type': 'string', 'description': 'Type of the perturbation to use to generate adversarial examples.', 'example': 'max-norm'}, 'perturbation-values': {'type': 'array', 'description': 'List of values to use for generating the perturbations.', 'items': {'type': 'number'}, 'example': '[0, 0.05, 0.1]'}}}
DefinitionsEvaluationparameters = {'type': 'object', 'required': ['dataset', 'trained-model'], 'properties': {'dataset': {'type': 'string', 'description': 'Path of the dataset to use for performance evaluation. It should be a file in format: hdf5.', 'example': '/data/cifar10.hdf5'}, 'trained-model': {'type': 'string', 'description': 'Path of the pre-trained model to evaluate. It should be a file in format: onnx.', 'example': '/data/resnet18/resnet18_pi123.onnx'}, 'performance-metric': {'type': 'string', 'description': 'Name of the performance metric to use. One of: classification-accuracy.', 'example': 'classification-accuracy'}, 'perturbation-type': {'type': 'string', 'description': 'Type of the perturbation to use to generate adversarial examples. One of: max-norm, random.', 'example': 'max-norm'}, 'perturbation-values': {'type': 'array', 'description': 'List of values to use for generating the perturbations.', 'items': {'type': 'number'}, 'example': '[0, 1.0, 2.0, 5.0, 10.0]'}, 'evaluation-mode': {'type': 'string', 'description': 'Whether to run the experiment in fast or complete mode. Complete mode takes longer to execute.', 'example': 'fast'}, 'task': {'type': 'string', 'description': 'Final task achieved by the model. This is used for choosing the attack algorithm. One of: classification, detection.', 'example': 'classification'}, 'indexes': {'type': 'array', 'description': 'List of indexes to use for sampling the dataset. If no value is passed, a random sample will be used, with number of elements chosen according to the selected mode.', 'items': {'type': 'number'}, 'example': '[0, 0.05, 0.1]'}, 'config-path': {'type': 'string', 'description': 'Path of the configuration file needed for correctly preparing the model. Currently required only for YOLO anchors. If no value is passed, the configuration will not be considered.', 'example': '/tmp/config.json'}, 'pipeline-path': {'type': 'string', 'description': 'File containing the specifications for the preprocessing pipeline. If no value is passed, the preprocessing will be ignored (a standard ToTensor will be applied).'}}}

validators = {
    ('security_evaluations', 'POST'): {'json': {'type': 'object', 'required': ['dataset', 'trained-model'], 'properties': {'dataset': {'type': 'string', 'description': 'Path of the dataset to use for performance evaluation. It should be a file in format: hdf5.', 'example': '/data/cifar10.hdf5'}, 'trained-model': {'type': 'string', 'description': 'Path of the pre-trained model to evaluate. It should be a file in format: onnx.', 'example': '/data/resnet18/resnet18_pi123.onnx'}, 'performance-metric': {'type': 'string', 'description': 'Name of the performance metric to use. One of: classification-accuracy.', 'example': 'classification-accuracy'}, 'perturbation-type': {'type': 'string', 'description': 'Type of the perturbation to use to generate adversarial examples. One of: max-norm, random.', 'example': 'max-norm'}, 'perturbation-values': {'type': 'array', 'description': 'List of values to use for generating the perturbations.', 'items': {'type': 'number'}, 'example': '[0, 1.0, 2.0, 5.0, 10.0]'}, 'evaluation-mode': {'type': 'string', 'description': 'Whether to run the experiment in fast or complete mode. Complete mode takes longer to execute.', 'example': 'fast'}, 'task': {'type': 'string', 'description': 'Final task achieved by the model. This is used for choosing the attack algorithm. One of: classification, detection.', 'example': 'classification'}, 'indexes': {'type': 'array', 'description': 'List of indexes to use for sampling the dataset. If no value is passed, a random sample will be used, with number of elements chosen according to the selected mode.', 'items': {'type': 'number'}, 'example': '[0, 0.05, 0.1]'}, 'config-path': {'type': 'string', 'description': 'Path of the configuration file needed for correctly preparing the model. Currently required only for YOLO anchors. If no value is passed, the configuration will not be considered.', 'example': '/tmp/config.json'}, 'pipeline-path': {'type': 'string', 'description': 'File containing the specifications for the preprocessing pipeline. If no value is passed, the preprocessing will be ignored (a standard ToTensor will be applied).'}}}, },
    ('security_evaluations', 'GET'): {'args': {'required': [], 'properties': {'status': {'description': 'Filter all evaluation jobs by status.', 'type': 'string'}}}},
    ('security_evaluations', 'DELETE'): {'args': {'required': [], 'properties': {'status': {'description': 'Delete all jobs with the given status', 'type': 'string'}}}},
    ('adversarial_examples', 'POST'): {'json': {'type': 'object', 'required': ['dataset', 'trained-model'], 'properties': {'dataset': {'type': 'string', 'description': 'Path of the dataset source of the sample image. It should be a file in format: hdf5', 'example': '/data/cifar.onnx'}, 'trained-model': {'type': 'string', 'description': 'Path of the pre-trained model to evaluate. It should be a file in format: onnx', 'example': 'data/c78as6cdasjh3ds.hdf5'}, 'performance-metric': {'type': 'string', 'description': 'Name of the performance metric to use. (available: scores)', 'example': 'scores'}, 'perturbation-type': {'type': 'string', 'description': 'Type of the perturbation to use to generate adversarial examples.', 'example': 'max-norm'}, 'perturbation-values': {'type': 'array', 'description': 'List of values to use for generating the perturbations.', 'items': {'type': 'number'}, 'example': '[0, 0.05, 0.1]'}}}},
    ('adversarial_examples', 'DELETE'): {'args': {'required': [], 'properties': {'status': {'description': 'Delete all jobs with the given status', 'type': 'string'}}}},
}

filters = {
    ('security_evaluations', 'POST'): {202: {'headers': {'Location': {'description': 'Location of the newly created job resource.', 'type': 'string'}}, 'schema': {'type': 'string', 'description': 'ID of the security evaluation job.', 'example': 'job123'}}, 400: {'headers': None, 'schema': None}, 404: {'headers': None, 'schema': None}, 422: {'headers': None, 'schema': None}},
    ('security_evaluations', 'GET'): {200: {'headers': None, 'schema': {'type': 'array', 'description': 'List of queued, running and finished jobs.', 'items': {'type': 'object', 'description': 'Job element.', 'properties': {'id': {'type': 'string', 'example': 'job123', 'description': 'ID of the job.'}, 'job-status': {'type': 'string', 'example': 'running', 'description': 'Status of the job.'}}}}}, 400: {'headers': None, 'schema': None}},
    ('security_evaluations', 'DELETE'): {200: {'headers': None, 'schema': None}, 400: {'headers': None, 'schema': None}},
    ('security_evaluations_id', 'GET'): {200: {'headers': None, 'schema': {'type': 'object', 'properties': {'job-status': {'type': 'string', 'example': 'running'}}, 'description': 'Status of the job.', 'example': 'running'}}, 303: {'headers': {'Location': {'description': 'Location of the results resource produced by the job.', 'type': 'string'}}, 'schema': None}, 404: {'headers': None, 'schema': None}},
    ('security_evaluations_id', 'DELETE'): {200: {'headers': None, 'schema': None}, 404: {'headers': None, 'schema': None}},
    ('security_evaluations_id_output', 'GET'): {200: {'headers': None, 'schema': {'type': 'object', 'properties': {'sec-level': {'type': 'string', 'description': 'Resulting security level of the security evaluation.', 'example': 'high'}, 'sec-value': {'type': 'number', 'example': '0.2', 'description': 'Resulting score of the security evaluation. The range depends on the chosen metric.'}, 'sec-curve': {'type': 'object', 'description': 'Complete security evaluation curve resulting from the security evaluation.', 'properties': {'x-values': {'type': 'array', 'items': {'type': 'number'}, 'example': '[0, 1, 2, 3]'}, 'y-values': {'type': 'array', 'items': {'type': 'number'}, 'example': '[10, 15, 18, 20]'}}}}}}, 307: {'headers': {'Location': {'description': 'Location of the job resource that is assigned to process the results.', 'type': 'string'}}, 'schema': None}, 404: {'headers': None, 'schema': None}, 410: {'headers': None, 'schema': None}},
    ('security_evaluations_id_output', 'DELETE'): {200: {'headers': None, 'schema': None}, 307: {'headers': None, 'schema': None}, 404: {'headers': None, 'schema': None}},
    ('adversarial_examples', 'POST'): {202: {'headers': None, 'schema': {'type': 'string', 'description': 'ID of the security evaluation job.', 'example': 'job123'}}, 404: {'headers': None, 'schema': None}, 405: {'headers': None, 'schema': None}, 422: {'headers': None, 'schema': None}},
    ('adversarial_examples', 'GET'): {200: {'headers': None, 'schema': {'type': 'array', 'description': 'List of queued, running and finished jobs.', 'items': {'type': 'object', 'description': 'Job element.', 'properties': {'id': {'type': 'string', 'example': 'job123', 'description': 'ID of the job.'}, 'job-status': {'type': 'string', 'example': 'running', 'description': 'Status of the job.'}}}}}, 400: {'headers': None, 'schema': None}},
    ('adversarial_examples', 'DELETE'): {200: {'headers': None, 'schema': None}, 400: {'headers': None, 'schema': None}},
    ('adversarial_examples_id', 'GET'): {200: {'headers': None, 'schema': {'type': 'object', 'properties': {'job-status': {'type': 'string', 'example': 'running'}}, 'description': 'Status of the job.', 'example': 'running'}}, 303: {'headers': {'Location': {'description': 'Location of the results resource produced by the job.', 'type': 'string'}}, 'schema': None}, 404: {'headers': None, 'schema': None}},
    ('adversarial_examples_id', 'DELETE'): {200: {'headers': None, 'schema': None}, 404: {'headers': None, 'schema': None}},
    ('adversarial_examples_id_output', 'GET'): {200: {'headers': None, 'schema': {'type': 'file', 'description': 'PNG image of adversarial example.'}}, 307: {'headers': {'Location': {'description': 'Location of the job resource that is assigned to process the results.', 'type': 'string'}}, 'schema': None}, 404: {'headers': None, 'schema': None}, 410: {'headers': None, 'schema': None}},
    ('adversarial_examples_id_output', 'DELETE'): {200: {'headers': None, 'schema': None}, 307: {'headers': None, 'schema': None}, 404: {'headers': None, 'schema': None}},
}

scopes = {
}


class Security(object):

    def __init__(self):
        super(Security, self).__init__()
        self._loader = lambda: []

    @property
    def scopes(self):
        return self._loader()

    def scopes_loader(self, func):
        self._loader = func
        return func

security = Security()


def merge_default(schema, value, get_first=True):
    # TODO: more types support
    type_defaults = {
        'integer': 9573,
        'string': 'something',
        'object': {},
        'array': [],
        'boolean': False
    }

    results = normalize(schema, value, type_defaults)
    if get_first:
        return results[0]
    return results


def normalize(schema, data, required_defaults=None):
    if required_defaults is None:
        required_defaults = {}
    errors = []

    class DataWrapper(object):

        def __init__(self, data):
            super(DataWrapper, self).__init__()
            self.data = data

        def get(self, key, default=None):
            if isinstance(self.data, dict):
                return self.data.get(key, default)
            return getattr(self.data, key, default)

        def has(self, key):
            if isinstance(self.data, dict):
                return key in self.data
            return hasattr(self.data, key)

        def keys(self):
            if isinstance(self.data, dict):
                return list(self.data.keys())
            return list(getattr(self.data, '__dict__', {}).keys())

        def get_check(self, key, default=None):
            if isinstance(self.data, dict):
                value = self.data.get(key, default)
                has_key = key in self.data
            else:
                try:
                    value = getattr(self.data, key)
                except AttributeError:
                    value = default
                    has_key = False
                else:
                    has_key = True
            return value, has_key

    def _merge_dict(src, dst):
        for k, v in six.iteritems(dst):
            if isinstance(src, dict):
                if isinstance(v, dict):
                    r = _merge_dict(src.get(k, {}), v)
                    src[k] = r
                else:
                    src[k] = v
            else:
                src = {k: v}
        return src

    def _normalize_dict(schema, data):
        result = {}
        if not isinstance(data, DataWrapper):
            data = DataWrapper(data)

        for _schema in schema.get('allOf', []):
            rs_component = _normalize(_schema, data)
            _merge_dict(result, rs_component)

        for key, _schema in six.iteritems(schema.get('properties', {})):
            # set default
            type_ = _schema.get('type', 'object')

            # get value
            value, has_key = data.get_check(key)
            if has_key:
                result[key] = _normalize(_schema, value)
            elif 'default' in _schema:
                result[key] = _schema['default']
            elif key in schema.get('required', []):
                if type_ in required_defaults:
                    result[key] = required_defaults[type_]
                else:
                    errors.append(dict(name='property_missing',
                                       message='`%s` is required' % key))

        additional_properties_schema = schema.get('additionalProperties', False)
        if additional_properties_schema is not False:
            aproperties_set = set(data.keys()) - set(result.keys())
            for pro in aproperties_set:
                result[pro] = _normalize(additional_properties_schema, data.get(pro))

        return result

    def _normalize_list(schema, data):
        result = []
        if hasattr(data, '__iter__') and not isinstance(data, dict):
            for item in data:
                result.append(_normalize(schema.get('items'), item))
        elif 'default' in schema:
            result = schema['default']
        return result

    def _normalize_default(schema, data):
        if data is None:
            return schema.get('default')
        else:
            return data

    def _normalize(schema, data):
        if schema is True or schema == {}:
            return data
        if not schema:
            return None
        funcs = {
            'object': _normalize_dict,
            'array': _normalize_list,
            'default': _normalize_default,
        }
        type_ = schema.get('type', 'object')
        if type_ not in funcs:
            type_ = 'default'

        return funcs[type_](schema, data)

    return _normalize(schema, data), errors
