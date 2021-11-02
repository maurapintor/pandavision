from __future__ import absolute_import

from .api.home import Home
from .api.security_evaluations import SecurityEvaluations
from .api.security_evaluations_id import SecurityEvaluationsId
from .api.security_evaluations_id_output import SecurityEvaluationsIdOutput
from .api.adversarial_examples import AdversarialExamples
from .api.adversarial_examples_id import AdversarialExamplesId
from .api.adversarial_examples_id_output import AdversarialExamplesIdOutput
from .api.attacks_list import AttackList, Attacks

routes = [
    dict(resource=Home, urls=['/', '/home'], endpoint='home'),
    dict(resource=SecurityEvaluations, urls=['/security_evaluations'], endpoint='security_evaluations'),
    dict(resource=SecurityEvaluationsId, urls=['/security_evaluations/<id>'], endpoint='security_evaluations_id'),
    dict(resource=SecurityEvaluationsIdOutput, urls=['/security_evaluations/<id>/output'], endpoint='security_evaluations_id_output'),
    dict(resource=AdversarialExamples, urls=['/adversarial_examples'], endpoint='adversarial_examples'),
    dict(resource=AdversarialExamplesId, urls=['/adversarial_examples/<id>'], endpoint='adversarial_examples_id'),
    dict(resource=AdversarialExamplesIdOutput, urls=['/adversarial_examples/<id>/output'], endpoint='adversarial_examples_id_output'),
    dict(resource=Attacks, urls=['/attacks'], endpoint='attacks'),
    dict(resource=AttackList, urls=['/attacks/<pert_type>'], endpoint='attacks_pert_type'),
]