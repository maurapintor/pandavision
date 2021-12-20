from .api.attacks_list import AttackList, Attacks, PertSizes
from .api.home import Home
from .api.security_evaluations import SecurityEvaluations
from .api.security_evaluations_id import SecurityEvaluationsId
from .api.security_evaluations_id_output import SecurityEvaluationsIdOutput
from .api.security_evaluations_id_output_csv import SecurityEvaluationsIdOutputCsv
from .api.security_evaluations_id_output_inspect import SecurityEvaluationsIdOutputInspect, \
    SecurityEvaluationsIdOutputInspectSample
from .api.upload_file import UploadFiles

routes = [
    dict(resource=Home, urls=['/', '/home'], endpoint='home'),
    dict(resource=UploadFiles, urls=['/upload/<data_type>'], endpoint='upload_data_type'),
    dict(resource=SecurityEvaluations, urls=['/security_evaluations'], endpoint='security_evaluations'),
    dict(resource=SecurityEvaluationsId, urls=['/security_evaluations/<id>'], endpoint='security_evaluations_id'),
    dict(resource=SecurityEvaluationsIdOutput, urls=['/security_evaluations/<id>/output'],
         endpoint='security_evaluations_id_output'),
    dict(resource=SecurityEvaluationsIdOutputCsv, urls=['/security_evaluations/<id>/output/csv'],
         endpoint='security_evaluations_id_output_csv'),
    dict(resource=SecurityEvaluationsIdOutputInspectSample,
         urls=['/security_evaluations/<id>/inspect/<sample_idx>'],
         endpoint='security_evaluations_id_output_inspect_eps_sample'),
    dict(resource=SecurityEvaluationsIdOutputInspect, urls=['/security_evaluations/<id>/inspect/'],
         endpoint='security_evaluations_id_output_inspect'),
    dict(resource=Attacks, urls=['/attacks'], endpoint='attacks'),
    dict(resource=AttackList, urls=['/attacks/<pert_type>'], endpoint='attacks_pert_type'),
    dict(resource=PertSizes, urls=['/pert_sizes/<pert_type>'], endpoint='pert_sizes_pert_type'),
]
