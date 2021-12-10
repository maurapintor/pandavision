from wtforms import RadioField
from wtforms.validators import DataRequired

from app.forms.shared_forms import ExperimentForm


class SecEvalForm(ExperimentForm):
    eval_mode = RadioField("eval_mode", choices=[('fast', 'Fast'), ('complete', 'Complete')], default='fast',
                           validators=[DataRequired()])
