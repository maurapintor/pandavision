from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, FileField, FloatField, FieldList, RadioField, IntegerField, BooleanField, \
    FormField
from wtforms.validators import DataRequired


class CWForm(FlaskForm):
    steps = IntegerField("steps")
    binary_search_steps = IntegerField("binary_search_steps")
    confidence = FloatField("kappa")
    initial_const = FloatField("c")
    abort_early = BooleanField("early_stopping", default=False)

class PGDForm(FlaskForm):
    steps = IntegerField("steps")
    rel_stepsize = FloatField("rel_stepsize")
    abs_stepsize = FloatField("abs_stepsize")
    random_start = BooleanField("random_start", default=False)


class SecEvalForm(FlaskForm):
    model = FileField('model', validators=[DataRequired()])
    addpreprocessing = RadioField("preprocessing", choices=[('none', 'no preprocessing'),
                                                            ('default', 'default preprocessing'),
                                                            ('custom', 'custom')], default='{}', validators=[DataRequired()])
    eval_mode = RadioField("eval_mode", choices=[('fast', 'Fast'), ('complete', 'Complete')], default='fast',
                           validators=[DataRequired()])
    preprocess_mean_R = FloatField()
    preprocess_mean_G = FloatField()
    preprocess_mean_B = FloatField()
    preprocess_std_R = FloatField()
    preprocess_std_G = FloatField()
    preprocess_std_B = FloatField()
    dataset = FileField('dataset', validators=[DataRequired()])
    pert_type = SelectField('perturbation_type', choices=[('linf', 'L-infinity'),
                                                          ('l2', 'L2')], validators=[DataRequired()])
    attack = SelectField('attack', choices=[])

    cw_form = FormField(CWForm)
    pgd_form = FormField(PGDForm)
    submit = SubmitField('submit')
