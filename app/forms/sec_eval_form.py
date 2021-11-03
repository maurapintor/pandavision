from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, FileField, FloatField, FieldList, RadioField, IntegerField, BooleanField, \
    FormField
from wtforms.validators import DataRequired


class CWForm(FlaskForm):
    steps = IntegerField("steps", default=None)
    binary_search_steps = IntegerField("binary_search_steps", default=None)
    confidence = FloatField("kappa", default=None)
    initial_const = FloatField("c", default=None)
    abort_early = BooleanField("early_stopping", default=None)

class PGDForm(FlaskForm):
    steps = IntegerField("steps", default=None)
    rel_stepsize = FloatField("rel_stepsize", default=None)
    abs_stepsize = FloatField("abs_stepsize", default=None)
    random_start = BooleanField("random_start", default=None)


class SecEvalForm(FlaskForm):
    model = FileField('model', validators=[DataRequired()])
    addpreprocessing = RadioField("preprocessing", choices=[('None', 'no preprocessing'),
                                                            ('{}', 'default preprocessing'),
                                                            ('custom', 'custom')], default='{}', validators=[DataRequired()])
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
