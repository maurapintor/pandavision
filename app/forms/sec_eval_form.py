from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, FileField, FloatField, FieldList, RadioField
from wtforms.validators import DataRequired


class SecEvalForm(FlaskForm):
    model = FileField('Model', validators=[DataRequired()])
    addpreprocessing = RadioField("Preprocessing", choices=[('None', 'no preprocessing'),
                                                            ('{}', 'default preprocessing'),
                                                            ('custom', 'custom')], validators=[DataRequired()])
    preprocess_mean_R = FloatField()
    preprocess_mean_G = FloatField()
    preprocess_mean_B = FloatField()
    preprocess_std_R = FloatField()
    preprocess_std_G = FloatField()
    preprocess_std_B = FloatField()
    dataset = FileField('Dataset', validators=[DataRequired()])
    pert_type = SelectField('Perturbation type', choices=[('linf', 'L-infinity'),
                                                          ('l2', 'L2')], validators=[DataRequired()])
    attack = SelectField('attack', choices=[])
    submit = SubmitField('Submit')
