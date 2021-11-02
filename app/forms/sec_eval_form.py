from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired



class SecEvalForm(FlaskForm):
    model = SelectField('Model', choices=['a', 'b'], validators=[DataRequired()])
    image = SelectField('Data', choices=['c', 'd'], validators=[DataRequired()])
    pert_type = SelectField('Perturbation type', choices=[('L-infinity', 'linf'),
                                                          ('L2', 'l2')], validators=[DataRequired()])
    attack = SelectField('attack', choices=[])
    submit = SubmitField('Submit')