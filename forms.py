from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField,ValidationError,SelectField,IntegerField,TextAreaField

class PredictForm(FlaskForm):
    first=IntegerField('Enter the Value Please')
    second=IntegerField('Enter the Value Please')
    third=IntegerField('Enter the Value Please')
    fourth=IntegerField('Enter the Value Please')
    fifth=IntegerField('Enter the Value Please')
    sixth=IntegerField('Enter the Value Please')
    seventh=IntegerField('Enter the Value Please')
    eight=IntegerField('Enter the Value Please')
    nine=IntegerField('Enter the Value Please')
    ten=IntegerField('Enter the Value Please')
    eleven=IntegerField('Enter the Value Please')
    twelve=IntegerField('Enter the Value Please')
    thirteen=IntegerField('Enter the Value Please')
    fourteen=IntegerField('Enter the Value Please')
    fifteen=IntegerField('Enter the Value Please')
    sixteen=IntegerField('Enter the Value Please')
    seventeen=IntegerField('Enter the Value Please')
    eighteen=IntegerField('Enter the Value Please')
    nineteen=IntegerField('Enter the Value Please')
    twenty=IntegerField('Enter the Value Please')
    submit = SubmitField('Predict!')

