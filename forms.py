from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField,ValidationError,SelectField,IntegerField,TextAreaField

class PredictForm(FlaskForm):
    age=StringField(' Enter the age of the patient')

    time_in_hospital=IntegerField('Enter the time in hospital')

    num_medications=IntegerField('Enter the number of medications')

    number_diagnoses=IntegerField('Enter the number of diagnosis done')

    metformin=StringField('Enter the Metmorphin Level')

    chlorpropamide=StringField('Enter the chlorpropamide level')

    glimepiride=StringField('Enter the glimepiride level')

    tolazamide=StringField('Enter the tolazamide level')

    insulin=StringField('Enter the insulin level')

    race=StringField('Enter the race of the patient')

    admission_type_id=IntegerField('Enter the admission type ID of the patient')

    admission_source_id=IntegerField('Enter the admission source ID')

    max_glu_serum=StringField('Enter the max_glu_serum level')

    A1Cresult=StringField('Enter the A1Cresult level')

    submit=SubmitField('Predict!')