from flask import Flask, request, jsonify, render_template
from forms import PredictForm
import keras
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisjustatest'
@app.route('/', methods=['GET','POST'])
def indexfunc():
    form=PredictForm()
    model = keras.models.load_model("price_prediction_model.h5")
    transformer = joblib.load("data_transformer.joblib")
    prediction_text=' Not Validated Yet..'
    if form.validate_on_submit():
        newdict={
                'age': [str(form.age.data)],
                'time_in_hospital': [int(form.time_in_hospital.data)],
                'num_medications': [int(form.num_medications.data)],
                'number_diagnoses': [int(form.number_diagnoses.data)],
                'metformin':[str(form.metformin.data)],
                'chlorpropamide':[str(form.chlorpropamide.data)],
                'glimepiride':[str(form.glimepiride.data)],
                'tolazamide':[str(form.tolazamide.data)],
                'insulin':[str(form.insulin.data)],
                'race':[str(form.race.data)],
                'admission_type_id':[int(form.admission_type_id.data)],
                'admission_source_id':[int(form.admission_source_id.data)],
                'max_glu_serum':[str(form.max_glu_serum.data)],
                'A1Cresult':[str(form.A1Cresult.data)]
            }
        newds=pd.DataFrame(newdict)

        prediction = model.predict(transformer.transform(newds))

        max_index_col = np.argmax(prediction, axis=1)

        if max_index_col==0:
            prediction_text=' The Patient will be readmitted > 30 Times '

        if max_index_col==1:
            prediction_text=' The Patient will be readmitted < 30 Times '

        if max_index_col==2:
            prediction_text=' The Patient will be not be readmitted '
   
    return render_template('regression.html', form=form,prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)