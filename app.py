from flask import Flask, request, jsonify, render_template
from forms import PredictForm
app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisjustatest'
@app.route('/', methods=['GET','POST'])
def indexfunc():
    form=PredictForm()
    first=form.first.data
    print('the value entered was', first)
    return render_template('regression.html', form=form)

if __name__ == "__main__":
    app.run(debug=True)