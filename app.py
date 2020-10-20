import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import pickle
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('done.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if("text" == "M"):
        text = 0
    else:
        text =1
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if output==0:
        output1='Normal'
    elif output==1:
        output1='Diabetic'
    elif output==2:
        output1='Pre-Diabetic'

    return render_template('index1.html', prediction_text='The Patient is {}'.format(output1))
    
if __name__ == "__main__":
    app.run(debug=True) 