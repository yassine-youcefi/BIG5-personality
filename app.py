from __future__ import division
from flask import Flask, render_template, request, jsonify
from predict import Predictor
from model import Model
import json
import bson

app = Flask(__name__)

M = Model()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    predictor = Predictor()
    print('Predict ----------------')
    text =  request.form.get('predict')
    if text:
        print('Text >> ',text )

        # text_json =  request.json
        # text = text_json['predict']
        prediction = predictor.predict([text])
        # print('Prediction >> ', prediction)

        return jsonify(prediction)
    
    # return 'ok'

@app.route('/predict-json', methods=['POST'])
def predict_json():
    predictor = Predictor()
    text_json =  request.json
    text = text_json['predict']
    print('Text >> ',text )
    prediction = predictor.predict([text])

    return jsonify(prediction)




if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
