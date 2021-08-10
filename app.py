from __future__ import division
from flask import Flask, render_template, request, jsonify
from predict import Predictor
from model import Model
import json
import bson

app = Flask(__name__)

M = Model()
predictor = Predictor()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print('Predict ----------------')
    text_json =  request.json
    text = text_json['predict']
    print('Text >> ',text )
    prediction = predictor.predict([text])
    print('Prediction >> ', prediction)

    # prediction = pd.DataFrame(prediction).to_html()
    # return prediction
    # return jsonify({'prediction': str(prediction)})
    return jsonify(prediction)
    #
    # return render_template('index.txt', predictions=prediction)




if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
