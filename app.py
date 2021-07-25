from __future__ import division
from math import sqrt
from flask import Flask, render_template, request, jsonify
from flask import Flask, request
from predict import Predictor
from model import Model
import json
from bson import json_util

app = Flask(__name__)

M = Model()
predictor = Predictor()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print('Predict ----------------')
    text = request.json
    print('Text >> ', text)
    prediction = predictor.predict([text])
    print('Prediction >> ', prediction)

    # prediction = pd.DataFrame(prediction).to_html()
    # return prediction
    # return jsonify({'prediction': str(prediction)})
    return jsonify(prediction)
    #
    # return render_template('index.txt', predictions=prediction)


@app.route('/my_network', methods=['GET'])
def my_network():
    my_network_predictions = predictor.my_network_json()
    return json.dumps(my_network_predictions, default=json_util.default)
    # return jsonify(my_network_predictions)
    #
    # return render_template('index.txt', predictions=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
