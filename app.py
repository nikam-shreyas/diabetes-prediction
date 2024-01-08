import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

FILE_PATH = 'diabetes_model'

app = Flask(__name__)
model = pickle.load(open(f'{FILE_PATH}/model.pkl', 'rb'))
scaler = pickle.load(open(f'{FILE_PATH}/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    prediction = model.predict(scaler.transform(np.array(list(data.values())).reshape(1, -1)))
    print(prediction)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)