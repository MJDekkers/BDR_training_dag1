from flask import Flask
from flask import request
import pickle
import os
import numpy as np

from src.models.Predict_model import predict_model
from src.Utils import normalize, load_processed_data, load_model

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Welcome!'


@app.route('/get_prediction', methods = ['POST'])
def get_prediction():
    print(request.is_json)
    content = request.json

    preprocess_settings = pickle.load(open(os.path.join(os.getcwd(), 'data/processed/data_incl_mean_std.pkl'), 'rb'))
    pp_mean = preprocess_settings['mean']
    pp_std = preprocess_settings['std']

    input_data = np.array(content).reshape(-1, 3, 32, 32)
    input_data = normalize(input_data, pp_mean, pp_std)

    label_to_names = load_processed_data()['label_to_names']

    model = load_model()
    prediction = predict_model(model, input_data)

    return str(label_to_names[prediction[0]])


if __name__ == '__main__':
   app.run()