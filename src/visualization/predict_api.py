from flask import Flask
from flask import request
import pickle
import os
import numpy as np


from src.Utils import normalize, load_processed_data, load_model, predict, _get_accuracy,classification_report

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def get_predict():
    model = load_model()
    data_dict = load_processed_data()
    predictions = predict(model, data_dict['test_X'])

    label_to_names = data_dict['label_to_names']
    test_y = data_dict['test_y']

    class_report = classification_report(test_y, predictions, target_names=list(label_to_names.values()))
    return  class_report
    #return str(label_to_names[predictions[0]])

@app.route('/prediction', methods = ['GET', 'POST'])
def get_prediction():

    dataset = load_processed_data()
    preprocess_settings = pickle.load(open(os.path.join(os.getcwd(), 'data/processed/data_mean_std.pkl'), 'rb'))
    pp_mean = preprocess_settings['mean'].astype('float')
    pp_std = preprocess_settings['std'].astype('float')
    train_X = dataset['train_X'].astype('int32')
    train_y = dataset['train_y']

    input_data = train_X, train_y

    input_data = np.array(train_X).reshape(-1, 3, 32, 32)
    input_data = np.array(input_data)
    input_data = normalize(input_data, pp_mean, pp_std)

    label_to_names = load_processed_data()['label_to_names']

    model = load_model()
    prediction = predict(model, input_data)

    return str(label_to_names[prediction[0]])

if __name__ == '__main__':
   app.run()