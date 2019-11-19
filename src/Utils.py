from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Input
import os
import numpy as np
import pickle
from sklearn.metrics import classification_report


def load_processed_data():
    pickle_dict_path = os.path.join(os.getcwd(), "data", "processed","cifar-10.pkl")
    return pickle.load(open(pickle_dict_path, 'rb'))


def load_processed_data_inc_mean_std():
    pickle_dict_path = os.path.join(os.getcwd(), "data", "processed","data_incl_mean_std.pkl")
    return pickle.load(open(pickle_dict_path, 'rb'))

def conv_net(train_X, nr_classes):
    input = Input(shape=train_X.shape[1:])

    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='valid',
                  data_format='channels_first', activation='relu')(input)
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='valid',
                  data_format='channels_first', activation='relu', strides=(2, 2))(conv)
    flatten = Flatten()(conv)

    output_layer = Dense(units=nr_classes, activation='softmax')(flatten)

    model = Model(inputs=input, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def normalize(data, mean, std):
        return (data-mean)/std


def predict(model, dataset):
    predictions = np.array(model.predict(dataset, batch_size=100))
    predictions = np.argmax(predictions, axis=1)
    return predictions


def make_predictions():
    model = load_model()
    data_dict = load_processed_data()
    predictions = predict(model, data_dict['test_X'])
    return predictions


def _get_accuracy(predictions, test_y):
    return np.sum(predictions == test_y) / float(len(predictions))


def get_results(predictions, save_class_report=True):
    data_dict = load_processed_data()
    label_to_names = data_dict['label_to_names']
    test_y = data_dict['test_y']

    accuracy = _get_accuracy(predictions, test_y)
    class_report = classification_report(test_y, predictions, target_names=list(label_to_names.values()))

    if save_class_report:
        class_report_path = os.path.join(os.getcwd(), 'reports', "last_results.txt")
        with open(class_report_path, 'w') as f:
            f.write(class_report)

    return accuracy, class_report


def save_model(model):
    model_filepath = _get_model_filepath()
    pickle.dump(model, open(model_filepath, 'wb'))

def _get_model_filepath():
        return os.path.join(os.getcwd(), 'data/model', 'model.pkl')


def load_model():
        model_filepath = _get_model_filepath()
        return pickle.load(open(model_filepath, 'rb'))



