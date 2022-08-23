import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api._v2 import keras
from keras.datasets import boston_housing
from rich import print

import params
from preprocessing import preprocess_dataset


def predict(dataset):
    model = keras.models.load_model(params.MODEL_FILE_PATH)
    pred = model.predict(dataset).flatten()
    return pred


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    test_data, test_labels = test_data[:10], test_labels[:10]  # 10データ分だけ推論をおこなう
    test_data = preprocess_dataset(dataset=test_data, is_training=False)
    pred = predict(dataset=test_data)
    print(f'prediction: {np.round(pred, 1)}')
    print(f'labels: {test_labels}')
