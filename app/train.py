import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Model
from keras.datasets import boston_housing
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import plot_model

import params
from preprocessing import preprocess_dataset
from model import DNNModel


def main():
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    train_data = preprocess_dataset(dataset=train_data, is_training=True)

    model: Model = DNNModel().build()

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'])
    model.summary()
    plot_model(model, to_file='model.pdf', show_shapes=True)

    callbacks = [
        EarlyStopping(patience=20),
        ModelCheckpoint(filepath=params.MODEL_FILE_PATH, save_best_only=True),
        TensorBoard(log_dir=params.LOG_DIR)]

    model.fit(
        x=train_data,
        y=train_labels,
        epochs=params.EPOCHS,
        validation_split=params.VALIDATION_SPLIT,
        callbacks=callbacks)
    
if __name__ == '__main__':
    main()
    # tensorboard --logdir ./logs
