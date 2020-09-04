import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import argparse


class Train():
    def __init__(self, opt):
        super().__init__()

        batch_size = 4
        no_epochs = 100
        learning_rate = 0.001
        no_classes = 3
        validation_split = 0.2
        verbosity = 1

        file_1 = open(opt.train_data, "rb")
        X_train = np.load(file_1)

        file_2 = open(opt.train_label, "rb")
        Y_train = np.load(file_2)

        file_3 = open(opt.test_data, "rb")
        X_test = np.load(file_3)

        file_4 = open(opt.test_label, "rb")
        Y_test = np.load(file_4)

        # Determine sample shape
        sample_shape = (520, 520, 15)

        targets_train = to_categorical(Y_train).astype(np.integer)
        targets_test = to_categorical(Y_test).astype(np.integer)

        # Create the model
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform',
                   input_shape=sample_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(no_classes, activation='softmax'))

        # Compile the model
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=learning_rate),
                      metrics=['accuracy'])

        # Fit data to model
        history = model.fit(X_train, targets_train,
                            batch_size=batch_size,
                            epochs=no_epochs,
                            verbose=verbosity,
                            validation_split=validation_split)

        # Generate generalization metrics
        score = model.evaluate(X_test, targets_test, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


def get_args():
    parser = argparse.ArgumentParser("Pre Processing Video Classification")
    parser.add_argument("-db1", "--train_data", type=str, default="train_data")
    parser.add_argument("-db2", "--train_label", type=str, default='train_label')
    parser.add_argument("-db3", "--test_data", type=str, default='test_data')
    parser.add_argument("-db4", "--test_label", type=str, default='test_label')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    Train(opt)