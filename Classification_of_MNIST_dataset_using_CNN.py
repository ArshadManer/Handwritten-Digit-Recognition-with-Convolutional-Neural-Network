import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model, Sequential

class MNISTClassifier:
    def __init__(self):
        self.model = None

    def load_mnist_data(self):
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
        return X_train, Y_train, X_test, Y_test

    def preprocess_data(self, X_train, X_test):
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train = X_train / 255
        X_test = X_test / 255
        return X_train, X_test

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(28, (3, 3), strides=(1, 1), input_shape=(28, 28, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        self.model = model

    def train_model(self, X_train, Y_train, epochs):
        history = self.model.fit(x=X_train, y=Y_train, epochs=epochs)
        return history

    def evaluate_model(self, X_test, Y_test):
        evaluation = self.model.evaluate(X_test, Y_test)
        return evaluation

def main():
    mnist_classifier = MNISTClassifier()

    # Loading data
    X_train, Y_train, X_test, Y_test = mnist_classifier.load_mnist_data()

    # Preprocessing data
    X_train, X_test = mnist_classifier.preprocess_data(X_train, X_test)

    # Building the model
    mnist_classifier.build_model()

    # Training the model
    epochs = 2
    history = mnist_classifier.train_model(X_train, Y_train, epochs)

    # Evaluating the model
    evaluation = mnist_classifier.evaluate_model(X_test, Y_test)

    print("Evaluation Results:", evaluation)

if __name__ == "__main__":
    main()
