import numpy as np

from datetime import datetime 
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import optimizers
from keras.utils import np_utils
from sklearn import metrics 


class CNN:
    def __init__(self, num_outputs, num_rows=40, num_columns=173, num_channels=1):
        self.num_outputs = num_outputs
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_channels = num_channels
        self.model = self.build_model()

    def build_model(self):
        # Construct model 
        model = Sequential()
        model.add(
            Conv2D(filters=16, kernel_size=2,
            input_shape=(self.num_rows, self.num_columns, self.num_channels), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(GlobalAveragePooling2D())

        model.add(Dense(self.num_outputs, activation='softmax'))
        return model

    def initialize(self, X_test, y_test, learning_rate=0.001):
        opt = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

        # Display model architecture summary 
        self.model.summary()

        # Calculate pre-training accuracy 
        score = self.model.evaluate(X_test, y_test, verbose=1)
        accuracy = 100*score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy) 

    def train(self, X_train, X_test, y_train, y_test, num_epochs, batch_size, verbose=1):
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                                    verbose=1, save_best_only=True)
        start = datetime.now()

        history = self.model.fit(
            X_train, y_train, batch_size=batch_size, epochs=num_epochs,
            validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=verbose)

        duration = datetime.now() - start
        print("Training completed in time: ", duration)
        return history

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        score = self.model.evaluate(X_train, y_train, verbose=0)
        print("Training Accuracy: ", score[1])

        score = self.model.evaluate(X_test, y_test, verbose=0)
        print("Testing Accuracy: ", score[1])