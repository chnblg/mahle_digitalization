from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from nptyping import NDArray, Float64
from typing import Dict, Tuple, List
from tensorflow.python.keras.callbacks import History
from src.helpers import pickle_keras_models


class RNN_std:

    def __init__(self, num_outputs: int, num_rows: int = 40, num_columns: int = 173,
                 num_channels: int = 1, DP_rate: float = 0.2) -> None:

        self.num_outputs = num_outputs
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_channels = num_channels
        self.DP_rate = DP_rate
        self.model = self.build_RNN_model()

    def build_RNN_model(self):

        # LSTM input shape = (batch, time_step = 173 , features = 40)
        # Spectrogram shape = (frequencies = 40, time_step = 173)
        # Inputs are transposed in "preprocessing.py"
        self.input_shape = (self.num_columns, self.num_rows)

        model = Sequential()
        model.add(LSTM(128, input_shape=(self.input_shape)))  # (X_train.shape[1], X_train.shape[2])
        model.add(Dropout(self.DP_rate))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(self.DP_rate))
        model.add(Dense(self.num_outputs, activation="softmax"))

        return model

    def initialize(self, X_test: NDArray[Float64], y_test: NDArray[Float64], learning_rate: float = 0.001) -> None:

        # call helper function to make model picklable
        pickle_keras_models.make_keras_picklable()

        opt = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)

        # Display model architecture summary
        self.model.summary()

        # Calculate pre-training accuracy
        score = self.model.evaluate(X_test, y_test, verbose=1)
        accuracy = 100*score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy)

    def train(self, X_train: NDArray[Float64], X_test: NDArray[Float64], y_train: NDArray[Float64],
              y_test: NDArray[Float64], batch_size: int, num_epochs: int, patience=50,
              verbose: int = 1, early_stop: bool = True) -> Tuple[History, timedelta]:

        start = datetime.now()

        if early_stop:
            es = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=patience)
            history = self.model.fit(X_train, y_train, batch_size=batch_size,
                                     epochs=num_epochs, validation_data=(X_test, y_test), verbose=verbose,
                                     callbacks=[es])
        else:
            history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                                     validation_data=(X_test, y_test), verbose=verbose)

        self.duration = datetime.now() - start
        print("Training completed in time: ", self.duration)
        return history, self.duration

    def evaluate_model(self, X_train: NDArray[Float64], X_test: NDArray[Float64], y_train: NDArray[Float64],
                       y_test: NDArray[Float64]) -> None:

        self.training_acc = self.model.evaluate(X_train, y_train, verbose=0)
        self.testing_acc = self.model.evaluate(X_test, y_test, verbose=0)

        print("Testing Accuracy: ", self.testing_acc[1])
        print("Duration of training: ", self.duration, "\n")

    def plot_history(self, history) -> None:
        fig, axs = plt.subplots(2, figsize=(10, 6), dpi=75)

        # Accuracy subplot
        axs[0].plot(history.history["accuracy"], label="train accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy Evaluation")

        # Error subplot
        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error Evaluation")

        plt.tight_layout()
        plt.show()

    def accuracies_vs_models(self) -> None:
        width = 0.35  # the width of the bars

        training_scores = np.array(list(self.training_acc.values()))[:, 1]
        testing_scores = np.array(list(self.testing_acc.values()))[:, 1]
        number_of_models = np.arange(5)

        # Training / Testing accuracies vs models

        fig, ax = plt.subplots(figsize=(10, 6), dpi=85)
        ax.bar(number_of_models - width/2, training_scores, width, label='Train Accuracy')
        ax.bar(number_of_models + width/2, testing_scores, width, label='Test Accuracy')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracies')
        ax.set_xlabel("Number of Model Layers")
        ax.set_title('Accuracies vs Number of Model Layers')
        ax.set_xticks(number_of_models)
        ax.legend(loc="lower right")

        for i, v in enumerate(training_scores):
            plt.annotate(str(np.round(v, 4)), xy=(
                number_of_models[i], training_scores[i]), ha='left',
                    va='bottom', xytext=(-50, 3), textcoords="offset points")

        for i, v in enumerate(testing_scores):
            plt.annotate(str(np.round(v, 4)),  xy=(
                number_of_models[i], testing_scores[i]), ha='center',
                    va='bottom', xytext=(50, 3), textcoords="offset points")

        plt.tight_layout()
        plt.show()

    # def duration_vs_models(self, models: Dict[str, list], durations: list) -> None:
    #     plt.figure(figsize=(10, 6), dpi=85)
    #     number_of_models = np.arange(len(models))

    #     # Convert datetime.timedelta object to total minutes and floor to 4 decimals
    #     duration_list = list(durations.values())
    #     duration_mins: List[float] = []

    #     for i in range(len(duration_list)):

    #         duration_mins.append(duration_list[i].total_seconds() / 60.0)

    #     duration_array = np.array(np.round(duration_mins, 4))

    #     plt.bar(number_of_models, duration_array, width=0.35)

    #     # Print values above markers
    #     for i, v in enumerate(duration_array):
    #         plt.annotate(str(v), xy=(number_of_models[i], duration_array[i]), ha='left', va='bottom')

    #     plt.ylabel("Time [minutes] ")
    #     plt.xlabel("Number of Model Layers")
    #     plt.title("Duration Time [min] vs Number of Model Layers")
    #     plt.xticks(np.arange(min(number_of_models), max(number_of_models) + 1, 1.0))
    #     plt.tight_layout()
    #     plt.show()
