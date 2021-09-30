from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from nptyping import NDArray, Float64
from typing import Dict, Tuple, List
from tensorflow.python.keras.callbacks import History


class RNN_loop:
    """ A class to generate several RNN models with different number of layers.

    Attributes
    ----------
        num_outputs
        num_rows
        num_columns
        num_channels
        num_models
        DP_rate
    Methods
    ----------
        build_RNN_model
        initialize
        train
        evaluate_model
        plot_history
        plot_all_histories
        duration_vs_models
        accuracies_vs_models
        save_best_model
        save_all_models
    """

    def __init__(self, num_outputs: int, num_rows: int = 40, num_columns: int = 173,
                 num_channels: int = 1, num_models: int = 5, DP_rate: float = 0.3) -> None:
        """
        Args:
            num_outputs (int): Number of desired class labels
            num_rows (int, optional): Row from the dimension of spectrogram =  number of frequencies.
            num_columns (int, optional): Column from the dimension of spectrogram/number of time-steps.
            num_channels (int, optional): Defaults to 1.
            num_models (int, optional): Defaults to 5.
            DP_rate (float, optional): Dropout rate to be applied on intermediate layers. Defaults to 0.3 (best result)
        """

        self.num_outputs = num_outputs
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_channels = num_channels
        self.num_models = num_models  # number of models to be built
        self.histories: Dict[str, History] = {}
        self.durations: Dict[str, timedelta] = {}
        self.training_scores: Dict[str, list] = {}
        self.testing_scores: Dict[str, list] = {}
        self.DP_rate = DP_rate

        # Initialize a dictionary to store different models
        self.models: Dict[str, Sequential] = {}

    def build_RNN_model(self) -> Dict[str, Sequential]:
        """ Builds a desired number of RNN models.
            All starting with an LSTM & Dropout as input layer and all ending with a Dense layer.
            First model has 1 intermediate layer with dropout (DP_rate), second model has 2 intermediate layers etc.
            Until the desired number of models is reached.

        Args:
            num_models (int): Desired number of models, each with corresponding number of intermediate layers

        Returns:
            models (dictionary): All generated RNN models
        """

        # LSTM input shape = (batch, time_step = 173 , features = 40)
        # Spectrogram shape = (frequencies = 40, time_step = 173)
        # Inputs are transposed in "preprocessing.py"
        self.input_shape = (self.num_columns, self.num_rows)

        for i in range(1, self.num_models + 1):

            # First add LSTM layer and dropout (value=0.2) for all models
            self.models[f"model_{i}"] = Sequential()
            self.models[f"model_{i}"].add(LSTM(128, input_shape=(self.input_shape)))
            self.models[f"model_{i}"].add(Dropout(0.2))

        # Add intermediate layers - loop over models
        for j in range(self.num_models, 0, -1):

            # loop over layers
            for k in range(j, 0, -1):

                number_neurons = 2**(k + 3)
                self.models[f"model_{j}"].add(Dense(number_neurons, activation="relu"))

                if self.DP_rate != 1.0:
                    self.models[f"model_{j}"].add(Dropout(self.DP_rate))  # DP_rate changes
                else:
                    continue

        # Add a final dense layer
        for i in range(1, self.num_models + 1):
            self.models[f"model_{i}"].add(Dense(self.num_outputs, activation="softmax"))

        return self.models

    def initialize(self, X_test: NDArray[Float64], y_test: NDArray[Float64], learning_rate: float = 0.001) -> None:
        """  Initializes RNN models with categorical cross entropy, prints the summaries, prints pre-training accuracies.

        Args:
            X_test (numpy array): features of testing data
            y_test (numpy array): labels of testing data
            learning_rate (float, optional): learning rate. Defaults to 0.001.
        """
        opt = optimizers.Adam(learning_rate=learning_rate)
        scores = {}
        accuracies = {}

        for i in range(1, self.num_models + 1):
            self.models[f"model_{i}"].compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)

            print(" SUMMARY FOR MODEL ", i)
            self.models[f"model_{i}"].summary()

            scores[f"score_{i}"] = self.models[f"model_{i}"].evaluate(X_test, y_test, verbose=1)
            accuracies[f"accuracy_{i}"] = 100 * scores[f"score_{i}"][1]

            print("Pre-training accuracy: %.4f%%" % accuracies[f"accuracy_{i}"])
            print()

    def train(self, X_train: NDArray[Float64], X_test: NDArray[Float64], y_train: NDArray[Float64],
              y_test: NDArray[Float64], batch_size: int, num_epochs: int,
              verbose: int = 1) -> Tuple[Dict[str, History], Dict[str, timedelta]]:

        """ Trains all RNN models, prints training duration

        Args:
            X_train (numpy array): features of training data
            X_test (numpy array): features of testing data
            y_train (numpy array): labels of training data
            y_test (numpy array): labels of testing data
            batch_size (int): batch size
            num_epochs (int): number of epochs
            verbose (int, optional): verbose. Defaults to 1.

        Returns:
            histories (dictionary): Records events of each RNN model into a History object and collects in a dictionary
            durations (dictionary): Collects duration of training for each RNN model into datetime.timedelta object
        """

        early_stop = {}  # instantiate callbacks for early stopping
        best_models = {}  # instantiate model checkpoint callbacks to save the best model

        for i in range(1, self.num_models + 1):

            # create an "early stopping" callback (stops training when triggered)
            # patience = number of epochs with no improvement after which training will be stopped
            early_stop[f"es_{i}"] = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=50)

            # create a "model checkpoint" callback to save the best model
            best_models[f"best_{i}"] = (
                ModelCheckpoint(filepath=f'models/saved_models/best_models_DP_{self.DP_rate}.hdf5',
                                verbose=1, save_best_only=True))

            print("Training for model ", i, " has started.")
            start = datetime.now()
            self.histories[f"history_{i}"] = (
                self.models[f"model_{i}"].fit(
                    X_train, y_train, batch_size=batch_size,
                    epochs=num_epochs, validation_data=(X_test, y_test), verbose=verbose,
                    callbacks=[early_stop[f"es_{i}"], best_models[f"best_{i}"]]))

            duration = datetime.now() - start

            print("Training for model ", i, " completed in time: ", duration, "seconds")
            self.durations[f"duration_{i}"] = duration

        return self.histories, self.durations

    def evaluate_model(self, X_train: NDArray[Float64], X_test: NDArray[Float64],
                       y_train: NDArray[Float64],
                       y_test: NDArray[Float64]) -> Tuple[Dict[str, list], Dict[str, list]]:
        """ Evaluates all RNN models, prints accuracies and duration of training

        Args:
            X_train (numpy array): features of training data
            X_test (numpy array): features of testing data
            y_train (numpy array): labels of training data
            y_test (numpy array): labels of testing data

        Returns:
            training_scores (dict): list of metrics
            testing_scores (dict): list of metrics
        """

        for i in range(1, self.num_models + 1):

            print("SCORES FOR MODEL WITH ", i, " LAYERS : ")

            self.training_scores[f"training_score_{i}"] = (
                self.models[f"model_{i}"].evaluate(X_train, y_train, verbose=0))

            print("Training Accuracy: ", self.training_scores[f"training_score_{i}"][1])

            self.testing_scores["testing_score_" + str(i)] = (
                self.models[f"model_{i}"].evaluate(X_test, y_test, verbose=0))

            print("Testing Accuracy: ", self.testing_scores[f"testing_score_{i}"][1])

            print("Duration of training: ", self.durations[f"duration_{i}"], "\n")

        return self.training_scores, self.testing_scores

    def plot_history(self, which_model: int = 1) -> None:
        """ Plots accuracy/loss for training/validation set of a model as a function of the epochs

        Args:
            which_model (int, optional): Selects the desired model. Defaults to 1.
        """

        fig, axs = plt.subplots(2, figsize=(10, 6), dpi=75)

        # Accuracy subplot
        axs[0].plot(self.histories[f"history_{which_model}"].history["accuracy"], label="train accuracy")
        axs[0].plot(self.histories[f"history_{which_model}"].history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy Evaluation")

        # Error subplot
        axs[1].plot(self.histories[f"history_{which_model}"].history["loss"], label="train error")
        axs[1].plot(self.histories[f"history_{which_model}"].history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error Evaluation")

        plt.tight_layout()
        plt.show()

    def plot_all_histories(self) -> None:

        for i in range(1, self.num_models + 1):

            print(" Plots for Model ", i)
            self.plot_history(which_model=i)
            print()

    def save_best_model(self, best_model: int = 1) -> None:            # DP changes here
        self.models[f"model_{best_model}"].save(f"models/saved_models_DP_{self.DP_rate}/best_model")

    def save_all_models(self) -> None:

        for i in range(1, self.num_models + 1):         # DP changes here
            self.models[f"model_{i}"].save(f"models/saved_models_DP_{self.DP_rate}/saved_model_{i}")

    def accuracies_vs_models(self) -> None:
        """Plots testing and training accuracies for each generated model in a bar plot
        """

        width = 0.35  # the width of the bars

        training_scores = np.array(list(self.training_scores.values()))[:, 1]
        testing_scores = np.array(list(self.testing_scores.values()))[:, 1]
        self.number_of_models = np.arange(1, self.num_models + 1)

        # Training / Testing accuracies vs models

        fig, ax = plt.subplots(figsize=(10, 6), dpi=85)
        ax.bar(self.number_of_models - width/2, training_scores, width, label='Train Accuracy')
        ax.bar(self.number_of_models + width/2, testing_scores, width, label='Test Accuracy')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracies')
        ax.set_xlabel("Number of Model Layers")
        ax.set_title('Accuracies vs Number of Model Layers')
        ax.set_xticks(self.number_of_models)
        ax.legend(loc="lower right")

        for i, v in enumerate(training_scores):
            plt.annotate(str(np.round(v, 4)), xy=(
                self.number_of_models[i], training_scores[i]), ha='left',
                    va='bottom', xytext=(-50, 3), textcoords="offset points")

        for i, v in enumerate(testing_scores):
            plt.annotate(str(np.round(v, 4)),  xy=(
                self.number_of_models[i], testing_scores[i]), ha='center',
                    va='bottom', xytext=(50, 3), textcoords="offset points")

        plt.tight_layout()
        plt.show()

    def duration_vs_models(self) -> None:
        """Plots durations for each generated model in a bar plot
        """

        plt.figure(figsize=(10, 6), dpi=85)
        self.number_of_models = np.arange(1, self.num_models + 1)

        # Convert datetime.timedelta object to total minutes and floor to 4 decimals
        duration_list = list(self.durations.values())
        duration_mins: List[float] = []

        for i in range(len(duration_list)):

            duration_mins.append(duration_list[i].total_seconds() / 60.0)

        duration_array = np.array(np.round(duration_mins, 4))

        plt.bar(self.number_of_models, duration_array, width=0.35)

        # Print values above markers
        for i, v in enumerate(duration_array):
            plt.annotate(str(v), xy=(self.number_of_models[i], duration_array[i]), ha='left', va='bottom')

        plt.ylabel("Time [minutes] ")
        plt.xlabel("Number of Model Layers")
        plt.title("Duration Time [min] vs Number of Model Layers")
        plt.xticks(np.arange(min(self.number_of_models), max(self.number_of_models) + 1, 1.0))
        plt.tight_layout()
        plt.show()
