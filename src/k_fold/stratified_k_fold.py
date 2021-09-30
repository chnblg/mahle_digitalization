from src.processing import preprocessing
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import copy
from src.helpers import pickle_keras_models
from typing import Dict, Tuple, Callable
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import History
from datetime import timedelta


class KFoldModelTrainer:

    def __init__(self, fold: int = 5, load_processed_data=True,
                 default_path: str = "/datasets/UrbanSound8K/processed/mean_mfcc_data_with_sample_rates.json") -> None:

        self.folds = fold  # number of folds
        self.le = LabelEncoder()

        # load processed spectrogrammes
        if load_processed_data:
            self.raw_data_df = pd.read_json(default_path)

        # preprocess the raw dataset
        else:
            self.raw_data_df = preprocessing.load_dataset()
            self.raw_data_df['feature'] = self.raw_data_df['feature'].apply(
                preprocessing.calculate_mean_mfcc, preprocessing.DEFAULT_SAMPLE_RATE)

        # filter mfccs to same shape
        self.data_df = preprocessing.filter_mfccs(self.raw_data_df)

        # generate X and y arrays
        self.X_array: np.ndarray = np.array(self.data_df.feature.tolist())
        self.y_array: np.ndarray = np.array(self.data_df.label.tolist())
        
        self.mean_accuracy_train = 0.0
        self.mean_accuracy_train_loss = 0.0
        self.mean_accuracy_test = 0.0
        self.mean_accuracy_test_loss = 0.0

    # jsut for visualization, split will be repeated for actual training
    def create_kfold_data(self, input_RNN=True) -> None:

        skfold = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)

        # Sort unique class names based on ClassID (0,1,2 ... 9)
        metadata = preprocessing.extend_metadata()
        classID = metadata.classID.unique().tolist()
        classNames = metadata["class"].unique().tolist()
        labels_sorted = [x for _, x in sorted(zip(classID, classNames))]
        dc_temp: Dict[str, np.ndarray] = {}

        # enumerate the splits and summarize the distributions
        for ite, (train_ix, test_ix) in enumerate(skfold.split(self.X_array, self.y_array)):
            X_train, X_test = self.X_array[train_ix], self.X_array[test_ix]
            y_train, y_test = self.y_array[train_ix], self.y_array[test_ix]

            if input_RNN:
                # transpose to fit input shape of LSTM layer = schleife
                X_train = X_train.transpose(0, 2, 1)
                X_test = X_test.transpose(0, 2, 1)

            # print(f'Fold: {ite}, Train set: {len(train_ix)}, Test set: {len(test_ix)}')

            for ID, label in enumerate(labels_sorted):

                dc_temp[f"train_{ID}"] = len(y_train[y_train == f"{label}"])
                dc_temp[f"test_{ID}"] = len(y_test[y_test == f"{label}"])

            print(f'> Fold {ite+1}, Train: 0=%d, 1=%d, 2=%d, 3=%d, 4=%d, 5=%d, 6=%d, 7=%d, 8=%d, 9=%d,\
                  \n  Fold {ite+1}, Test: 0=%d, 1=%d, 2=%d, 3=%d, 4=%d, 5=%d, 6=%d, 7=%d, 8=%d, 9=%d' %
                  (dc_temp["train_0"], dc_temp["train_1"], dc_temp["train_2"],
                   dc_temp["train_3"], dc_temp["train_4"],
                   dc_temp["train_5"], dc_temp["train_6"], dc_temp["train_7"],
                   dc_temp["train_8"], dc_temp["train_9"],
                   dc_temp["test_0"], dc_temp["test_1"], dc_temp["test_2"],
                   dc_temp["test_3"], dc_temp["test_4"],
                   dc_temp["test_5"], dc_temp["test_6"], dc_temp["test_7"],
                   dc_temp["test_8"], dc_temp["test_9"]))
            print()

    # compile the model
    def initialize_model(self, model,
                         loss="categorical_crossentropy", metrics=['accuracy'], opt=Adam) -> None:

        # call helper function to make model picklable
        pickle_keras_models.make_keras_picklable()

        self.models: Dict[str, Sequential] = {}
        optimizer = opt(learning_rate=0.001)
        for i in range(self.folds):
            self.models[f"model_{i}"] = copy.deepcopy(model)
            # self.models[f"model_{i}"].compile(loss=loss, metrics=metrics, optimizer=optimizer)
            self.models[f"model_{i}"].compile(loss=loss, metrics=metrics, optimizer=copy.deepcopy(optimizer))

    # function to change training data for data augmentation
    def change_training_data(self, new_path: str) -> Tuple[np.ndarray, np.ndarray]:

        new_raw_data_df = pd.read_json(new_path)
        new_data_df = preprocessing.filter_mfccs(new_raw_data_df)

        # generate X and y arrays
        X_array: np.ndarray = np.array(new_data_df.feature.tolist())
        y_array: np.ndarray = np.array(new_data_df.label.tolist())

        return X_array, y_array

    def train_models(self,
                     new_path: str = "/datasets/UrbanSound8K/processed/mean_mfcc_data_with_sample_rates.json",
                     input_RNN=True,
                     batch_size: int = 256,
                     epochs: int = 1000,
                     patience: int = 50,
                     early_stop: bool = True,
                     train_on_new_df: bool = False,
                     data_augmentation_function: Callable = None) -> None:

        # Define per-fold score containers
        self.train_acc_per_fold: np.ndarray = []
        self.test_acc_per_fold: np.ndarray = []
        self.train_loss_per_fold: np.ndarray = []
        self.test_loss_per_fold: np.ndarray = []

        # save trainings of all models
        self.histories: Dict[str, History] = {}
        self.durations: Dict[str, timedelta] = {}

        skfold = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)

        if train_on_new_df:
            X_array, y_array = self.change_training_data(new_path)
        else:
            X_array = self.X_array
            y_array = self.y_array

        # start training
        for ite, (train_ix, test_ix) in enumerate(skfold.split(X_array, y_array)):

            X_train, X_test = X_array[train_ix], X_array[test_ix]
            y_train, y_test = y_array[train_ix], y_array[test_ix]

            # convert labels to 1-HOT array
            y_train = to_categorical(self.le.fit_transform(y_train))
            y_test = to_categorical(self.le.fit_transform(y_test))

            # apply data augmentation only on training data
            if data_augmentation_function is not None:
                X_train, y_train = data_augmentation_function(X_train, y_train)

            if input_RNN:
                # transpose features to fit input shape of LSTM layer
                X_train = X_train.transpose(0, 2, 1)
                X_test = X_test.transpose(0, 2, 1)
            else:
                # transpose features to fit input shape of CNN layer
                num_rows = X_array.shape[1]
                num_columns = X_array.shape[2]
                num_channels = 1
                X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
                X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

            print('------------------------------------------------------------------------')
            print(f'Training for fold {ite+1} ...')

            # implement early stopping
            start = datetime.now()
            if early_stop:
                es = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=patience)

                self.histories[f"history_{ite}"] = self.models[f"model_{ite}"].fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[es],
                    verbose=1)
            else:
                self.histories[f"history_{ite}"] = self.models[f"model_{ite}"].fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

            self.durations[f"duration_{ite}"] = datetime.now() - start

            # generate generalization metrics
            train_evals = self.models[f"model_{ite}"].evaluate(X_train, y_train, verbose=1)
            test_evals = self.models[f"model_{ite}"].evaluate(X_test, y_test, verbose=1)

            print(f'Testing scores for fold {ite+1}: \
                {self.models[f"model_{ite}"].metrics_names[0]} of {test_evals[0]}; \
                {self.models[f"model_{ite}"].metrics_names[1]} of {test_evals[1]*100}%')

            print("Training completed in time: ", self.durations[f"duration_{ite}"])

            self.train_acc_per_fold.append(train_evals[1] * 100)
            self.test_acc_per_fold.append(test_evals[1] * 100)
            self.train_loss_per_fold.append(train_evals[0])
            self.test_loss_per_fold.append(test_evals[0])
        
        self.mean_accuracy_train = np.mean(self.train_acc_per_fold)
        self.mean_accuracy_train_loss = np.mean(self.train_loss_per_fold)
        self.mean_accuracy_test = np.mean(self.test_acc_per_fold)
        self.mean_accuracy_test_loss = np.mean(self.test_loss_per_fold)
        
    # print results
    def post_train_eval(self):

        print('------------------------------------------------------------------------')
        print('Score per fold\n')

        for i in range(0, self.folds):
            print(f'> Fold {i+1} - Loss: {self.train_loss_per_fold[i]:.4f} \
                - Training Accuracy: {self.train_acc_per_fold[i]:.4f} %')
            print(f'> Fold {i+1} - Loss: {self.test_loss_per_fold[i]:.4f} \
                - Testing Accuracy: {self.test_acc_per_fold[i]:.4f} %\n')

        print('------------------------------------------------------------------------')
        print('Average scores for all folds:\n')
        print(f'> Training Accuracy: {np.mean(self.train_acc_per_fold):.4f} % \
            (+- {np.std(self.train_acc_per_fold):.4f})')
        print(f'> Testing Accuracy: {np.mean(self.test_acc_per_fold):.4f} % \
            (+- {np.std(self.test_acc_per_fold):.4f})')
        print(f'> Loss: {np.mean(self.test_loss_per_fold):.4f}')
        print('------------------------------------------------------------------------')

    def plot_histories(self) -> None:

        for fold in range(self.folds):

            print(f" Plots for Model {fold+1}")
            fig, axs = plt.subplots(2, figsize=(10, 6), dpi=75)

            # accuracy subplot
            axs[0].plot(self.histories[f"history_{fold}"].history["accuracy"], label="train accuracy")
            axs[0].plot(self.histories[f"history_{fold}"].history["val_accuracy"], label="test accuracy")
            axs[0].set_ylabel("Accuracy")
            axs[0].legend(loc="lower right")
            axs[0].set_title("Accuracy Evaluation")

            # error subplot
            axs[1].plot(self.histories[f"history_{fold}"].history["loss"], label="train error")
            axs[1].plot(self.histories[f"history_{fold}"].history["val_loss"], label="test error")
            axs[1].set_ylabel("Error")
            axs[1].set_xlabel("Epoch")
            axs[1].legend(loc="upper right")
            axs[1].set_title("Error Evaluation")

            plt.tight_layout()
            plt.show()

    def plot_accuracies_vs_folds(self) -> None:

        width = 0.15

        # Training / Testing accuracies vs models
        ticks = np.arange(1, self.folds + 1)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=85)
        ax.bar(ticks - width/2, self.train_acc_per_fold, width, label='Train Accuracy')
        ax.bar(ticks + width/2, self.test_acc_per_fold, width, label='Test Accuracy')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracies')
        ax.set_xlabel("Folds")
        ax.set_title('Accuracies vs Folds')
        ax.set_xticks(ticks)
        ax.legend(loc="lower right")

        for i, v in enumerate(self.train_acc_per_fold):
            plt.annotate(str(np.round(v, 4)), xy=(
                ticks[i], self.train_acc_per_fold[i]), ha='left',
                    va='bottom', xytext=(-50, 3), textcoords="offset points")

        for i, v in enumerate(self.test_acc_per_fold):
            plt.annotate(str(np.round(v, 4)),  xy=(
                ticks[i], self.test_acc_per_fold[i]), ha='center',
                    va='bottom', xytext=(50, 3), textcoords="offset points")

        plt.tight_layout()
        plt.show()

    # change path variable according to the project
    def save_all_models(self, path="data_science/models/lstm_k_fold/saved_model_") -> None:

        for i in range(0, self.folds):
            self.models[f"model_{i}"].save(path + str(i))

    # change path variable according to the project
    def save_best_model(self, best_fold: int, path="data_science/models/lstm_k_fold/best_model") -> None:

        self.models[f"model_{best_fold}"].save(path)
