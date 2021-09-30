from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sn
from sklearn.metrics import precision_score, recall_score
from src.visualization.visualize import Visualize
from src.processing import preprocessing


class RNN_eval:
    """Class to load and evaluate a model.
    """

    def __init__(self):

        # self.model_file_path = model_file_path

        fulldatasetpath = '/datasets/UrbanSound8K/audio/'
        metadata = pd.read_csv(fulldatasetpath + '../metadata/UrbanSound8K.csv')

        # Sort class names based on classID (0,1,2...9)
        classID = metadata.classID.unique().tolist()
        className = metadata["class"].unique().tolist()
        self.labels_sorted = [x for _, x in sorted(zip(classID, className))]

        for class_ID, label in enumerate(self.labels_sorted):
            print(class_ID, label)

        # Create visualization object
        self.vis = Visualize()

    # Method to work on a live model instead of loading a saved one
    def eval_on_new_model(self, new_model):

        self.model = new_model

    def load_model(self, model_file_path="models/saved_models_DP_0.2/best_model"):

        self.model_file_path = model_file_path
        self.model = load_model(self.model_file_path)
        self.model.summary()

        return self.model

    def evaluate(self, X_train, X_test, y_train, y_test, verbose=1):

        self.training_accuracy = self.model.evaluate(X_train, y_train, verbose)[1]
        self.testing_accuracy = self.model.evaluate(X_test, y_test, verbose)[1]

        print("Testing Accuracy: ", self.testing_accuracy)
        print("Training Accuracy: ", self.training_accuracy)

        return self.training_accuracy, self.testing_accuracy

    def make_prediction(self, X):

        self.y_pred = self.model.predict_classes(X)

    def create_confusion_matrix(self, y):

        # True labels is a 1-hot vector. Convert it to single digits.
        self.y_labels = np.argmax(y, axis=1)

        self.cm = confusion_matrix(self.y_labels, self.y_pred)
        self.df_confusion = pd.crosstab(
            self.y_labels, self.y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

        return self.df_confusion

    def tabulate_confusion_matrix(self):

        data = {'y_labels': self.y_labels, 'y_pred': self.y_pred}
        class_range = np.arange(0, 10)

        plt.figure(figsize=(15, 6), dpi=85)
        plt.title("Confusion Matrix for 10 Labels and 7327 Samples")

        df = pd.DataFrame(data, columns=['y_labels', 'y_pred'])
        confusion_matrix = pd.crosstab(
            df['y_labels'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'], margins=True)
        sn.heatmap(confusion_matrix, annot=True, fmt='g')

        plt.xticks(class_range, self.labels_sorted, rotation=45)
        plt.yticks(class_range, self.labels_sorted, rotation=45)

        plt.show()
        plt.tight_layout()

    def calculate_precision(self):

        precision = precision_score(self.y_labels, self.y_pred, average=None)
        precision_data = {"labels_sorted": self.labels_sorted, "precision_score": precision}
        self.precision_df = pd.DataFrame(precision_data, columns=["labels_sorted", "precision_score"])
        pd.options.display.float_format = "{:,.4f}".format

        self.precision_df.style.set_table_attributes('style="font-size: 15px"')

        return self.precision_df

    def calculate_recall(self):

        recall = recall_score(self.y_labels, self.y_pred, average=None)
        recall_data = {"labels_sorted": self.labels_sorted, "recall_score": recall}
        self.recall_df = pd.DataFrame(recall_data, columns=["labels_sorted", "recall_score"])
        pd.options.display.float_format = "{:,.4f}".format

        self.recall_df.style.set_table_attributes('style="font-size: 15px"')

        return self.recall_df

    def plot_sample_rates_mispredicted_classes(self, data_df):

        # data_df = preprocessing.filter_mfccs(raw_data_df)
        new_df = preprocessing.add_sample_rate_to_features(self.y_labels, self.y_pred, data_df)
        self.vis.plot_sample_rates_mispredicted_classes(new_df)
