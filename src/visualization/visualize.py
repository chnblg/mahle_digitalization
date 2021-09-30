import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from src.processing import preprocessing


class Visualize:

    def __init__(self):

        # DEFAULT_PATH = '/datasets/UrbanSound8K/'
        # self.metadata = pd.read_csv(DEFAULT_PATH + 'metadata/UrbanSound8K.csv')

        self.metadata = preprocessing.extend_metadata()
        self.audiodf = preprocessing.create_audiodf()

        # Sort unique class ID´s
        classID = self.metadata.classID.unique().tolist()
        self.classID = sorted(classID)

        # Sort unique class names based on ClassID (0,1,2 ... 9)
        classNames = self.metadata["class"].unique().tolist()
        self.labels_sorted = [x for _, x in sorted(zip(classID, classNames))]

        # Get sorted native sample rates and their counts
        native_sample_rates = self.audiodf["sample_rate"].value_counts().index.tolist()
        sample_rate_counts = self.audiodf.sample_rate.value_counts().tolist()

        self.rates_sorted = sorted(native_sample_rates)
        self.counts_sorted = [x for _, x in sorted(zip(native_sample_rates, sample_rate_counts))]

        # Get unique label names = 10
        self.num_outputs = self.metadata['classID'].unique().shape[0]

        # Get total number of sample rates = 11
        self.total_different_sample_rates = self.metadata["sample_rate"].unique().shape[0]

    def df_sample_rate_vs_counts(self):

        sample_rate_counts = {"sample_rates_sorted": self.rates_sorted, "counts": self. counts_sorted}
        sample_rate_counts_df = pd.DataFrame.from_dict(sample_rate_counts)
        return sample_rate_counts_df

    def plot_samples_vs_native_sample_rates(self):

        fig, ax = plt.subplots(figsize=(10, 6), dpi=85)
        width = 0.35

        self.types = np.arange(0, self.total_different_sample_rates)  # 11
        ax.bar(self.types - width/2, self.counts_sorted, width, align="edge")
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Sample Rates [Hz]")
        ax.set_title("Number of Samples vs Native Sample Rates")

        for i, v in enumerate(self.counts_sorted):
            plt.annotate(str(v), xy=(
                self.types[i], self.counts_sorted[i]), ha='center',
                    va='bottom', xytext=(0, 0), textcoords="offset points")

        plt.xticks(self.types, self.rates_sorted, rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_samples_vs_classes(self):

        # Sort sample counts based on ClassID (0, 1, 2 ... 9)
        self.sample_counts_sorted = self.metadata.groupby('class').count()["classID"].tolist()

        plt.figure(figsize=(10, 6), dpi=85)
        plt.bar(self.classID, self.sample_counts_sorted, width=0.35)
        plt.ylabel("Number of Samples")
        plt.xlabel("Labels (Classes)")
        plt.title("Total Number of Samples vs Classes")
        plt.xticks(self.classID, self.labels_sorted, rotation=45)

        for i, v in enumerate(self.sample_counts_sorted):
            plt.annotate(str(v), xy=(
                self.classID[i], self.sample_counts_sorted[i]), ha='center',
                    va='bottom', xytext=(0, 0), textcoords="offset points")

        plt.tight_layout()
        plt.show()

    # col_name = either "true_pred" or "false_pred"
    # new_df = processing.add_samplerate_to_features()
    def dict_sample_rate_counts_vs_classes(self, col_name, new_df):

        self.all_classes_df = {}

        for i in range(len(self.labels_sorted)):

            self.all_classes_df[f"{self.labels_sorted[i]}"] = (
                new_df.loc[new_df[f"{col_name}"] == f"{self.labels_sorted[i]}"])

            self.all_classes_df[f"{self.labels_sorted[i]}_sample_rate_counts"] = (
                self.all_classes_df[f"{self.labels_sorted[i]}"].sample_rate.value_counts().tolist())

            self.all_classes_df[f"{self.labels_sorted[i]}_native_sample_rates"] = (
                self.all_classes_df[f"{self.labels_sorted[i]}"].sample_rate.unique().tolist())

            self.all_classes_df[f"{self.labels_sorted[i]}_rates_sorted"] = (
                sorted(self.all_classes_df[f"{self.labels_sorted[i]}_native_sample_rates"]))

            self.all_classes_df[f"{self.labels_sorted[i]}_counts_sorted"] = (
                [x for _, x in sorted(
                    zip(self.all_classes_df[f"{self.labels_sorted[i]}_native_sample_rates"], (
                        self.all_classes_df[f"{self.labels_sorted[i]}_sample_rate_counts"])))])

            self.all_classes_df[f"{self.labels_sorted[i]}_length"] = (
                np.arange(0, len(self.all_classes_df[f"{self.labels_sorted[i]}_counts_sorted"])))

        self.all_classes = {}
        # Pad nonobserved frequencies of each class with zeros
        for i in range(len(self.labels_sorted)):

            self.all_classes[f"{self.labels_sorted[i]}_rates_sorted_pad"], (
                self.all_classes[f"{self.labels_sorted[i]}_counts_sorted_pad"]) = (
                    self.zero_pad_list(self.rates_sorted, self.all_classes_df[f"{self.labels_sorted[i]}_rates_sorted"],
                                       self.all_classes_df[f"{self.labels_sorted[i]}_counts_sorted"]))

        return self.all_classes

    # Function to pad nonoberserved frequencies of each class with zeros
    def zero_pad_list(self, full_list, short_list, counts_list):

        expanded_list = [x if x in short_list else 0 for x in full_list]
        expanded_counts_list = []
        index = 0
        for x in expanded_list:

            if x != 0:
                expanded_counts_list.append(counts_list[index])
                index = index + 1
            else:
                expanded_counts_list.append(0)

        return expanded_list, expanded_counts_list

    def df_sample_rate_counts_vs_classes(self):

        self.rate_class = {}
        counts_list = []

        # Create a new dataframe from all native sample rates and their respective classes
        rate_and_class = {"sample_rate": self.audiodf["sample_rate"].tolist(),
                          "labels": self.metadata["class"].tolist()}
        self.rate_and_class_df = pd.DataFrame(rate_and_class, columns=["sample_rate", "labels"])

        self.dict_label = self.dict_sample_rate_counts_vs_classes("labels", self.rate_and_class_df)

        for i in range(len(self.labels_sorted)):
            self.rate_class["sample_rates [Hz]"] = sorted(self.audiodf["sample_rate"].value_counts().index.tolist())
            counts_list.append(self.all_classes[f"{self.labels_sorted[i]}_counts_sorted_pad"])
            self.rate_class[f"{self.labels_sorted[i]}"] = counts_list[i]

        self.rate_class_df = pd.DataFrame.from_dict(self.rate_class)

        return self.rate_class_df

    def plot_sample_rates_vs_counts(self):

        # self.dict_label = self.dict_sample_rate_counts_vs_classes("label", self.rate_and_class_df)

        fig, axs = plt.subplots(10, figsize=(11, 30), dpi=85)
        width = 0.6

        for i in range(self.num_outputs):

            plt.tight_layout()
            axs[i].bar(self.types - width/2,
                       self.dict_label[f"{self.labels_sorted[i]}_counts_sorted_pad"], width, align="edge")
            axs[i].set_xticks(self.types)
            axs[i].set_xticklabels(self.rates_sorted, rotation=45)
            axs[i].set_title(f"{self.labels_sorted[i]}", fontsize=13)
            plt.sca(axs[i])
            axs[i].set_ylabel("Number of Samples", fontsize=10)

            for j, v in enumerate(self.dict_label[f"{self.labels_sorted[i]}_counts_sorted_pad"]):
                plt.annotate(v, xy=(self.types[j], (
                    self.dict_label[f"{self.labels_sorted[i]}_counts_sorted_pad"][j])),
                        ha='center', va='bottom', xytext=(0, 0), textcoords="offset points")

            plt.tight_layout()

        axs[-1].set_xlabel("Native Sample Rates [Hz]", fontsize=13)
        plt.tight_layout()

    def plot_sample_rates_mispredicted_classes(self, new_df):

        # Get true and false predicted sample rates
        dict_true_pred = self.dict_sample_rate_counts_vs_classes("true_pred", new_df)
        dict_false_pred = self.dict_sample_rate_counts_vs_classes("false_pred", new_df)

        fig, axs = plt.subplots(10, figsize=(11, 30), dpi=85)
        width = 0.3
        types = np.arange(0, self.total_different_sample_rates)

        for i in range(self.num_outputs):

            plt.tight_layout()

            axs[i].bar(types - width/2, dict_true_pred[f"{self.labels_sorted[i]}_counts_sorted_pad"],
                       width, align="edge", label="true")
            axs[i].bar(types + width/2, dict_false_pred[f"{self.labels_sorted[i]}_counts_sorted_pad"],
                       width, align="edge", label="false")

            axs[i].set_xticks(types)
            axs[i].set_xticklabels(self.rates_sorted, rotation=45)
            axs[i].set_title(f"{self.labels_sorted[i]}", fontsize=13)
            plt.sca(axs[i])
            axs[i].set_ylabel("Number of Samples", fontsize=10)
            axs[i].legend()

            for j, v in enumerate(dict_true_pred[f"{self.labels_sorted[i]}_counts_sorted_pad"]):
                plt.annotate(str(v), xy=(types[j], (
                    dict_true_pred[f"{self.labels_sorted[i]}_counts_sorted_pad"][j])),
                        ha='left', va='bottom', xytext=(-20, 3), textcoords="offset points")

            for j, v in enumerate(dict_false_pred[f"{self.labels_sorted[i]}_counts_sorted_pad"]):
                plt.annotate(str(v), xy=(types[j], (
                    dict_false_pred[f"{self.labels_sorted[i]}_counts_sorted_pad"][j])),
                        ha='center', va='bottom', xytext=(20, 3), textcoords="offset points")

            plt.tight_layout()

        axs[-1].set_xlabel("Native Sample Rates [Hz]", fontsize=13)
        plt.tight_layout()
