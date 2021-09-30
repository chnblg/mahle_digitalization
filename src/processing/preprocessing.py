import librosa
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from src.helpers.wavfilehelper import WavFileHelper
DEFAULT_PATH = '/datasets/UrbanSound8K/'
DEFAULT_SAMPLE_RATE = 22050


def load_metadata(path=DEFAULT_PATH):

    return pd.read_csv(path + 'metadata/UrbanSound8K.csv')


def load_metadata_apm(path='/home/iotmaster/samples/', dataset_metadata='10-percent-steps-1-only/10-percent-steps-1-only.csv'):

    return pd.read_csv(path + dataset_metadata, sep=';')


# Extend metadata dataframe by adding a "sample_rate" column
def extend_metadata():

    metadata = load_metadata()
    audiodf = create_audiodf()
    metadata["sample_rate"] = audiodf["sample_rate"].tolist()

    return metadata


def load_dataset(path=DEFAULT_PATH, sample_rate=DEFAULT_SAMPLE_RATE, num_samples=None):

    metadata = extend_metadata()

    dataset_list = []

    # iterrows() is a generator which yields both index (i) and row (sample)
    for i, sample in metadata.iterrows():
        file_name = os.path.join(path, 'audio', 'fold' + str(sample.fold), str(sample.slice_file_name))
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=sample_rate)
        dataset_list.append((audio, sample['class'], sample['fold']))

        if i % (len(metadata) // 10) == 0:
            print(f'{i / len(metadata):.2f} of data loaded')
        if (num_samples is not None):
            if i == num_samples - 1:
                break

    data = pd.DataFrame(dataset_list, columns=['feature', 'label', 'fold'])
    data["sample_rate"] = metadata["sample_rate"].tolist()

    return data


# Extract all samples with sample rate > 44 kHz and load
def load_high_freq_dataset(path=DEFAULT_PATH, sample_rate=DEFAULT_SAMPLE_RATE, num_samples=None):

    metadata = extend_metadata()
    high_freq_dataset_list = []

    # Save indices of all samples with a sample rate value > 44 kHz into a list
    high_freq_indices = metadata.index[metadata["sample_rate"] > 44000].tolist()

    # Create a second dataframe from metadata based on indices of high frequency samples
    high_freq_metadata = metadata.iloc[high_freq_indices, :].copy()

    for i, sample in high_freq_metadata.iterrows():
        file_name = os.path.join(path, 'audio', 'fold' + str(sample.fold), str(sample.slice_file_name))
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=sample_rate)
        high_freq_dataset_list.append((audio, sample['class'], sample['fold']))

        if i % (len(high_freq_metadata) // 10) == 0:
            print(f'{i / len(high_freq_metadata):.2f} of data loaded')
        if (num_samples is not None):
            if i == num_samples - 1:
                break

    high_freq_data = pd.DataFrame(high_freq_dataset_list, columns=['feature', 'label', 'fold'])
    high_freq_data["native_sample_rate"] = metadata["sample_rate"].tolist()

    return high_freq_data

def load_dataset_apm(path='/home/iotmaster/samples/', dataset='10-percent-steps-1-only', sample_rate=44100, num_samples=None):

    metadata = load_metadata_apm(dataset_metadata=dataset + '/' + dataset + '.csv')

    dataset_list = []

    # iterrows() is a generator which yields both index (i) and row (sample)
    for i, sample in metadata.iterrows():
        file_name = os.path.join(path, dataset + '/' + str(sample['filename']))
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=sample_rate)
        dataset_list.append((audio, sample['hardware']))

        if i % (len(metadata) // 10) == 0:
            print(f'{i / len(metadata):.2f} of data loaded')
        if (num_samples is not None):
            if i == num_samples - 1:
                break

    data = pd.DataFrame(dataset_list, columns=['feature', 'label'])

    return data


def calculate_mean_mfcc(audio_array, sample_rate=DEFAULT_SAMPLE_RATE):

    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
    mfccs = mfccs[:, :173]  # limit clips to ~5 seconds
    return mfccs

def calculate_mean_mfcc_apm(audio_array, sample_rate=44100):

    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
    mfccs = mfccs[:, 173:346]  # limit clips to ~5 seconds
    return mfccs

# def calculate_melspectrogram(audio_array, sample_rate=DEFAULT_SAMPLE_RATE):
#     spectrogram = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_fft=2048)
#     # mfccs = mfccs[:, :173] # limit clips to ~5 seconds
#     return mfccs


def calculate_spectrogram(audio_array, sample_rate=DEFAULT_SAMPLE_RATE):

    frequencies, times, spectrogram = signal.spectrogram(audio_array, sample_rate)
    return spectrogram


# Some of the spectrogrammes doesn´t have exactly the shape (40, 173), their time steps are shorter (< 173).
# Only select the spectrogrammes with exactly the shape (40, 173). Drop others.
# Decreases the size of df.
def filter_mfccs(df):

    drop_index = []
    for i in range(len(df)):
        mfc_array = np.array(df['feature'].iloc[i])
        if mfc_array.shape[1] == 173:
            df['feature'].iloc[i] = mfc_array
        else:
            drop_index.append(i)

    df = df.drop(drop_index, axis=0)
    df.reset_index(drop=True, inplace=True)
    return df


# Some of the spectrogrammes doesn´t have exactly the shape (40, 173), their time steps are shorter (< 173).
# Pad the spectrogrammes with zeros the have exactly the shape (40, 173)
def pad_mfccs(df, target_sample_length: int = 173):
    for i in range(len(df)):
        mfc_array = np.array(df['feature'].iloc[i])
        if mfc_array.shape[1] < target_sample_length:
            padding_width = target_sample_length - mfc_array.shape[1]
            mfc_array = np.pad(mfc_array, [(0, 0), (0, padding_width)], mode='constant', constant_values=(0))
        df['feature'].iloc[i] = mfc_array
    return df


def create_training_data_RNN(data_df):

    X_array = np.array(data_df.feature.tolist())
    y_array = np.array(data_df.label.tolist())

    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y_array))

    X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.2, random_state=42)

    # num_outputs = data_df['label'].unique().shape[0]

    X_train_t = X_train.transpose(0, 2, 1)
    X_test_t = X_test.transpose(0, 2, 1)

    return X_train_t, X_test_t, y_train, y_test


# use all data to make predictions with the best saved model
def create_prediction_data(data_df):

    X_array = np.array(data_df.feature.tolist())
    y_array = np.array(data_df.label.tolist())

    le = LabelEncoder()
    y_true = to_categorical(le.fit_transform(y_array))  # true labels

    X_t = X_array.transpose(0, 2, 1)

    return X_t, y_true


# Extract number of audio channels, sample rate and bit-depth into a Pandas dataframe
def create_audiodf(path=DEFAULT_PATH):

    # from helpers.wavfilehelper import WavFileHelper
    wavfilehelper = WavFileHelper()

    audiodata = []
    # path = '/datasets/UrbanSound8K/audio/'
    fullpath = path + "audio/"
    metadata = load_metadata(path)

    for index, row in metadata.iterrows():

        file_name = os.path.join(
            os.path.abspath(fullpath), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
        data = wavfilehelper.read_file_properties(file_name)
        audiodata.append(data)

    # Convert into a Pandas dataframe
    audiodf = pd.DataFrame(audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])

    return audiodf


# Function to add "true predictions" and "false predictions" columns to features (spectrogrammes)
def add_sample_rate_to_features(y_labels, y_pred, df):

    # Sort unique class names based on ClassID (0,1,2 ... 9)
    metadata = extend_metadata()
    classID = metadata.classID.unique().tolist()
    classNames = metadata["class"].unique().tolist()
    labels_sorted = [x for _, x in sorted(zip(classID, classNames))]

    true_predictions = []
    false_predictions = []
    new_df = df.copy()

    for actual, prediction in zip(y_labels, y_pred):
        if actual != prediction:
            false_predictions.append(prediction)
            true_predictions.append(None)
        else:
            true_predictions.append(prediction)
            false_predictions.append(None)

    # add 2 columns for predictions (classID)
    new_df["true_pred"] = true_predictions
    new_df["false_pred"] = false_predictions

    # replace class ID´s of predictions with label names
    new_df["true_pred"].replace({set_ID: f"{label}" for set_ID, label in enumerate(labels_sorted)}, inplace=True)
    new_df["false_pred"].replace({set_ID: f"{label}" for set_ID, label in enumerate(labels_sorted)}, inplace=True)

    return new_df


def normalize_mfccs(data_df):
    matrix = [x for x in data_df['feature']]
    dataframe_length = len(matrix)
    dataframe_shape_0 = matrix[0].shape[0]
    dataframe_shape_1 = matrix[0].shape[1]
    matrix_flat = np.reshape(matrix, (dataframe_length * dataframe_shape_0 * dataframe_shape_1, 1))
    transformer = RobustScaler().fit(matrix_flat)
    matrix_flat = transformer.transform(matrix_flat)
    matrix_transform_back = np.reshape(matrix_flat, (dataframe_length, dataframe_shape_0, dataframe_shape_1))
    for i in range(len(matrix_transform_back)):
        data_df.at[i, 'feature'] = matrix_transform_back[i]

    return data_df
