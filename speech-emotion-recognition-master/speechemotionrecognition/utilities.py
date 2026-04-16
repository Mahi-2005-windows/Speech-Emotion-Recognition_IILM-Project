#"""
#author: harry-7

#This file contains functions to read the data files from the given folders and
#generate Mel Frequency Cepestral Coefficients features for the given audio
#files as training samples.
#"""

# import os
# import sys
# from typing import Tuple

# import numpy as np
# import scipy.io.wavfile as wav
# from speechpy.feature import mfcc

# mean_signal_length = 32000  # Empirically calculated for the given data set


# def get_feature_vector_from_mfcc(file_path: str, flatten: bool,
#                                  mfcc_len: int = 39) -> np.ndarray:
#     """
#     Make feature vector from MFCC for the given wav file.

#     Args:
#         file_path (str): path to the .wav file that needs to be read.
#         flatten (bool) : Boolean indicating whether to flatten mfcc obtained.
#         mfcc_len (int): Number of cepestral co efficients to be consider.

#     Returns:
#         numpy.ndarray: feature vector of the wav file made from mfcc.
#     """
#     fs, signal = wav.read(file_path)
#     s_len = len(signal)
#     # pad the signals to have same size if lesser than required
#     # else slice them
#     if s_len < mean_signal_length:
#         pad_len = mean_signal_length - s_len
#         pad_rem = pad_len % 2
#         pad_len //= 2
#         signal = np.pad(signal, (pad_len, pad_len + pad_rem),
#                         'constant', constant_values=0)
#     else:
#         pad_len = s_len - mean_signal_length
#         pad_len //= 2
#         signal = signal[pad_len:pad_len + mean_signal_length]
#     mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
#     if flatten:
#         # Flatten the data
#         mel_coefficients = np.ravel(mel_coefficients)
#     return mel_coefficients


# def get_data(data_path: str, flatten: bool = True, mfcc_len: int = 39,
#              class_labels: Tuple = ("Neutral", "Angry", "Happy", "Sad")) -> \
#         Tuple[np.ndarray, np.ndarray]:
#     """Extract data for training and testing.

#     1. Iterate through all the folders.
#     2. Read the audio files in each folder.
#     3. Extract Mel frequency cepestral coefficients for each file.
#     4. Generate feature vector for the audio files as required.

#     Args:
#         data_path (str): path to the data set folder
#         flatten (bool): Boolean specifying whether to flatten the data or not.
#         mfcc_len (int): Number of mfcc features to take for each frame.
#         class_labels (tuple): class labels that we care about.

#     Returns:
#         Tuple[numpy.ndarray, numpy.ndarray]: Two numpy arrays, one with mfcc and
#         other with labels.
#     """
#     data = []
#     labels = []
#     names = []
#     cur_dir = os.getcwd()
#     sys.stderr.write('curdir: %s\n' % cur_dir)
#     os.chdir(data_path)
#     for i, directory in enumerate(class_labels):
#         sys.stderr.write("started reading folder %s\n" % directory)
#         os.chdir(directory)
#         for filename in os.listdir('.'):
#             filepath = os.getcwd() + '/' + filename
#             feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
#                                                           mfcc_len=mfcc_len,
#                                                           flatten=flatten)
#             data.append(feature_vector)
#             labels.append(i)
#             names.append(filename)
#         sys.stderr.write("ended reading folder %s\n" % directory)
#         os.chdir('..')
#     os.chdir(cur_dir)
#     return np.array(data), np.array(labels)


#NEW UPDATED CODE 
import os
import sys
from typing import Tuple

import numpy as np
import scipy.io.wavfile as wav
import librosa

mean_signal_length = 32000


def extract_features(file_path: str):
    signal, sr = librosa.load(file_path, sr=None)

    # Padding / Trimming
    if len(signal) < mean_signal_length:
        pad_len = mean_signal_length - len(signal)
        signal = np.pad(signal, (0, pad_len), mode='constant')
    else:
        signal = signal[:mean_signal_length]

    # 🔥 Feature Extraction
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T, axis=0)

    # Combine all features
    return np.hstack([mfcc, chroma, mel])


def get_data(data_path: str,
             class_labels: Tuple = ("Neutral", "Angry", "Happy", "Sad")):

    data = []
    labels = []
    cur_dir = os.getcwd()

    os.chdir(data_path)

    for i, directory in enumerate(class_labels):
        print(f"Reading folder: {directory}")
        os.chdir(directory)

        for filename in os.listdir('.'):
            filepath = os.path.join(os.getcwd(), filename)
            features = extract_features(filepath)
            data.append(features)
            labels.append(i)

        os.chdir('..')

    os.chdir(cur_dir)

    X = np.array(data)
    y = np.array(labels)

    # 🔥 Normalization (important for ML)
    X = (X - np.mean(X)) / np.std(X)

    return X, y