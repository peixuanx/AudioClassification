#encoding: utf-8
import os
import librosa
import numpy as np
import glob
from audioClassification import Classes

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)

    # sftf
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(sub_dirs, mode, file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for sub_dir in sub_dirs:
        print("sub_dir: %s" % (sub_dir))
        for fn in glob.glob(os.path.join(sub_dir, file_ext)):
            name = fn.split('/')[-1].split('_')[0]
            print("name: %s" % (name))
            print("extract file: %s" % (fn))
            try:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            if mode=="train":
                labels = np.append(labels, Classes[name])
    return np.array(features), np.array(labels, dtype = np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
