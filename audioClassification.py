import pickle
from features import *
import numpy as np
import dataset
from dataset import TrainData
from dataset import TestData
import tensorflow as tf
import csv

train_data_pickle = 'train.pkl'
test_data_pickle = 'test.pkl'

Classes = {"BassClarinet": 0, "BassTrombone": 1, "BbClarinet": 2, "Cello": 3, "EbClarinet": 4, "Marimba": 5, "TenorTrombone": 6, "Viola": 7, "Violin": 8, "Xylophone": 9}

def traintest():
    if not os.path.isfile(train_data_pickle):
        # training data
        train_features, train_labels = get_features(['train_data'], "train")
        traindata = TrainData(train_features, train_labels)
        with open(train_data_pickle, mode='wb') as f:
            pickle.dump(traindata, f)
    else:
        print("loading: %s" % (train_data_pickle))
        with open(train_data_pickle, mode='rb') as f:
            traindata = pickle.load(f)
            train_features = traindata.train_inputs
            train_labels = traindata.train_targets
    
    if not os.path.isfile(test_data_pickle):
        # testing data
        test_features, _ = get_features(['test_data'], "test")
        testdata = TestData(test_features)
        with open(test_data_pickle, mode='wb') as f:
            pickle.dump(testdata, f)
    else:
        print("loading: %s" % (test_data_pickle))
        with open(test_data_pickle, mode='rb') as f:
            testdata = pickle.load(f)
            test_features = testdata.test_inputs
    

    train_labels = one_hot_encode(train_labels)

    n_dim = train_features.shape[1]
    print("input dim: %s" % (n_dim))
    
     # random train and test sets.
    '''
    train_test_split = np.random.rand(len(train_features)) < 0.80
    Xtr = train_features[train_test_split]
    Ytr = train_labels[train_test_split]
    Xte = train_features[~train_test_split]
    Yte = train_labels[~train_test_split]
    '''
    Xtr = train_features
    Ytr = train_labels
    Xte = test_features

    knn(n_dim, Xtr, Ytr, Xte)    
    #regression(n_dim, Xtr, Ytr, Xte, Yte)    
    

def knn(n_dim, Xtr, Ytr, Xte, Yte=None):
    # create placeholder
    xtr = tf.placeholder(tf.float32, [None, n_dim])
    xte = tf.placeholder(tf.float32, [n_dim])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)
    
    accuracy = 0.

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # loop over test data
        for i in range(len(Xte)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            prediction = np.argmax(Ytr[nn_index])
            
            generate_csv(i, prediction)
            # Get nearest neighbor class label and compare it to its true label
            #print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
             #   "True Class:", np.argmax(Yte[i]))
            # Calculate accuracy
            #if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
             #   accuracy += 1./len(Xte)
        #print("Done!")
        #print("Accuracy:", accuracy)   

def generate_csv(index, prediction):
    with open('submission_peixuan.csv', 'a') as f:
        writer = csv.writer(f)
        filename = "Unknown_"+str(index+1).zfill(3)+".wav"
        for instrument, label in Classes.items():
            if prediction==label:
                writer.writerow([filename, instrument])

def get_features(sub_dirs, mode):
    try:
        features, labels = parse_audio_files(sub_dirs, mode)
    except Exception as e:
        print("[Error] parse error. %s" % e)
    return features, labels

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

if __name__ == '__main__':
    '''
    waves, names = dataset.get_files("train_data")
    for wave in waves:
        print("="*10)
        print("file: %s" % wave)

    raw_waves = raw_sounds = load_sound_files(waves)
    '''
    traintest()

