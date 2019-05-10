"""

author: harry-7

This file contains functions to read the data files from the given folders and generate the data interms of features
"""
import numpy as np
import math
import scipy.io.wavfile as wav
import os
import speechpy
from sklearn.model_selection import train_test_split

class_labels = ["Neutral","Sad","Angry","Happy",]

mslen = 32000  # Empirically calculated for the given dataset
dataset_path='dataset'

def read_wav(filename):
    
    return wav.read(filename)


def get_data(dataset_path, flatten=True, mfcc_len=39):
   
    data = []
    labels = []
    max_fs = 0
    s = 0
    cnt = 0
    cur_dir = os.getcwd()
    print('curdir', cur_dir)
    os.chdir(dataset_path)
    for i, directory in enumerate(class_labels):
        print( "started reading folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            max_fs = max(max_fs, fs)
            #print("fs=",max_fs)
            s_len = len(signal)
            # pad the signals to have same size if lesser than required
            # else slice them
            if s_len < mslen:
                pad_len = mslen - s_len
                pad_rem = pad_len % 2
                pad_len /= 2
                signal = np.pad(signal, (math.ceil(pad_len),math.ceil(pad_len) + math.ceil(pad_rem)), 'constant', constant_values=0)
                
            else:
                pad_len = s_len - mslen
                pad_len /= 2
                signal = signal[math.ceil(pad_len):math.ceil(pad_len) + mslen]
            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)
            if flatten:
                # Flatten the data
                mfcc = mfcc.flatten()
            data.append(mfcc)
            labels.append(i)
            cnt += 1
        print( "ended reading folder", directory)
        #print(mfcc)
        os.chdir('..')
    os.chdir(cur_dir)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
   # print(x_train, x_test, y_train, y_test)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
get_data(dataset_path, flatten=True, mfcc_len=39)
