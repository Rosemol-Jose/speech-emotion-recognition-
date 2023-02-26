This project is trying to plot mfcc and spectogram of speech and derived signals of speech-emotion-recognition done using cnn-lstm networks and find its accuracy


Dataset used is 4 classes from the EMODB Berlin Emotional Speech Database 

step 1: visualising an input audio sample which is done using input.py

step 2: its spectrogram in spectrogrm.py

step 3: feature extraction using mfcc for each class:4 classes used here angry, happy, sad and neutral

step 4: Execute the main program mainn.py which calls in subprograms  _init_.py, utilities.py, dnn.py in ser to train samples using  cnn and lstm

Accuracy of about 86% is obtained
