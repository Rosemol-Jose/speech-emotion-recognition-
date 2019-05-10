#mfcc featres plot
#need to include print mfcc features from featu

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
frequency_sampling, audio_signal = wavfile.read(r"C:\Users\Rose\Desktop\success\speech-emotion-recognition-master\dataset\Angry\03a01Wa.wav")
audio_signal = audio_signal[:15000]
features_mfcc = mfcc(audio_signal, frequency_sampling)
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')
filterbank_features = logfbank(audio_signal, frequency_sampling)
print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])
filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()
