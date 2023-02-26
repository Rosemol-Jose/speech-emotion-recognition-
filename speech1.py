import numpy
import librosa
import matplotlib.pyplot as plt
DATA_DIR=r"C:\Users\Rose\Desktop\recordings\0_jackson_0.wav"
wav, sr = librosa.load(DATA_DIR)
print ('sr:', sr)
print ('wav shape:', wav.shape)
print ('length:', sr/wav.shape[0], 'secs')
plt.plot(wav)
#to zoom
plt.plot(wav)
plt.show()

