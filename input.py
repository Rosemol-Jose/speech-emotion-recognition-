#visualise ur input signal

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
frequency_sampling, audio_signal = wavfile.read('03a02Wb.wav')
print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration:', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')
audio_signal1 = audio_signal / np.power(2, 15)
audio_signal = audio_signal [:100]
time_axis = 1000 * np.arange(0, len(audio_signal), 1) / float(frequency_sampling)
plt.plot(time_axis,audio_signal, color='blue')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()
