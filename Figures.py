import os
import matplotlib.pyplot as plt
import pandas as pd
from nptdms import TdmsFile
import numpy as np
import scipy.signal as signal
import pywt
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.signal import welch, periodogram
from numpy.fft import fftshift, fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from librosa.display import specshow
import librosa
import numpy as np
import scipy.signal
import numpy as np
import scipy.signal
from pandas import DataFrame
import numpy as np
import scipy.signal
from pandas import DataFrame
import numpy as np
import scipy.signal
from pandas import DataFrame

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
tdms_file = TdmsFile("Test3.tdms")
sample_rate = 25600
grp1_data = tdms_file.object('Untitled').as_dataframe()

Z = grp1_data['Z']
Y = grp1_data['Y']
X = grp1_data['X']

Feature_vectors = np.empty((0, 118))
lower_bound = 140000
upper_bound = 510000

for x in range(0, 14):
    signal_window = (X[lower_bound:upper_bound])


win = 4 * sample_rate
freqs, psd = periodogram(signal_window, sample_rate)

x = psd
print('Detect peaks with minimum height and distance filters.')
indexes, value = scipy.signal.find_peaks(psd, height=0.0005, distance=50)
a = value['peak_heights']
sorted_list = sorted(a, reverse=True)
b = sorted_list[0:3]
print('Peaks are: %s' % (indexes))
markers_off = indexes.tolist()
plt.plot(x, '-gD', markevery=markers_off, linewidth=2.5, markerfacecolor='yellow', markeredgewidth=1.5)
plt.legend
plt.show()

# amplitude correction factor
corr = 0.5
# calculate the psd with welch
sample_freq, power = welch(X, fs=sample_rate, window="hann", nperseg=256, noverlap=18, scaling='spectrum')
# fftshift the output
sample_freq = fftshift(sample_freq)
power = fftshift(power) / corr
# check that the power sum is right
# print (sum(power))
plt.figure(figsize=(9.84, 3.94))
plt.plot(sample_freq, power)
plt.xlabel("Frequency (kHz)", fontsize=18)
plt.ylabel("Relative power (dB)", fontsize=18)
plt.show()

x = psd
print('Detect peaks with minimum height and distance filters.')
indexes, value = scipy.signal.find_peaks(power, height=0.005, distance=500)
print('Peaks are: %s' % (indexes))
markers_off = indexes.tolist()
plt.plot(x, '-gD', markevery=markers_off, linewidth=2.5, markerfacecolor='yellow', markeredgewidth=1.5)
plt.legend
plt.show()


win = 4 * sample_rate
freqs, psd = signal.welch(X, sample_rate, nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, freqs.max()])
sns.despine()

x = psd
print('Detect peaks with minimum height and distance filters.')
indexes, value = scipy.signal.find_peaks(x, height=0.005, distance=50)
print('Peaks are: %s' % (indexes))
markers_off = indexes.tolist()
plt.plot(x, '-gD', markevery=markers_off, linewidth=2.5, markerfacecolor='yellow', markeredgewidth=1.5)
plt.legend
plt.show()

X1 = np.array(X)
D = librosa.amplitude_to_db(librosa.stft(X1, hop_length=2048), ref=np.max)
librosa.display.specshow(D, y_axis='linear', sr=sample_rate, hop_length=128, bins_per_octave=24)
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()

# %%
cmap = plt.get_cmap('PiYG')
plt.figure(figsize=(9.84, 3.94))
plt.specgram(X, NFFT=4096, noverlap=2048, Fs=1, cmap=cmap)
plt.xlabel("Frequency (kHz)", fontsize=18)
plt.ylabel("Relative power (dB)", fontsize=18)
plt.show()
# %%
waveletname = 'sym5'
fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))
for ii in range(5):
    (data, coeff_d) = pywt.dwt(X1, waveletname)
    axarr[ii, 0].plot(data, 'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])
plt.tight_layout()
plt.show()

f, t, Zxx = signal.stft(X1, fs=sample_rate)
cmap = plt.get_cmap('PiYG')
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.pcolormesh(t, f, np.abs(Zxx), cmap=cmap)
fig.colorbar(im)
ax.set_title("STFT Plot ", fontsize=24)
ax.set_xlabel("Time", fontsize=24)
ax.set_ylabel("Frequency", fontsize=24)
plt.tick_params(labelsize=20)
plt.show()
fig.savefig('Basic.png', dpi=300)
a = np.abs(Zxx)
