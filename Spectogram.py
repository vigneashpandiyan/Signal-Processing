import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as π
from scipy.signal import spectrogram
import pywt

f_s = 200              # Sampling rate = number of measurements per second in [Hz]
t   = np.arange(-10,10, 1 / f_s) # Time between [-10s,10s].
T1  = np.tanh(t)/2  + 1.0 # Period in [s]
T2  = 0.125               # Period in [s]
f1  = 1 / T1              # Frequency in [Hz]
f2  = 1 / T2              # Frequency in [Hz] 

N = len(t)
x = 13 * np.sin(2 * π * f1 * t) + 42 * np.sin(2 * π * f2 * t)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex = True, figsize = (10,8))

# Signal
ax1.plot(t, x)
ax1.grid(True)
ax1.set_ylabel("$x(t)$")
ax1.set_title("Signal x(t)")

# Frequency change
ax2.plot(t, f1)
ax2.grid(True)
ax2.set_ylabel("$f_1$ in [Hz]")
ax2.set_title("Change of frequency $f_1(t)$")

# Moving fourier transform, i.e. spectrogram
Δt = 4 # window length in [s]
Nw = np.int(2**np.round(np.log2(Δt * f_s))) # Number of datapoints within window
f, t_, Sxx = spectrogram(x, f_s, window='hanning', nperseg=Nw, noverlap = Nw - 100, detrend=False, scaling='spectrum')
Δf  =  f[1] - f[0]
Δt_ = t_[1] - t_[0]
t2  = t_ + t[0] - Δt_

im = ax3.pcolormesh(t2, f - Δf/2, np.sqrt(2*Sxx), cmap = "inferno_r")#, alpha = 0.5)
ax3.grid(True)
ax3.set_ylabel("Frequency in [Hz]")
ax3.set_ylim(0, 10)
ax3.set_xlim(np.min(t2),np.max(t2))
ax3.set_title("Spectrogram using FFT and Hanning window")


# Wavelet transform, i.e. scaleogram
cwtmatr, freqs = pywt.cwt(x, np.arange(1, 512), "gaus1", sampling_period = 1 / f_s)
im2 = ax4.pcolormesh(t, freqs, cwtmatr, vmin=0, cmap = "inferno" )  
ax4.set_ylim(0,10)
ax4.set_ylabel("Frequency in [Hz]")
ax4.set_xlabel("Time in [s]")
ax4.set_title("Scaleogram using wavelet GAUS1")

# plt.savefig("./fourplot.pdf")

plt.show()