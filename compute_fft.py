import numpy as np
from scipy.signal import windows

class ComputeFFT():
    def __init__(self, virtual):
        self.virtual = virtual
    
    # computing the range doppler mapping from the virtual radar data
    def compute_rdm(self, virtual: np.ndarray) -> np.ndarray:
        NS, NC, V = virtual.shape
        # generating blackman window to add stability in corners of the signal
        win_r = windows.blackman(NS).reshape(-1, 1, 1).astype(np.float32)
        win_d = windows.blackman(NC).reshape(1, -1, 1).astype(np.float32)

        # FFT across fast time
        range_signal = virtual*win_r
        range_fft = np.fft.fft(range_signal, NS, axis = 0)
        range_fft = range_fft[:NS//2, :, :]

        # FFT across slow time
        doppler_signal = range_fft*win_d
        doppler_fft = np.fft.fft(doppler_signal, NC, axis = 1)
        doppler_fft = np.fft.fftshift(doppler_fft, axes = 1) # shifting the zero frequency component to the center

        # average over virtual antennas
        rdm = np.mean(np.abs(doppler_fft), axis = 2)
        print(f"range doppler map shape: {rdm.shape}")
        return rdm

    # computing the range angle map from the virtual radar data
    def compute_ram(self, virtual: np.ndarray) -> np.ndarray:
        NS, NC, V = virtual.shape

        # generating blackman window for stability and to prevent leakage
        win_r = windows.blackman(NS).reshape(-1, 1, 1).astype(np.float32)
        win_a = windows.hann(V).reshape(1, -1).astype(np.float32)

        # fft across fast time
        range_signal = virtual*win_r
        range_fft = np.fft.fft(range_signal, NS, axis = 0)
        range_fft = range_fft[:NS//2, :, :]

        # averaing over doppler axis
        range_fft = np.mean(np.abs(range_fft), axis = 1)

        # fft across virtual antennas
        range_fft = range_fft*win_a
        ram = np.fft.fft(range_fft, n = 64 ,axis = 1)
        ram = np.fft.fftshift(np.abs(ram), axes = 1)
        print(f"range angle map shape: {ram.shape}")
        return ram