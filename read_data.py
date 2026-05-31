import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import windows
from scipy.io import loadmat
from pathlib import Path

# setting constants for hardware Texas Instruments AWR1843

C = 3e8 # speed of light (m/s)
FC = 77e9 # carrier Frequency (Hz)
BANDWIDTH = 0.67e9 # (Hz)
CHIRP_TIME = 60e-6 # single chirp per antenna (sec)
LAMDA = C/FC # wavelength (m)
D_ANTENNA  = LAMDA / 2 # half-wavelength spacing between virtual antennas 

N_SAMPLES = 128 # fast time samples per chirp
N_CHIRPS = 255  # slow time chirps per frame
N_RX = 4 # physical recieve antenna 
N_TX = 2 # physical transmit antenna
N_VIRTUAL = N_RX*N_TX # 8 virtual antennas after TDM MIMO demux

# physical quantities

RANGE_RES = C/(2*BANDWIDTH)
VEL_RES = LAMDA/(2*N_CHIRPS*CHIRP_TIME)
MAX_RANGE = N_SAMPLES*RANGE_RES
MAX_VEL = VEL_RES*N_CHIRPS/2

#TOTAL_FRAMES = 19754
# paths

dataset = Path.cwd()/"Automotive"
radar_data_paths = list(dataset.rglob("*.mat"))

# printing this physical quatities

print(f"Range Resolution: {RANGE_RES*100:.2f} cm")
print(f"Maximum Range: {MAX_RANGE:.2f} m")
print(f"Velocity Resolution: {VEL_RES:.2f} m/s")
print(f"Maximum Velocity: {MAX_VEL:.2f} m/s")

# load and reshape
# loading a fram from its path and returning the data in the form of ndarray
def load_frame(path: str) -> np.ndarray:
    # reading the matlab file
    radar_data = loadmat(path)
    # extracting the adc data from the dictionary
    radar_adc_data = radar_data["adcData"]
    return radar_adc_data

# demux the tdm ( we have MIMO), hence from 4x2 antennas we get 8 pairs
def tdm_demux(adc: np.ndarray) -> np.ndarray:
    virtual = np.transpose(adc,(0,1,3,2))
    virtual = virtual.reshape(virtual.shape[0], virtual.shape[1], -1)
    return virtual

# computing the range doppler mapping from the virtual radar data
def compute_rdm(virtual: np.ndarray) -> np.ndarray:
    NS, NC, V = virtual.shape
    # generating blackman window to add stability in corners of the signal
    win_r = windows.blackman(NS).reshape(-1, 1, 1).astype(np.float32)
    win_d = windows.blackman(NC).reshape(1, -1, 1).astype(np.float32)

    # FFT across fast time
    range_signal = virtual*win_r
    range_fft = np.fft.fft(range_signal, NS, axis = 0)

    # FFT across slow time
    doppler_signal = range_fft*win_d
    doppler_fft = np.fft.fft(doppler_signal, NC, axis = 1)
    doppler_fft = np.fft.fftshift(doppler_fft, axes = 1) # shifting the zero frequency component to the center

    # average over virtual antennas
    rdm = np.mean(np.abs(doppler_fft), axis = 2)
    print(f"range doppler map shape: {rdm.shape}")
    return rdm

def draw_rdm_heatmap(rdm: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots()

    n_range, n_doppler = rdm.shape
    r_ax = (np.arange(n_range)*RANGE_RES)
    v_ax = (np.arange(n_doppler) - n_doppler//2)*VEL_RES
    rdm_db = 20*np.log10(rdm + 1e-9)
    im = ax.imshow(rdm_db, 
                   extent = [v_ax[0], v_ax[-1], r_ax[0], r_ax[-1]],
                   aspect = "auto",
                   origin = "lower")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Range-Doppler Map")
    plt.show()
    



def main():
    i = np.random.randint(0,len(radar_data_paths))
    print(f"Id for this frame is {radar_data_paths[i].stem}")
    # loading the matlab file to get data
    radar_data = load_frame(radar_data_paths[i])
    virtual = tdm_demux(radar_data)
    print(f"Data Shape: {radar_data.shape}")
    print(f"Virtual Data Shape: {virtual.shape}")
    rdm = compute_rdm(virtual)
    draw_rdm_heatmap(rdm)
    return 


if __name__ == "__main__":
    main()