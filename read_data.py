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
    range_fft = range_fft[:NS//2, :, :]

    # FFT across slow time
    doppler_signal = range_fft*win_d
    doppler_fft = np.fft.fft(doppler_signal, NC, axis = 1)
    doppler_fft = np.fft.fftshift(doppler_fft, axes = 1) # shifting the zero frequency component to the center

    # average over virtual antennas
    rdm = np.mean(np.abs(doppler_fft), axis = 2)
    print(f"range doppler map shape: {rdm.shape}")
    return rdm

def draw_plots(rdm: np.ndarray,
               ram: np.ndarray):
    fig, axes = plt.subplots(1,2, figsize = (12,5))

    # plot for rdm
    ax = axes[0]
    n_range, n_doppler = rdm.shape
    r_ax = np.arange(n_range)*RANGE_RES
    d_ax = (np.arange(n_doppler) - n_doppler//2)*VEL_RES
    rdm_db = 20*np.log10(rdm + 1e-9)
    v_max_rdm = rdm_db.max()

    im1 = ax.imshow(rdm_db, 
                   extent = [d_ax[0], d_ax[-1], r_ax[0], r_ax[-1]],
                   aspect = "auto",
                   origin = "lower",
                   vmin = v_max_rdm - 80)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Range-Doppler Map")

    # plot for ram
    ax = axes[1]
    n_range, n_virtual = ram.shape
    k = (np.arange(n_virtual) - n_virtual//2)
    print(k)
    sin_theta = 2*k/n_virtual
    v_ax = np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))
    print(v_ax)
    ram_db = 20*np.log10(ram + 1e-9)
    v_max_ram = ram_db.max()

    im2 = ax.imshow(ram_db,
                    extent = [v_ax[0], v_ax[-1], r_ax[0], r_ax[-1]],
                    aspect = "auto",
                    origin = "lower",
                    vmin = v_max_ram - 80)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Range(m)")
    ax.set_title("Range Angle Map")

    plt.show()

def compute_ram(virtual: np.ndarray) -> np.ndarray:
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
    

def main():
    i = np.random.randint(0,len(radar_data_paths))
    print(f"Id for this frame is {radar_data_paths[i].stem}")

    # loading the matlab file to get data
    radar_data = load_frame(radar_data_paths[155])
    virtual = tdm_demux(radar_data)
    print(f"Data Shape: {radar_data.shape}")
    print(f"Virtual Data Shape: {virtual.shape}")

    # computing range doppler and range angle map
    rdm = compute_rdm(virtual)
    ram = compute_ram(virtual)

    # plotting the maps
    draw_plots(rdm, ram)
    return 


if __name__ == "__main__":
    main()