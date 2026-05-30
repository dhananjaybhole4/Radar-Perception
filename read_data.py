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

def tdm_demux(adc: np.ndarray) -> np.ndarray:
    virtual = np.transpose(adc,(0,1,3,2))
    virtual = virtual.reshape(virtual.shape[0], virtual.shape[1], -1)
    return virtual

def main():
    i = np.random.randint(0,len(radar_data_paths))
    print(f"Id for this frame is {radar_data_paths[i].stem}")
    # loading the matlab file to get data
    radar_data = load_frame(radar_data_paths[i])
    virtual = tdm_demux(radar_data)
    print(f"Array Type: {type(radar_data)}")
    print(f"Data Type: {radar_data.dtype}")
    print(f"Data Shape: {radar_data.shape}")
    print(f"Virtual Data Shape: {virtual.shape}")
    return 


if __name__ == "__main__":
    main()