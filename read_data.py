import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import windows
from scipy.io import loadmat
from pathlib import Path
from PIL import Image

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
camera_data_paths = list(dataset.rglob("*.jpg"))

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

# computing the range angle map from the virtual radar data
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
    
def draw_plots(rdm: np.ndarray,
               ram: np.ndarray,
               pointcloud: np.ndarray,
               i: int):
    fig, axes = plt.subplots(2,2, figsize = (12,10))

    # plot for rdm
    ax = axes[0,0]
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
    ax = axes[0,1]
    n_range, n_virtual = ram.shape
    k = (np.arange(n_virtual) - n_virtual//2)
    sin_theta = 2*k/n_virtual
    v_ax = np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))
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

    # plot the pointcloud
    ax = axes[1,0]
    if len(pointcloud) > 0:
        pointcloud = np.array(pointcloud)
        x_range = pointcloud[:, 0]
        y_range = pointcloud[:, 1]
        mag = pointcloud[:, 3]
        im3 = ax.scatter(x_range, y_range, c = mag, cmap = "RdBu_r")
        plt.colorbar(im3, ax=ax, label='Mag')
    ax.set_xlabel("X lateral (m)")
    ax.set_ylabel("Y_forward (m)")
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)


    # plot the image
    ax = axes[1,1]
    im4 = ax.imshow(Image.open(camera_data_paths[i]))

    plt.show()

def ca_cfar_2d(rdm: np.ndarray,
               guard: int = 4,
               train: int = 8,
               pfa: float = 1e-2,
               static_bins: int = 2,
               remove_static: bool = True) -> list[tuple[int,int]]:
    """Function to apply constant false alarm rate to get the detections from range doppler map

    Args:
        rdm (np.ndarray): range doppler map_
        guard (int, optional): cells to ignore around cell under test (CUT). Defaults to 4.
        train (int, optional): cells to consider for averaging. Defaults to 8.
        pfa (float, optional): probability of false alarm. Defaults to 1e-3.

    Returns:
        list[tuple[int,int]]: list containing all the detected pair of range and velocity
    """
    n_range, n_doppler = rdm.shape
    alpha = train*(pfa**(-1.0/train) - 1)
    pad = guard + train
    detection = []
    zero_bin = n_doppler//2

    for r in range(pad, n_range - pad):
        for d in range(pad, n_doppler - pad):
            if (remove_static and abs(zero_bin - d) <= static_bins):
                continue
            region = rdm[r-pad : r + pad + 1, d - pad : d + pad + 1].copy()
            guard_mask = np.zeros_like(region, dtype = bool)
            g = guard
            guard_mask[pad - g : pad + g + 1, pad - g : pad + g + 1] = True
            noise = np.mean(region[~guard_mask])
            if rdm[r,d] > noise*alpha:
                detection.append((r,d))
    
    print(f"CFAR Detections : {len(detection)}")
    return detection

def estimate_angles(virtual: np.ndarray,
                    detections: list,
                    n_angle_fft: int = 64) -> list:
    NS, NC, V = virtual.shape
    
    # Generated windows to add to signal to prevent leakage
    win_r = windows.blackman(NS).reshape(-1, 1, 1).astype(np.float32)
    win_d = windows.blackman(NC).reshape(1, -1, 1).astype(np.float32)
    win_a = windows.hann(V).astype(np.float32)

    # Performing FFT
    range_fft = np.fft.fft(virtual*win_r, NS, axis = 0)
    doppler_fft = np.fft.fft(range_fft*win_d, NC, axis = 1)
    doppler_fft = np.fft.fftshift(doppler_fft, axes = 1)

    # Angle axis
    angle_bin_idx = 2*(np.arange(n_angle_fft) - n_angle_fft//2)/n_angle_fft
    angles_deg = np.degrees(np.arcsin(np.clip(angle_bin_idx,-1.0, 1.0)))

    results = []
    for (r, d) in detections:
        if r > NS or d > NC:
            continue
        angle_signal = doppler_fft[r, d, :]*win_a
        angle_fft = np.fft.fft(angle_signal, n_angle_fft)
        angle_fft = 20*np.log10(np.abs(np.fft.fftshift(angle_fft)))
        idx = int(np.argmax(angle_fft))
        angle = angles_deg[idx]
        results.append((r, d, angle, float(angle_fft[idx])))
    return results

def point_cloud(detection_with_angles: np.ndarray,
                NC = 255) -> np.ndarray:

    points = []

    for (r_bin, d_bin, angle_deg, mag) in detection_with_angles:
        # getting range from the range bin
        range_m = r_bin*RANGE_RES

        # getting velocity from doppler bin
        v_centered = d_bin - NC//2
        vel_mps = v_centered*VEL_RES

        # getting angles angle in rad
        angle_rad = np.radians(angle_deg)

        x_range = np.sin(angle_rad)*range_m
        y_range = np.cos(angle_rad)*range_m
        points.append([x_range, y_range, vel_mps, mag])
    return points

def main():
    i = np.random.randint(0,len(radar_data_paths))
    print(f"Id for this frame is {i}")

    # loading the matlab file to get data
    radar_data = load_frame(radar_data_paths[i])
    virtual = tdm_demux(radar_data)
    print(f"Data Shape: {radar_data.shape}")
    print(f"Virtual Data Shape: {virtual.shape}")

    # computing range doppler and range angle map
    rdm = compute_rdm(virtual)
    ram = compute_ram(virtual)
    detections = ca_cfar_2d(rdm, guard = 2, train = 4,remove_static=True)
    print(detections)
    angles = estimate_angles(virtual, detections)
    print(angles)
    pointcloud = point_cloud(angles)
    print(pointcloud)

    # plotting the maps
    draw_plots(rdm, ram, pointcloud, i)
    return 


if __name__ == "__main__":
    main()