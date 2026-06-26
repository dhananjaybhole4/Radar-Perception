import numpy as np
from scipy.signal import windows
from pathlib import Path

from visualize_data import Visualize
from compute_fft import ComputeFFT
from read_data import ReadData

# hardware constants (AWR1843)
C = 3e8
FC = 77e9
BANDWIDTH = 0.67e9
CHIRP_TIME = 60e-6
LAMDA = C / FC
D_ANTENNA = LAMDA / 2

N_SAMPLES = 128
N_CHIRPS = 255
N_RX = 4
N_TX = 2
N_VIRTUAL = N_RX * N_TX

RANGE_RES = C / (2 * BANDWIDTH)
VEL_RES = LAMDA / (2 * N_CHIRPS * CHIRP_TIME)
MAX_RANGE = N_SAMPLES * RANGE_RES
MAX_VEL = VEL_RES * N_CHIRPS / 2

dataset = Path.cwd() / "Automotive"
radar_data_paths = list(dataset.rglob("*.mat"))
camera_data_paths = list(dataset.rglob("*.jpg"))

print(f"Range Resolution: {RANGE_RES*100:.2f} cm")
print(f"Maximum Range: {MAX_RANGE:.2f} m")
print(f"Velocity Resolution: {VEL_RES:.2f} m/s")
print(f"Maximum Velocity: {MAX_VEL:.2f} m/s")


def ca_cfar_2d(rdm, guard=4, train=8, pfa=1e-2, static_bins=2, remove_static=True):
    n_range, n_doppler = rdm.shape
    alpha = train * (pfa ** (-1.0 / train) - 1)
    pad = guard + train
    detection = []
    zero_bin = n_doppler // 2

    for r in range(pad, n_range - pad):
        for d in range(pad, n_doppler - pad):
            if remove_static and abs(zero_bin - d) <= static_bins:
                continue
            region = rdm[r - pad: r + pad + 1, d - pad: d + pad + 1].copy()
            guard_mask = np.zeros_like(region, dtype=bool)
            g = guard
            guard_mask[pad - g: pad + g + 1, pad - g: pad + g + 1] = True
            noise = np.mean(region[~guard_mask])
            if rdm[r, d] > noise * alpha:
                detection.append((r, d))

    print(f"CFAR Detections : {len(detection)}")
    return detection


def estimate_angles(virtual, detections, n_angle_fft=64):
    NS, NC, V = virtual.shape

    win_r = windows.blackman(NS).reshape(-1, 1, 1).astype(np.float32)
    win_d = windows.blackman(NC).reshape(1, -1, 1).astype(np.float32)
    win_a = windows.hann(V).astype(np.float32)

    range_fft = np.fft.fft(virtual * win_r, NS, axis=0)
    doppler_fft = np.fft.fft(range_fft * win_d, NC, axis=1)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=1)

    angle_bin_idx = 2 * (np.arange(n_angle_fft) - n_angle_fft // 2) / n_angle_fft
    angles_deg = np.degrees(np.arcsin(np.clip(angle_bin_idx, -1.0, 1.0)))

    results = []
    for (r, d) in detections:
        if r >= NS or d >= NC:
            continue
        angle_signal = doppler_fft[r, d, :] * win_a
        angle_fft = np.fft.fft(angle_signal, n_angle_fft)
        angle_fft = 20 * np.log10(np.abs(np.fft.fftshift(angle_fft)) + 1e-9)
        idx = int(np.argmax(angle_fft))
        angle = angles_deg[idx]
        results.append((r, d, angle, float(angle_fft[idx])))
    return results


def point_cloud(detections_with_angles, NC=255):
    points = []
    for (r_bin, d_bin, angle_deg, mag) in detections_with_angles:
        range_m = r_bin * RANGE_RES
        v_centered = d_bin - NC // 2
        vel_mps = v_centered * VEL_RES
        angle_rad = np.radians(angle_deg)

        x_range = np.sin(angle_rad) * range_m
        y_range = np.cos(angle_rad) * range_m
        points.append([x_range, y_range, vel_mps, mag])
    return points


def main():
    i = np.random.randint(0, len(radar_data_paths))
    print(f"Id for this frame is {i}")

    reader = ReadData(str(radar_data_paths[i]))
    radar_data = reader.load_frame()
    virtual = reader.tdm_demux(radar_data)
    print(f"Data Shape: {radar_data.shape}")
    print(f"Virtual Data Shape: {virtual.shape}")

    fft_computer = ComputeFFT(virtual)
    rdm = fft_computer.compute_rdm(virtual)
    ram = fft_computer.compute_ram(virtual)

    detections = ca_cfar_2d(rdm, guard=2, train=4, remove_static=True)
    print(detections)
    angles = estimate_angles(virtual, detections)
    print(angles)
    pointcloud = point_cloud(angles)
    print(pointcloud)

    viz = Visualize(rdm, ram, pointcloud, i, RANGE_RES, VEL_RES, camera_data_paths)
    viz.draw_plots(rdm, ram, pointcloud, i)


if __name__ == "__main__":
    main()