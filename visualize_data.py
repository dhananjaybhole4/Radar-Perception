import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

class Visualize():
    def __init__(self, 
                 rdm: np.ndarray, 
                 ram: np.ndarray, 
                 pointcloud: np.ndarray, 
                 i: int, 
                 RANGE_RES, 
                 VEL_RES,
                 camera_data_paths):
        self.rdm = rdm
        self.ram = ram
        self.pointcloud = pointcloud
        self.i = i
        self.RANGE_RES = RANGE_RES
        self.VEL_RES = VEL_RES
        self.camera_data_paths = camera_data_paths  

    def draw_plots(self, rdm: np.ndarray,
               ram: np.ndarray,
               pointcloud: np.ndarray,
               i: int):
        fig, axes = plt.subplots(2,2, figsize = (12,10))

        # plot for rdm
        ax = axes[0,0]
        n_range, n_doppler = rdm.shape
        r_ax = np.arange(n_range)*self.RANGE_RES
        d_ax = (np.arange(n_doppler) - n_doppler//2)*self.VEL_RES
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
        im4 = ax.imshow(Image.open(self.camera_data_paths[i]))

        plt.show()