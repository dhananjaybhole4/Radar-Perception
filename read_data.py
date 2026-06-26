import numpy as np

from scipy.io import loadmat



class ReadData():
    def __init__(self, path: str):
        self.path = path
    
    # load and reshape
    # loading a frame from its path and returning the data in the form of ndarray
    def load_frame(self) -> np.ndarray:
        # reading the matlab file
        radar_data = loadmat(self.path)
        # extracting the adc data from the dictionary
        radar_adc_data = radar_data["adcData"]
        return radar_adc_data

    # demux the tdm ( we have MIMO), hence from 4x2 antennas we get 8 pairs
    def tdm_demux(self, adc: np.ndarray) -> np.ndarray:
        virtual = np.transpose(adc,(0,1,3,2))
        virtual = virtual.reshape(virtual.shape[0], virtual.shape[1], -1)
        return virtual
    
