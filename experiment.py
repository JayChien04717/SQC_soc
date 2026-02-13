from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qick.asm_v2 import AveragerProgramV2
from .program_mixins import QickProgramMixin
from ..helpers import get_next_filename_labber, hdf5_generator, yml_comment
from ..plotter.liveplot import liveplotfun

# Common data path
DATA_PATH = r"C:\Users\QEL\Desktop\testqick\data"

class Experiment(ABC):
    """
    Abstract Base Class for QICK Experiments.
    Standardizes the interface for running, plotting, and saving experiments.
    """
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config


    @abstractmethod
    def run(self, py_avg: int, liveplot: bool = False, **kwargs):
        """
        Execute the experiment.
        """
        pass

    @abstractmethod
    def liveplot(self, py_avg: int, **kwargs):
        """
        Execute the experiment with live plotting.
        """
        pass

    @abstractmethod
    def saveLabber(self, qb_idx: int, comment:str):
        """
        Save the experiment data to HDF5 format compatible with Labber.
        """
        pass
