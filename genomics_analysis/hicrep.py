#!/usr/bin/env python # [1]
"""\
This script contains all the functions 
used to run Hi-C rep (doi: 10.1101/gr.220640.117) in the paper
"Putative Looping Factor ZNF143/ZFP143 is 
an Essential Transcriptional Regulator with No Looping Function".

Uses hicreppy: https://github.com/cmdoret/hicreppy

Author: Domenic Narducci
"""

# Package imports
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import ndimage as ndi
import cooler as clr
from hicreppy.hicrep import h_train, genome_scc
import h5py
from itertools import product
from tqdm import tqdm

def read_cooler(cooler_path, resolution=250):
    """
    Simple function wrapper to read cooler file.
    """
    if clr.fileops.is_cooler(cooler_path):
        return clr.Cooler(cooler_path)
    elif clr.fileops.is_multires_file(cooler_path):
        return clr.Cooler(cooler_path + "::resolutions/" + str(int(resolution)))
    
def list_resolutions(cooler_path):
    """
    A helper function that reads possible resolutions from mcool file. 
    """
    with h5py.File(cooler_path, 'r') as f:
        print(f["1"].keys())
        return list(f['resolutions'].keys())