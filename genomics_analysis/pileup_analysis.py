#!/usr/bin/env python # [1]
"""\
This script contains all the functions 
used for pileup analysis in the paper
"Putative Looping Factor ZNF143/ZFP143 is 
an Essential Transcriptional Regulator with No Looping Function"

Author: Domenic Narducci
"""

# Package imports
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from coolpuppy import coolpup
from coolpuppy.lib import numutils
from coolpuppy.lib.puputils import divide_pups
from coolpuppy import plotpup
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cooler
import bioframe
import cooltools
from cooltools import expected_cis, expected_trans
from cooltools.lib import plotting
import cooler as clr
import h5py

def read_cooler(cooler_path, resolution=250):
    """
    Simple function wrapper to read cooler file at a particular resolution.
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
        return list(f['resolutions'].keys())
    
def read_csv_as_bedpe(csv_path, pd_kwargs={}):
    """
    A helper function that reads a tsv file in as a pandas dataframe, converting names to be consistent
    with bedpe file format.
    
    Parameters
    ----------
    csv_path: (str) path to file
    pd_kwargs: keyword arguments to pass to pandas read_csv method
    """
    df = pd.read_csv(csv_path, names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'], **pd_kwargs)
#     df.rename(columns={'BIN1_CHR': 'chrom1', 'BIN1_START': 'start1', 
#                                    'BIN1_END': 'end1', 'BIN2_CHROMOSOME': 'chrom2', 'BIN2_START': 'start2', 
#                                    'BIN2_END': 'end2'}, inplace=True)
    return df

def generate_multiple_dot_pileups(cooler_path_list, csv_path_list, resolution, view_df=None, expected=None,
                                   pileup_kwargs={}, parent_dir="", csv_dir=""):
    """
    A utility function for generating pilups for multiple datasets. 
    
    Parameters
    ----------
    cooler_path_list: (list of str) a list of paths to cool or mcool files
    csv_path_list: (list of str) a list of paths to files in bedpe format listing dot loci
    resolution: (int) if files are mcools, resolution of dataset to pull. 
    pileup_kwargs: (dict) keyword argument to pass to coolpuppy pileup function. 
        See: https://coolpuppy.readthedocs.io/en/latest/modules.html#coolpuppy.coolpup.pileup
    parent_dir: (str) parent directory containing cooler and csv files
    """
    if len(cooler_path_list) != len(csv_path_list):
        raise ValueError('cooler_path_list and csv_path_list must have same dimension')
        
    pileup_dfs = []
    
    if expected is None or view_df is None:
        for c_path, csv_path in zip(cooler_path_list, csv_path_list):
            curr_cooler = read_cooler(os.path.join(parent_dir, c_path), resolution)

            if csv_path.split('.')[-1] == 'tsv':
                pd_kwargs = {'sep':'\t'}
            elif csv_path.split('.')[-1] == 'bedpe':
                pd_kwargs = {'sep':'\t'}
            elif csv_path.split('.')[-1] == 'csv':
                pd_kwargs = {}

            curr_bedpe = read_csv_as_bedpe(os.path.join(csv_dir, csv_path), pd_kwargs)

            curr_pileup = coolpup.pileup(curr_cooler, curr_bedpe, "bedpe", **pileup_kwargs)
            pileup_dfs.append(curr_pileup)
            
    else:
        for c_path, csv_path, expect in zip(cooler_path_list, csv_path_list, expected):
            curr_cooler = read_cooler(os.path.join(parent_dir, c_path), resolution)

            if csv_path.split('.')[-1] == 'tsv':
                pd_kwargs = {'sep':'\t'}
            elif csv_path.split('.')[-1] == 'bedpe':
                pd_kwargs = {'sep':'\t'}
            elif csv_path.split('.')[-1] == 'csv':
                pd_kwargs = {}

            curr_bedpe = read_csv_as_bedpe(os.path.join(csv_dir, csv_path), pd_kwargs)

            curr_pileup = coolpup.pileup(curr_cooler, curr_bedpe, "bedpe", view_df=view_df,
                                         expected_df=expect, **pileup_kwargs)
            pileup_dfs.append(curr_pileup)
    
    return pd.concat(pileup_dfs)


# Functions to help with plotting: https://cooltools.readthedocs.io/en/latest/notebooks/insulation_and_boundaries.html
def pcolormesh_45deg(ax, matrix_c, start=0, resolution=1, *args, **kwargs):
    """
    From cooltools.
    """
    start_pos_vector = [start+resolution*i for i in range(len(matrix_c)+1)]
    import itertools
    n = matrix_c.shape[0]
    t = np.array([[1, 0.5], [-1, 0.5]])
    matrix_a = np.dot(np.array([(i[1], i[0])
                                for i in itertools.product(start_pos_vector[::-1],
                                                           start_pos_vector)]), t)
    x = matrix_a[:, 1].reshape(n + 1, n + 1)
    y = matrix_a[:, 0].reshape(n + 1, n + 1)
    im = ax.pcolormesh(x, y, np.flipud(matrix_c), *args, **kwargs)
    im.set_rasterized(True)
    return im

from matplotlib.ticker import EngFormatter
bp_formatter = EngFormatter('b')
def format_ticks(ax, x=True, y=True, rotate=True):
    """
    From cooltools.
    """
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)