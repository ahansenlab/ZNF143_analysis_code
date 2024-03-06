#!/usr/bin/env python # [1]
"""\
This script contains all the functions 
used to analyze the fast SPT data in the paper
"Putative Looping Factor ZNF143/ZFP143 is 
an Essential Transcriptional Regulator with No Looping Function".

Author: Domenic Narducci
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from importlib import reload
sys.path.append('Spot-On-cli/fastspt')
import fastspt
from saspt import sample_detections, StateArray, RBME, StateArrayDataset
from tqdm import tqdm
import tqdm as tq
import pickle
import seaborn as sns

from scipy import ndimage as ndi
from skimage import measure
import json

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def prepare_quot_for_spoton(quot_df, pixel_size, frame_interval):
    """
    Converts a quot-style dataframe into a spoton compatible 
    dataframe.
    """
    spoton_df = quot_df[["frame", "trajectory", "x", "y"]]
    spoton_df["x"] = spoton_df["x"].multiply(pixel_size)
    spoton_df["y"] = spoton_df["y"].multiply(pixel_size)
    spoton_df["t"] = spoton_df["frame"].multiply(frame_interval)
    spoton_df = spoton_df[["frame", "t", "trajectory", "x", "y"]]
    return spoton_df

def prepare_spoton_for_anders(spoton_df):
    """
    Converts a spoton-style dataframe into an anders style list
    """
    spoton_df_grouped = spoton_df.groupby(["trajectory"])
    new_df = []
    for x in spoton_df_grouped:
        xy = x[1][["x", "y"]].to_numpy()
        t = x[1][["t"]].to_numpy()
        frame = x[1][["frame"]].to_numpy(dtype=np.uint8)
        new_df.append((xy, np.transpose(t), np.transpose(frame)))
        
    return np.array(new_df,  dtype=[('xy', object), ('t',object), ('frame', object)])
    

def batch_prepare_for_spoton(input_dir, output_dir, 
                             pixel_size, frame_interval):
    """
    Converts each quot-style dataframe in 'input_dir' into a
    spoton-style dataframe in 'output_dir'
    """
    for file in os.listdir(input_dir):
        if file.split('.')[-1] == "csv":
            quot_df = pd.read_csv(os.path.join(input_dir, file))
            if not quot_df.empty:
                spoton_df = prepare_quot_for_spoton(quot_df, 
                                                    pixel_size, 
                                                    frame_interval)
                spoton_df.to_csv(os.path.join(output_dir, "SPOTON_"+file))
            else:
                print("File ", file, " is empty!")

def convert_spoton_to_anders_format(input_dir, contains=None, 
                                    min_avg_particles=None, max_avg_particles=None, concatenated=True):
    """
    Converts spoton format to anders format
    """
    anders_format_files = []
    for file in tqdm(os.listdir(input_dir)):
        if file.split('.')[-1] == "csv":
            if contains is None or contains in file:
                spoton_df = pd.read_csv(os.path.join(input_dir, file))
                if not spoton_df.empty:
                    if min_avg_particles is not None or max_avg_particles is not None:
                        spoton_df = restrict_density(spoton_df, min_avg_particles, max_avg_particles)
                    anders_df = prepare_spoton_for_anders(spoton_df)
                    anders_format_files.append(anders_df)
                else:
                    print("File ", file, " is empty!")
    if len(anders_format_files) > 0:
        if concatenated:
            return np.concatenate(anders_format_files)
        
        else:
            return anders_format_files
    
    else: 
        return None
    
def fastspt_plot_histogram(HistVecJumps, emp_hist, HistVecJumpsCDF=None, sim_hist=None,
                   TimeGap=None, SampleName=None, CellNumb=None,
                   len_trackedPar=None, Min3Traj=None, CellLocs=None,
                   CellFrames=None, CellJumps=None, ModelFit=None,
                   D_free=None, D_bound=None, F_bound=None):
    """Function that plots an empirical histogram of jump lengths,
    with an optional overlay of simulated/theoretical histogram of 
    jump lengths. Comes from the fastspt python implementation."""

    ## Parameter parsing for text labels
    if CellLocs != None and CellFrames != None:
        locs_per_frame = round(CellLocs/CellFrames*1000)/1000
    else:
        locs_per_frame = 'na'    
    if SampleName == None:
        SampleName = 'na'
    if CellNumb == None:
        CellNumb = 'na'
    if len_trackedPar == None:
        len_trackedPar = 'na'
    if Min3Traj == None:
        Min3Traj = 'na'
    if CellLocs == None:
        CellLocs = 'na'
    if CellFrames == None:
        CellFrames = 'na'
    if CellJumps == None:
        CellJumps = 'na'
    if ModelFit == None:
        ModelFit = 'na'
    if D_free == None:
        D_free = 'na'
    if D_bound == None:
        D_bound = 'na'
    if F_bound == None:
        F_bound = 'na'

    ## Do something
    JumpProb = emp_hist
    scaled_y = sim_hist
    
    histogram_spacer = 0.055
    number = JumpProb.shape[0]
    cmap = plt.get_cmap('viridis')
    colour = [cmap(i) for i in np.linspace(0, 1, number)]

    for i in range(JumpProb.shape[0]-1, -1, -1):
        new_level = (i)*histogram_spacer
        colour_element = colour[i] #colour[round(i/size(JumpProb,1)*size(colour,1)),:]
        plt.plot(HistVecJumps, (new_level)*np.ones(HistVecJumps.shape[0]), 'k-', linewidth=1)
        for j in range(1, JumpProb.shape[1]): ## Looks like we are manually building an histogram. Why so?
            x1 = HistVecJumps[j-1]
            x2 = HistVecJumps[j]
            y1 = new_level
            y2 = JumpProb[i,j-1]+new_level
            plt.fill([x1, x1, x2, x2], [y1, y2, y2, y1], color=colour_element) # /!\ TODO MW: Should use different colours
        if type(sim_hist) != type(None): ## HistVecJumpsCDF should also be provided
            plt.plot(HistVecJumpsCDF, scaled_y[i,:]+new_level, 'k-', linewidth=2)
        if TimeGap != None:
            plt.text(0.6*max(HistVecJumps), new_level+0.3*histogram_spacer, '$\Delta t$ : {} ms'.format(TimeGap*(i+1)))
        else:
            plt.text(0.6*max(HistVecJumps), new_level+0.3*histogram_spacer, '${} \Delta t$'.format(i+1))

    plt.xlim(0,HistVecJumps.max())
    plt.ylabel('Probability')
    plt.xlabel('jump length ($\mu m$)')
    if type(sim_hist) != type(None):
        plt.title('{}; Cell number {}; Fit Type = {}; Dfree = {}; Dbound = {}; FracBound = {}, Total trajectories: {}; => Length 3 trajectories: {}, \nLocs = {}, Locs/Frame = {}; jumps: {}'
          .format(
              SampleName, CellNumb, ModelFit,
              D_free, D_bound, F_bound,
              len_trackedPar, Min3Traj, CellLocs,
              locs_per_frame,
              CellJumps))
    else:
        plt.title('{}; Cell number {}; Total trajectories: {}; => Length 3 trajectories: {}, \nLocs = {}, Locs/Frame = {}; jumps: {}'
          .format(
              SampleName, CellNumb,
              len_trackedPar, Min3Traj, CellLocs,
              locs_per_frame,
              CellJumps))
    plt.yticks([])

def fit_spoton_2_state(dataset, dT, cdf=True, use_entire_traj=True, frac_bound=[0, 1], d_free=[0.15, 25],
                               d_bound=[0.0005, 0.08], sigma_bound = [0.005, 0.1],
                               loc_error=0.035, iterations=3, dZ=0.700, 
                                fit_sigma=False, use_Z_corr=True):
    """
    Helper/wrapper function for fitting 2 state spotOn model.
    dataset: SPT dataset in anders format
    dT: time between frames in seconds
    cdf: use CDF to generate jump length distribution
    use_entire_traj: use the entire trajectory instead of 8
    frac_bound: fit range for bound fraction
    d_free: fit range for free diffusion coefficient
    d_bound: fit range for bound diffusion coefficient
    loc_error: localization error in um, if None fits it from the data
    iterations: number of iterations of fitting to perform
    dZ: axial illumination slice size in nm
    fit_2_states: fit a 2 state model?
    use_Z_corr: use Z correction?
    """
    fit_2_states=True
    # Generate jump length distribution
    h1 = fastspt.compute_jump_length_distribution(dataset, CDF=cdf, useEntireTraj=use_entire_traj)

    if cdf:
        HistVecJumps = h1[2]
        JumpProb = h1[3]
        HistVecJumpsCDF = h1[0]
        JumpProbCDF = h1[1]
        
    else:
        HistVecJumps = h1[0]
        JumpProb = h1[1]
        HistVecJumpsCDF = h1[0]
        JumpProbCDF = h1[1]

    print("Computation of jump lengths performed in {:.2f}s".format(h1[-1]['time']))
    
    if loc_error is not None:
        ## Generate a dictionary of parameters
        LB = [d_free[0], d_bound[0], frac_bound[0]]
        UB = [d_free[1], d_bound[1], frac_bound[1]]
    else:
        ## Generate a dictionary of parameter
        LB = [d_free[0], d_bound[0], frac_bound[0], sigma_bound[0]]
        UB = [d_free[1], d_bound[1], frac_bound[1], sigma_bound[1]]

    params = {'UB': UB,
              'LB': LB,
              'LocError': loc_error, # Manually input the localization error in um: 35 nm = 0.035 um.
              'iterations': iterations, # Manually input the desired number of fitting iterations:
              'dT': dT, # Time between frames in seconds
              'dZ': dZ, # The axial illumination slice: measured to be roughly 700 nm
              'ModelFit': [1,2][False],
              'fit2states': fit_2_states,
              'fitSigma': fit_sigma,
              'a': 0.15716,
              'b': 0.20811,
              'useZcorr': use_Z_corr
    }
    
    ## Perform the fit
    fit = fastspt.fit_jump_length_distribution(JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF, **params)
    
    ## Generate the PDF corresponding to the fitted parameters
    y = fastspt.generate_jump_length_distribution(fit.params, 
                                                  JumpProb = JumpProbCDF, r=HistVecJumpsCDF,
                                                  LocError = fit.params['sigma'].value, 
                                                  dT = params['dT'], dZ = params['dZ'], 
                                                  a = params['a'], b = params['b'], norm=True, 
                                                  useZcorr=params['useZcorr'])
    ## Normalization does not work for PDF yet (see commented line in fastspt.py)
    if cdf:
        y*=float(len(HistVecJumpsCDF))/float(len(HistVecJumps))
    
    return h1, fit, y

def fit_spoton_3_state(dataset, dT, cdf=True, use_entire_traj=True, frac_bound=[0, 1], frac_fast=[0, 1],
                       d_free=[0.15, 25], d_med=[0.15, 10], d_bound=[0.0005, 0.08], sigma_bound = [0.005, 0.1],
                               loc_error=0.035, iterations=3, dZ=0.700, 
                                fit_2_states=True, fit_sigma=False, use_Z_corr=True):
    """
    Helper/wrapper function for fitting 2 state spotOn model.
    dataset: SPT dataset in anders format
    dT: time between frames in seconds
    cdf: use CDF to generate jump length distribution
    use_entire_traj: use the entire trajectory instead of 8
    frac_bound: fit range for bound fraction
    d_free: fit range for free diffusion coefficient
    d_bound: fit range for bound diffusion coefficient
    loc_error: localization error in um, if None fits it from the data
    iterations: number of iterations of fitting to perform
    dZ: axial illumination slice size in nm
    fit_2_states: fit a 2 state model?
    use_Z_corr: use Z correction?
    """
    fit_2_states=False
    # Generate jump length distribution
    h1 = fastspt.compute_jump_length_distribution(dataset, CDF=cdf, useEntireTraj=use_entire_traj)

    if cdf:
        HistVecJumps = h1[2]
        JumpProb = h1[3]
        HistVecJumpsCDF = h1[0]
        JumpProbCDF = h1[1]
        
    else:
        HistVecJumps = h1[0]
        JumpProb = h1[1]
        HistVecJumpsCDF = h1[0]
        JumpProbCDF = h1[1]

    print("Computation of jump lengths performed in {:.2f}s".format(h1[-1]['time']))
    
    if loc_error is not None:
        ## Generate a dictionary of parameters
        LB = [d_free[0], d_med[0], d_bound[0], frac_fast[0], frac_bound[0]]
        UB = [d_free[1], d_med[1], d_bound[1], frac_fast[1], frac_bound[1]]
    else:
        ## Generate a dictionary of parameter
        LB = [d_free[0], d_med[0], d_bound[0], frac_fast[0], frac_bound[0], sigma_bound[0]]
        UB = [d_free[1], d_med[1], d_bound[1], frac_fast[1], frac_bound[1], sigma_bound[1]]

    params = {'UB': UB,
              'LB': LB,
              'LocError': loc_error, # Manually input the localization error in um: 35 nm = 0.035 um.
              'iterations': iterations, # Manually input the desired number of fitting iterations:
              'dT': dT, # Time between frames in seconds
              'dZ': dZ, # The axial illumination slice: measured to be roughly 700 nm
              'ModelFit': [1,2][False],
              'fit2states': fit_2_states,
              'fitSigma': fit_sigma,
              'a': 0.15716,
              'b': 0.20811,
              'useZcorr': use_Z_corr
    }
    
    ## Perform the fit
    fit = fastspt.fit_jump_length_distribution(JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF, **params)
    
    ## Generate the PDF corresponding to the fitted parameters
    y = fastspt.generate_jump_length_distribution(fit.params, 
                                                  JumpProb = JumpProbCDF, r=HistVecJumpsCDF,
                                                  LocError = fit.params['sigma'].value, 
                                                  dT = params['dT'], dZ = params['dZ'], 
                                                  a = params['a'], b = params['b'], norm=True, 
                                                  fit2states = params['fit2states'], useZcorr=params['useZcorr'])
    ## Normalization does not work for PDF yet (see commented line in fastspt.py)
    if cdf:
        y*=float(len(HistVecJumpsCDF))/float(len(HistVecJumps))
    
    return h1, fit, y

#### Masking functions ####

def mask_tracked_image(fake_img, gauss_laplace_sigma=18, gl_thresh=-0.001, strel_size=3, 
                       iters=5, area_thresh=2000):
    """
    Automatic masker for tracking images. Uses supplied parameters and applies them to 
    `fake_img` to make a binary mask.
    """
    gauss_laplace_img = ndi.gaussian_laplace(fake_img, gauss_laplace_sigma)
    binary_mask = gauss_laplace_img < gl_thresh
    strel = np.ones((strel_size,strel_size))
    binary_mask = ndi.binary_closing(binary_mask, strel)
    binary_mask = ndi.binary_fill_holes(binary_mask)
    binary_mask = ndi.binary_erosion(binary_mask, strel, iters)
    mask_labels, num_labels = measure.label(binary_mask, return_num=True)
    binary_mask = np.zeros(binary_mask.shape)
    for i in range(num_labels):
        if np.sum(mask_labels == i + 1) >= area_thresh:
            binary_mask[mask_labels == i + 1] = 1
    
    binary_mask = ndi.binary_dilation(binary_mask, strel, iters-1)
    return binary_mask

def make_fake_img(traj_df):
    """
    Makes a fake image from tracking data.
    """
    xs, ys = traj_df["x"].to_numpy(), traj_df["y"].to_numpy()
    xs = np.concatenate((xs, np.array([0.0, 200.0])))
    ys = np.concatenate((ys, np.array([0.0, 200.0])))
    fake_img_hist = np.histogram2d(xs, ys, bins=201)
    return fake_img_hist[0]

def apply_mask_to_track(traj_df, binary_mask):
    """
    Applies binary mask to trajectory dataframe. 
    """
    x_round = np.round(traj_df["x"]).astype(int)
    y_round = np.round(traj_df["y"]).astype(int)
    binary_val = binary_mask[x_round, y_round]
    excluded_trajs = np.unique(traj_df[binary_val == False]["trajectory"].to_numpy())
    exclusions = np.array([traj not in excluded_trajs for traj in traj_df["trajectory"]])
    return traj_df[exclusions]

def bulk_test_masking_params(input_dir, contains=None, masking_kwargs=None):
    """
    Tests masking parameters on a folder of tracking data. Used to select masking parameters. 
    """
    if masking_kwargs is None:
        masking_kwargs = {"gauss_laplace_sigma":18, "gl_thresh":-0.001, "strel_size":3, "iters":5, "area_thresh":2000}

    n_files = len(os.listdir(input_dir))
    n_rows = 1 + (n_files - 1) // 5
    fig, ax = plt.subplots(2 * n_rows, min(n_files, 5), figsize=(14, 5*n_rows))
    for i, file in enumerate(os.listdir(input_dir)):
        if contains is None or contains in file:
            traj_df = pd.read_csv(os.path.join(input_dir, file))
            fake_img = make_fake_img(traj_df)
            ax[2 * (i // 5), i%5].imshow(fake_img.T, cmap='gray', vmax=10, aspect=1)
            binary_mask = mask_tracked_image(fake_img, **masking_kwargs)
            excluded_img = fake_img.copy()
            excluded_img[binary_mask == 0] = 0
            ax[1 + 2 * (i // 5), i%5].imshow(excluded_img.T, cmap='gray', vmax=10, aspect=1)
            ax[2 * (i // 5), i%5].set_title(" ".join(file.split("_")[3:6]))

    for ax_i in ax.flatten():
        ax_i.set_axis_off()
    fig.suptitle(masking_kwargs)

def apply_masking_parameters(input_file, masking_kwargs):
    """
    Reads file `input_file` and returns masked version based on masking kwargs
    """
    traj_df = pd.read_csv(input_file)
    fake_img = make_fake_img(traj_df)
    binary_mask = mask_tracked_image(fake_img, **masking_kwargs)
    new_df = apply_mask_to_track(traj_df, binary_mask)
    return new_df

def batch_apply_mask(masking_file):
    """
    Batch applies masking parameters to trajectories to get masked trajectories. 
    """
    masking_df = pd.read_csv(masking_file)
    for file_path, contains, masking_kwargs_str in zip(masking_df["Path"], masking_df["Contains"], masking_df["kwargs"]):
        out_path = file_path.split("/")
        out_path = out_path[:-2] + ["masked"] + [out_path[-2]]
        out_path = "/".join(out_path)
        masking_kwargs = json.loads(masking_kwargs_str)
        for i, file in tq.tqdm(enumerate(os.listdir(file_path))):
            if contains in file:
                input_file = os.path.join(file_path, file)
                new_df = apply_masking_parameters(input_file, masking_kwargs)
                if not os.path.exists(out_path):
                    os.makedirs(out_path) 
                new_df.to_csv(os.path.join(out_path, file))

