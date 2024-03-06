#!/usr/bin/env python # [1]
"""\
This script contains all the functions 
used to analyze the FRAP data in the paper
"Putative Looping Factor ZNF143/ZFP143 is 
an Essential Transcriptional Regulator with No Looping Function".

Author: Domenic Narducci
"""

# imports
import numpy as np
import seaborn as sns
import pickle
import pandas as pd
import os
from scipy.optimize import curve_fit
import scipy.stats as st
from tqdm import tqdm
from matplotlib import pyplot as plt
from itertools import product

sns.set_style(
    style='white',
    rc={'font_scale':2,'font':'Arial',  
        'xtick.bottom': True,
        'xtick.top': False,
        'ytick.left': True,
        'ytick.right': False,
        'axes.linewidth': 5}
)
sns.set_palette(palette="tab10")
sns_c = sns.color_palette(palette="tab10")

plt.rcParams['figure.figsize'] = [10, 7]

def listdir_nohidden(path):
    """
    Helper function to iterate over a directory ignoring hidden files.
    """
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            
def read_all_movies(path):
    """
    Reads all movies into a single dataframe
    path (str): a string containing the directory with the data
    returns all_data (structure containing movie data) and names (a list of names)
    """
    all_data = []
    names = []
    for filename in listdir_nohidden(path):
        file = pd.read_csv(os.path.join(path, filename))
        names.append(filename.split(".")[0])
        all_data.append(file)
    for i in range(len(all_data)):
        all_data[i]["label"] = names[i]
    return all_data, names
        
def computeGapRatio(data, start_frame):
    """
    Computes gap ratio from frap_img
    :param data: a data object
    :param start_frame: the bleaching frame
    :return: gap ratio
    """
    nonbleach_intensities = data["corrected_nonbleach"]
    return 1 - (np.mean(nonbleach_intensities[-10:]) / np.mean(nonbleach_intensities[:start_frame]))
        
def plot_single_movie(movie, name, ax=None):
    """
    Plots a single movie's bleaching curve and nonbleaching curve on the same axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # make plot
    ax.plot(movie["time"], movie["intensity"])
    ax.plot(movie["time"], movie["corrected_nonbleach"])
    
    # add labels
    ax.set_title(name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(["Bleached ROI", "Nonbleached Nuclear ROI"])
    
def plot_single_bleach_profile(movie, name, ax=None):
    """
    Plots a single movie's bleaching profile as a function of radial distance
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # make plot
    ax.plot(movie["bleach_distances"], movie["bleach_profile"])
    ax.axvline(x=movie["radius_uniform"][0], linestyle='--', color='k')
    
    # add labels
    ax.set_title(name)
    ax.set_xlabel("Radial Distance [um]")
    ax.set_ylabel("Normalized Intensity")
    
def plot_all_plots_square(all_data, names):
    """
    Plots all profiles in a square 
    """
    # get dimensions of square
    n_experiments = len(names)
    sq_dim = int(np.ceil(np.sqrt(n_experiments)))
    
    # make subplots with movies
    fig, ax = plt.subplots(sq_dim, sq_dim)
    fig.set_size_inches(20.0, 12.5)
    for i in range(n_experiments):
        print(names[i])
        plot_single_movie(all_data[i], names[i], ax[i//sq_dim, i%sq_dim])
    plt.subplots_adjust(bottom=0.0, top=1.2)
    plt.show()
    
def plot_all_radial_profiles_square(all_data, names):
    """
    Plots all radial profiles in a square
    """
    # get dimensions of square
    n_experiments = len(names)
    sq_dim = int(np.ceil(np.sqrt(n_experiments)))
    
    # make subplots with radial profiles
    fig, ax = plt.subplots(sq_dim, sq_dim)
    fig.set_size_inches(20.0, 12.5)
    for i in range(n_experiments):
        plot_single_bleach_profile(all_data[i], names[i], ax[i//sq_dim, i%sq_dim])
    plt.subplots_adjust(bottom=0.0, top=1.2)
    plt.show()
    
def plot_averaged_radial_profile(all_data):
    """
    Plots average radial profile with fit
    """
    combined_data = pd.concat(all_data)
    bleach_profiles = combined_data[["bleach_distances", "bleach_profile"]].dropna()
    bleach_profiles = bleach_profiles.sort_values("bleach_distances")
    fig, ax = plt.subplots()
    radius = 0.5

    # fit
    popt, pcov = curve_fit(photobleachingProfile, bleach_profiles["bleach_distances"], 
                           bleach_profiles["bleach_profile"], None,
                           bounds=([0, 0, 0], [1.25 * radius, 10 * radius, np.Inf]))

    # make plot
    ax.plot(bleach_profiles["bleach_distances"], bleach_profiles["bleach_profile"], 'x')

    time_pts = np.linspace(0, 2.00, 1000)
    ax.plot(time_pts, photobleachingProfile(time_pts, *popt), 'k-', linewidth=3)


    # add labels
    ax.set_title("Bleach Profile")
    ax.set_xlabel("Radial Distance [um]")
    ax.set_ylabel("Normalized Intensity")
    print("R_C = {}, sigma = {}, theta = {}".format(*popt))
    
def plot_qc_metrics(all_data):
    """
    Makes plots for nuclear radii, roi_radii, gap_ratio, bleaching_depth, and uniform_radii
    distributions
    """
    nuclear_radii = np.array([movie["nuclear_radius"][0] for movie in all_data])
    roi_radii = np.array([movie["roi_radius"][0] for movie in all_data])
    gap_ratios = np.array([movie["gap_ratio"][0] for movie in all_data])
    bleaching_depths = np.array([movie["bleaching_depth"][0] for movie in all_data])
    uniform_radii = np.array([movie["radius_uniform"][0] for movie in all_data])

    names = ["Nuclear Radius", "ROI Radius", "Gap Ratio", "Bleaching Depth", "Uniform Radius"]
    values = [nuclear_radii, roi_radii, gap_ratios, bleaching_depths, uniform_radii]

    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(20.0, 3.0)
    for i in range(len(values)):
        name = names[i]
        value = values[i]
        ax[i].hist(value)
        ax[i].axvline(x=np.mean(value), linestyle="--", color="k")
        ax[i].set_title(name)
        
def plot_data_model(all_data, ax=None, weight_fit=True, plot_model=True,
                   enforce_bleach_frame=False, bleach_idx=10):
    """
    Makes plots with continuous error bars for the data 
    """
    if enforce_bleach_frame:
        all_data = set_bleach_frame(all_data, bleach_idx)
    
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8.0, 6.0)
        
    max_len = max([len(df.index) for df in all_data])
    for df in all_data:
        if len(df.index) < max_len:
            df.append((max_len - len(df.index)) * [len(df.columns)*[np.nan]])
            df.reset_index()
    
    combined_data = pd.concat(all_data)
    combined_data = combined_data[["time", "intensity", "label"]]
    display_data = combined_data.copy()
    # normalize time points to the average time for that index
    display_data["time"] = pd.Series(np.array(len(all_data) * [np.mean(x["time"].to_numpy()) \
                                                   for _, x in display_data.groupby(level=0)]))
    combined_data["sigmas"] = pd.Series(np.array(len(all_data) * [np.std(x["intensity"].to_numpy()) \
                                        for _, x in display_data.groupby(level=0)]))

    display_data = display_data.dropna()

    # remove first and last rows to avoid anomalies
    display_data.drop(display_data.tail(1).index,inplace=True) # drop last row
    display_data.drop(display_data.head(1).index,inplace=True) # drop first row
    display_data = display_data.reset_index()

    sns.lineplot(data=display_data, x="time", y="intensity", ax=ax)
    
    time_pts = np.linspace(0, display_data["time"].to_numpy()[-1]+25, 1000)
    time_pts[0] = np.finfo(float).eps
    fit_fun = twoReactions
    lower = np.full(4, np.finfo(float).eps)
    upper = np.full(4, np.inf)
    fit_data = combined_data[combined_data["time"] >= 0]
    
    if weight_fit:
        popt,pcov = curve_fit(fit_fun, np.array(fit_data["time"]),np.array(fit_data["intensity"]), 
                              bounds=(lower, upper), sigma=fit_data["sigmas"])
    else:
        popt,pcov = curve_fit(fit_fun, np.array(fit_data["time"]),np.array(fit_data["intensity"]), 
                              bounds=(lower, upper))
    if plot_model:
        sns.lineplot(x=time_pts, y=fit_fun(time_pts, *popt), linewidth=3, color='black', ax=ax)
    
    
    # print out parameters
    k_1_off, k_2_off, C_1_eq, C_2_eq = popt
    if k_1_off < k_2_off:
        k_1_off, k_2_off, C_1_eq, C_2_eq = k_2_off,k_1_off,C_2_eq,C_1_eq
    k_1_on, k_2_on = computeKons(popt)
    F_eq = computeFreeFraction(popt)
    print("k_on_1 = ", k_1_on, 
          "\nk_off_1 = ", k_1_off,
          "\nk_on_2 = ", k_2_on,
          "\nk_off_2 = ", k_2_off,
          "\nBound Fraction 1 = ", C_1_eq,
          "\nBound Fraction 2 = ", C_2_eq,
          "\nFree Fraction = ", F_eq)
    
    ax.set_ylim([0, max(1.05, np.amax(combined_data["intensity"]))])

def plot_multiple_datasets(list_data, names, plot_model=True, weight_fit=True, 
                           enforce_bleach_frame=False, bleach_idx=10):
    """
    Plots all datasets in the list as different colors
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(19.2, 10.8)
    total_data = []
    
    time_pts = np.linspace(0, 700, 1000)
    time_pts[0] = np.finfo(float).eps
    if plot_model:
        fit_fun = twoReactions
        lower = np.full(4, np.finfo(float).eps)
        upper = np.full(4, np.inf)

        model_fits = []
        model_summary_data = []
    for i, all_data in enumerate(list_data):
        max_len = max([len(df.index) for df in all_data])
        for df in all_data:
            if len(df.index) < max_len:
                df.append((max_len - len(df.index)) * [len(df.columns)*[np.nan]])
                df.reset_index()
        # make plotting data
        if enforce_bleach_frame:
            all_data=set_bleach_frame(all_data, bleach_idx)
        combined_data = pd.concat(all_data)

        combined_data = combined_data[["time", "intensity", "label"]]
        display_data = combined_data.copy()
        # normalize time points to the average time for that index
        display_data["time"] = pd.Series(np.array(len(all_data) * [np.mean(x["time"].to_numpy()) \
                                                       for _, x in display_data.groupby(level=0)]))
        combined_data["sigmas"] = pd.Series(np.array(len(all_data) * [np.std(x["intensity"].to_numpy()) \
                                            for _, x in display_data.groupby(level=0)]))

        display_data = display_data.dropna()

        # remove first and last rows to avoid anomalies
        display_data.drop(display_data.tail(1).index,inplace=True) # drop last row
        display_data.drop(display_data.head(1).index,inplace=True) # drop first row
        display_data = display_data.reset_index()
        display_data["Condition"] = names[i]
        total_data.append(display_data)
        if plot_model:
            # fit model
            fit_data = combined_data[combined_data["time"] >= 0]

            if weight_fit:
                popt,pcov = curve_fit(fit_fun, np.array(fit_data["time"]),np.array(fit_data["intensity"]), 
                                      bounds=(lower, upper), sigma=fit_data["sigmas"])
            else:
                popt,pcov = curve_fit(fit_fun, np.array(fit_data["time"]),np.array(fit_data["intensity"]), 
                                      bounds=(lower, upper))
            model_fits.append(pd.DataFrame({'time':time_pts, 'fit':fit_fun(time_pts, *popt), 
                                            'Model Condition': names[i] + ' Model'}))

            # get parameters
            k_1_off, k_2_off, C_1_eq, C_2_eq = popt
            if k_1_off < k_2_off:
                k_1_off, k_2_off, C_1_eq, C_2_eq = k_2_off,k_1_off,C_2_eq,C_1_eq
            k_1_on, k_2_on = computeKons(popt)
            F_eq = computeFreeFraction(popt)
            model_summary_data.append(pd.DataFrame({'Condition':names[i], 'k_off_1':k_1_off, 'k_off_2':k_2_off,
                                                   'k_on_1':k_1_on, 'k_on_2':k_2_on, 'C_1_eq':C_1_eq, 
                                                    'C_2_eq':C_2_eq, 'F_eq':F_eq, 'Tau_r_1': (1/k_1_off), 
                                                   'Tau_r_2': (1/k_2_off)}, index=[0]))
        
    total_data = pd.concat(total_data).reset_index()
    sns.lineplot(data=total_data, x="time", y="intensity", hue="Condition", ax=ax)
    
    if plot_model:
        model_fits = pd.concat(model_fits).reset_index()
        sns.lineplot(data=model_fits,x="time", y="fit", style="Model Condition", 
                     linewidth=3, color='black', ax=ax)

        model_summary_data = pd.concat(model_summary_data).reset_index()
        print(model_summary_data)
        
    ax.set_ylim([0.0, 1.1])
    ax.set_xlim([display_data["time"].values[0]-10, display_data["time"].values[-1]+10])
    ax.axhline(1.0)
    ax.axvline(0.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Intensity")
    plt.savefig("skok_plot.svg", format="svg") 
    return model_summary_data, ax
    
def set_bleach_frame(list_data, bleach_index):
    """
    Sets the bleach frame to the frame given by bleach_index.
    """
    new_list_data = []
    for all_data in list_data:
        new_all_data = all_data.copy()
        new_all_data["time"] -= new_all_data["time"][bleach_index]
        new_list_data.append(new_all_data)
    
    return new_list_data
    
def twoReactions(x, k_1_off, k_2_off, C_1_eq, C_2_eq):
    """
    A two state reaction model
    """
    return 1 - C_1_eq * np.exp(- k_1_off * x) - C_2_eq * np.exp(- k_2_off * x)

def photobleachingProfile(x, r_c, sigma, theta):
    """
    Function for photobleaching profile
    :param x: x vals
    :param r_c: uniform radius
    :param theta: bleaching depth
    :param sigma: sd of normal
    :return: function evaluated on x
    """
    data = np.zeros(x.shape)
    data[x <= r_c] = theta
    data[x > r_c] = 1 - (1 - theta) * np.exp(-(x[x > r_c] - r_c)**2 / (2 * sigma**2))
    return data

def computeKons(popt):
    """
    A helper function to compute kons
    """
    k_1_off,k_2_off,C_1_eq, C_2_eq = popt
    if k_1_off < k_2_off:
        k_1_off,k_2_off,C_1_eq, C_2_eq = k_2_off,k_1_off,C_2_eq,C_1_eq
    return -(k_1_off * C_1_eq) / (C_1_eq + C_2_eq - 1), -(k_2_off * C_2_eq) / (C_1_eq + C_2_eq - 1)

def computeFreeFraction(popt):
    """
    A helper function to compute the free fraction.
    """
    return 1 - popt[2] - popt[3]

def computeKon(C_eq, k_off):
    """
    A helper function to compute k_on from k_off and Ceq
    """
    return C_eq * k_off / (1 - C_eq)

def func(x, tau):
    return np.exp(-tau * x)

def two_state_frap(t, k_off_1, k_off_2, C_1, C_2):
    """
    The two state FRAP model for fitting.
    """
    return 1 - C_1 * np.exp(-k_off_1 * t) - C_2 * np.exp(-k_off_2 * t)