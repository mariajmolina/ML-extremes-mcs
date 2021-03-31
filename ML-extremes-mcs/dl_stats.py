import xarray as xr
import numpy as np
import pandas as pd
import calendar

def extract_train_stats(filepath, variable, fileIDs):
    """
    Extract the stats files for respective ERA5 variable and train IDs.
    Args:
        filepath (str): Path of stats and variable data.
        variable (str): Name of ERA5 variable as a string.
        fileIDs (array): List of IDs being used for training.
    Returns:
        Averages, standard deviations, minimums, and maximums.
    """
    ds = xr.open_dataset(f"{filepath}/stats_era5_{variable}.nc")
    avgs = ds['averages'][fileIDs]
    sigs = ds['sigma'][fileIDs]
    mins = ds['mins'][fileIDs]
    maxs = ds['maxs'][fileIDs]
    return avgs, sigs, mins, maxs

def z_score(da, avgs, stds):
    """
    Compute z-score for training data.
    Normalizing data for ML model training.
    Args:
        da (array): Array of training variable.
        avgs (array): Array of averages for training variable.
        stds (array): Array of standard deviations for training variable.
    """
    return (da - np.nanmean(avgs))/np.nanstd(stds)

def min_max_scale(da, mins, maxs):
    """
    Scale training data by minimum and maximum values.
    Args:
        da (array): Array of training variable.
        mins (array): Array of minimum values for training variable.
        maxs (array): Array of maximum values for training variable.
    """
    if np.nanmax(maxs) == np.nanmin(mins):
        raise Exception("Max and min are equal to each other!")
    if np.nanmax(maxs) != np.nanmin(mins):
        return (da - np.nanmin(mins))/(np.nanmax(maxs) - np.nanmin(mins))
