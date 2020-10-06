import xarray as xr
import numpy as np
import pandas as pd
import calendar
from config import main_path_003

def extract_train_stats(filepath, variable, htype, fileIDs):
    
    """
    Extract the stats files for respective variable and train IDs.
    
    Args:
        filepath (str): Path of stats and variable data.
        variable (str): Name of variable as a string.
        htype (str): String representing hourly treatment. 
                     E.g., ``h3`` for average or ``h4`` instantaneous.
        fileIDs (array): List of IDs being used for training.

    """
    ds = xr.open_dataset(f"{filepath}/stats_{htype}_{variable}.nc")
    avgs = ds['averages'][fileIDs]
    sigs = ds['sigma'][fileIDs]
    maxs = ds['maxs'][fileIDs]
    mins = ds['mins'][fileIDs]
    return avgs, sigs, maxs, mins
