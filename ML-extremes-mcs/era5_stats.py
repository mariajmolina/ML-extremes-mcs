import pandas as pd
import xarray as xr
import numpy as np
from id_selector import IDSelector

def generate_era5_IDs(dict_freq):
    """
    Generate the full list of IDs.
    Args:
        dict_freq (str): Hourly frequency of files for training. Defaults to ``3H``.
    """
    # generate list of IDs
    test = IDSelector(main_path = '/glade/scratch/molina/cesm_mcs/cesm_era5', 
                      start_year = 2004, 
                      end_year = 2016, 
                      month_only=None, year_only=None, mcs_only=False, 
                      percent_train=0.7, ens_num='era5')
    IDlist = test.generate_IDarray(pre_dict=True, dict_freq=dict_freq, start_str=None, end_str=None, dictsave=None)
    return IDlist

def run_stats_era(analysis_variable, data_path, ens_num='era5', dict_freq='3H'):
    """
    Run basic distribution stats and save to file.
    Args:
        analysis_variable (str): ERA5 variable.
        data_path (str): Path where the ERA5 stats will be saved.
        ens_num (str): Model data being used for training. Defaults to ``era5``.
        dict_freq (str): Hourly frequency of files for training. Defaults to ``3H``.
    """
    # generate list of IDs
    IDlist = generate_era5_IDs(dict_freq)
    if analysis_variable == 'sp':
        VAR = 'SP'
    if analysis_variable == '10v':
        VAR = 'VAR_10V'
    if analysis_variable == '10u':
        VAR = 'VAR_10U'
    if analysis_variable == '2t':
        VAR = 'VAR_2T'
    if analysis_variable == '2d':
        VAR = 'VAR_2D'
    # initialize a blank list to use for storing mean values 
    mean_list = []
    stds_list = []
    maxs_list = []
    mins_list = []
    date_list = []
    for i in IDlist:
        # open variable mask
        tmp = xr.open_dataset(f'{data_path}/dl_files/{dict_freq}/file003_{analysis_variable}_ID{i}.nc')
        date_list.append(tmp['utc_date'].values)
        tmp = tmp[VAR]
        # compute stat for each time step in file
        mean_list.append(tmp.mean(dim=['latitude','longitude'], skipna=True).values)
        stds_list.append(tmp.std(dim=['latitude','longitude'], skipna=True).values)
        maxs_list.append(tmp.max(dim=['latitude','longitude'], skipna=True).values)
        mins_list.append(tmp.min(dim=['latitude','longitude'], skipna=True).values)
    # flatten the list of mean values
    mean_array = np.array([mean_list]).squeeze()
    stds_array = np.array([stds_list]).squeeze()
    maxs_array = np.array([maxs_list]).squeeze()
    mins_array = np.array([mins_list]).squeeze()
    date_array = np.array([date_list]).squeeze()
    # create xarray dataset
    xarray_array = xr.Dataset({'averages': (['time'], mean_array),
                               'sigma':    (['time'], stds_array),
                               'maxs':     (['time'], maxs_array),
                               'mins':     (['time'], mins_array),
                               'dates':    (['time'], date_array)})
    xarray_array.to_netcdf(f'{data_path}/stats_{ens_num}_{analysis_variable}.nc')
    print(f"File saved and completed for {analysis_variable}")
