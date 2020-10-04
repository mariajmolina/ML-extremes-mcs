import preprocess
import pandas as pd
import xarray as xr
import numpy as np
from config import main_path_003, start_year, end_year
import argparse

parser = argparse.ArgumentParser(description='Computing basic stats to use for data preprocessing.')
parser.add_argument("--variable", action='store', required=True, type=str, help="Variable for computation.")
parser.add_argument("--ensemble", action='store', required=True, type=str, help="Variable for computation.")
parser.add_argument("--instant", action='store_true', required=False, dest='boolean_switch', help="Select if the variable is instantaneous.")
args=parser.parse_args()

# ensemble and variable for analysis (using index)
analysis_variable = args.variable
ens_num = args.ensemble

# create array of years for analysis
years = pd.date_range(start=start_year+'-01-01', end=end_year+'-01-01', freq='AS-JAN').year

# assign the string for instant versus mean cesm variable
if args.instant:
    inst_str = "h4"
if not args.instant:
    inst_str = "h3"

# initialize a blank list to use for storing mean values 
mean_list = []
stds_list = []
maxs_list = []
mins_list = []

for i in years:
    # open the mask dataset for the year of analysis
    data_mask = xr.open_dataset(f'{main_path_003}/mask_camPD_{i}.nc')
    # loop through the files to grab average values
    datavar = xr.open_dataset(f'{main_path_003}/b.e13.B20TRC5CN.ne120_g16.003.cam.{inst_str}.{analysis_variable}.{i}010100Z-{i}123121Z.regrid.23x0.31.nc')
    # slice the variable grid based on the mask
    tmp = preprocess.slice_003_grid(data_mask, datavar, var=analysis_variable)
    # compute mean for each time step in file
    mean_list.append(tmp[:,:,:].mean(dim=['lat','lon'], skipna=True).values)
    stds_list.append(tmp[:,:,:].std(dim=['lat','lon'], skipna=True).values)
    maxs_list.append(tmp[:,:,:].max(dim=['lat','lon'], skipna=True).values)
    mins_list.append(tmp[:,:,:].min(dim=['lat','lon'], skipna=True).values)

# flatten the list of mean values
mean_array = np.array([b for a in mean_list for b in a])
stds_array = np.array([b for a in stds_list for b in a])
maxs_array = np.array([b for a in maxs_list for b in a])
mins_array = np.array([b for a in mins_list for b in a])

# create xarray dataset
xarray_array = xr.Dataset({'averages': (['time'], mean_array),
                           'sigma':    (['time'], stds_array),
                           'maxs':     (['time'], maxs_array),
                           'mins':     (['time'], mins_array)})

# save the file
xarray_array.to_netcdf(f'{main_path_003}/averages_{month}_{analysis_variab}_plev{analysis_height}.nc')
