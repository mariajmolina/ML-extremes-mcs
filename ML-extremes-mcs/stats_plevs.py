import preprocess
import pandas as pd
import xarray as xr
import numpy as np
from config import main_path, start_year, end_year
import argparse

parser = argparse.ArgumentParser(description='Computing basic stats to use for data preprocessing.')
parser.add_argument("--month", required=True, type=str, help="This is the month string choice (3-letter all caps).")
parser.add_argument("--height", required=True, type=int, help="Height index for pressure level. 0 thru 6; 950 thru 500.")
parser.add_argument("--variable", required=True, type=str, help="Variable for computation.")
args=parser.parse_args()

# month for analysis
month = args.month

# height and variable for analysis (using index)
analysis_height = args.height
analysis_variab = args.variable

# create array of years for analysis
years = pd.date_range(start=start_year+'-01-01', end=end_year+'-01-01', freq='AS-JAN').year

# initialize a blank list to use for storing mean values 
mean_list = []
stds_list = []
maxs_list = []
mins_list = []

# open the mask dataset for the month of analysis
data_mask = xr.open_dataset(main_path+f'mask_camPD_{month}.nc')

# loop through the files to grab average values
for i in years:
    datavar = xr.open_dataset(main_path+f'plevs_FV_768x1152.bilinear.{month}_b.e13.B20TRC5CN.ne120_g16.002_{analysis_variab}_{i}.nc')
    tmp = preprocess.slice_plevs_grid(data_mask, datavar, ncl5=analysis_height, var=analysis_variab)
    # compute mean for each time step in file
    mean_list.append(tmp[:,:,:].mean(dim=['ncl6','ncl7'], skipna=True).values)
    stds_list.append(tmp[:,:,:].std(dim=['ncl6','ncl7'], skipna=True).values)
    maxs_list.append(tmp[:,:,:].max(dim=['ncl6','ncl7'], skipna=True).values)
    mins_list.append(tmp[:,:,:].min(dim=['ncl6','ncl7'], skipna=True).values)

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
xarray_array.to_netcdf(main_path+f'averages_{month}_{analysis_variab}_plev{analysis_height}.nc')