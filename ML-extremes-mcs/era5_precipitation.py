import glob
from datetime import timedelta
import pandas as pd
import xarray as xr


def era5_precip_hourly(glob_string, variable, save_directory):
    """
    Convert files from forecast coordinates to one time coordinate.

    Args:
        glob_string (str): String where files are located.
        variable (str): Name of variable.
        save_directory (str): Directory where to save restructured ERA5 file.
    """
    # 2004 - 2019
    filelist = sorted(glob.glob(glob_string))[(128 * 12)-1: -81] # this needs updating as time passes!

    print('List assembled')

    for filename in filelist:

        ds = xr.open_dataset(filename)

        time_array = []

        for i in ds[variable].coords['forecast_initial_time']:

            for j in ds[variable].coords['forecast_hour']:

                time_array.append(
                    pd.to_datetime(i.values) + timedelta(hours=j.values.item()-1))

        tmp = ds.stack(time=['forecast_initial_time', 'forecast_hour'])

        tmp = tmp.assign_coords(time=time_array)

        tmp.to_netcdf(save_directory+'/'+filename.split('/')[-1])

        print(filename.split('/')[-1], 'completed!')
