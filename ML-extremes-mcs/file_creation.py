import xarray as xr
import numpy as np
import pandas as pd
import calendar
from config import main_path, start_year, end_year

class GenerateTrainData:
    
    """Class instantiation of GenerateTrainData:
    
    Here we will be preprocessing data for deep learning model training.
    The objective is to save files for each time index to subsequently load for training.
    IDs are assigned to each file.
    
    Attributes:
        math_path (str): Path where files are located.
        start_year (int): Start year of analysis.
        end_year (int): Final year of analysis.
        month (int): Month of analysis.
        variable (str): Variable for file generation.
        height (int): The respective height index of the file.
        pressure (boolean): Whether to use pressure surfaces variable file.
    """
        
    def __init__(self, main_path, start_year, end_year, month, variable, height, pressure=True):
        
        self.main_directory = main_path
        self.year_start = start_year
        self.year_end = end_year
        self.month = month
        self.variable = variable
        self.height = height
        self.pressure = pressure
        
    def make_month(self):
        """
        Returns all caps month string.
        """
        return calendar.month_abbr[self.month].upper()
    
    def make_dict(self):
        """
        Create dictionary of the indices using study time range.
        These indices must be fixed values, hence not using custom time ranges.
        """
        alldates = pd.date_range(start=f'04-01-1991', end='07-31-2005', freq='3H')
        alldates = alldates[(alldates.month>=4)&(alldates.month<=7)]
        dict_dates = {}
        for i, j in enumerate(alldates):
            dict_dates[j] = i
        return dict_dates
        
    def make_years(self):
        """
        Returns array of years for data generation.
        """
        years = pd.date_range(start=self.year_start+'-01-01', end=self.year_end+'-01-01', freq='AS-JAN').year
        return years
    
    def slice_plevs_grid(self, data_mask, data, lat='lat', lon='lon'):
        """
        Slice the variable data on pressure level grid to match the mask data.

        Args:
            mask (xarray dataset): MCS object mask.
            data (xarray dataset): The data to slice.
            lat (str): Latitude dimension name. Defaults to ``lat``.
            lon (str): Longitude dimension name. Defaults to ``lon``.

        Returns:
            Variable data sliced to spatial extent of the mask data.
        """
        datavar = data[self.variable]
        datavar = datavar.isel(ncl5=self.height,
                               ncl6=slice(np.where(data[lat].values==data_mask['lat'][0].values)[0][0],
                                          np.where(data[lat].values==data_mask['lat'][-1].values)[0][0]+1),
                               ncl7=slice(np.where(data[lon].values==data_mask['lon'][0].values)[0][0],
                                          np.where(data[lon].values==data_mask['lon'][-1].values)[0][0]+1))
        return datavar
        
    def open_variable_file(self, year):
        """
        Returns opened and sliced spatial region of data for respective year.
        """
        if self.pressure:
            data = xr.open_dataset(
                self.main_directory+f'/plevs_FV_768x1152.bilinear.{self.make_month()}_b.e13.B20TRC5CN.ne120_g16.002_{self.variable}_{year}.nc')
            data = data.assign_coords(ncl4=("ncl4",data.time),
                                      ncl6=("ncl6",data.lat.values),
                                      ncl7=("ncl7",data.lon.values))
        return data
    
    def open_mask_file(self):
        """
        Open the MCS mask file.
        """
        return xr.open_dataset(f"{self.main_directory}/mask_camPD_{self.make_month()}.nc")
        
    def generate_files(self):
        """
        Save the files for each time period and variable with respective ID.
        """
        print("starting file generation...")
        yr_array = self.make_years()
        indx_array = self.make_dict()
        mask = self.open_mask_file()
        print("opening variable file...")
        for yr in yr_array:
            data = self.open_variable_file(yr)
            if self.pressure:
                data = self.slice_plevs_grid(mask, data)
                print(f"{yr} opened and sliced successfully...")
                for t in data.ncl4:
                    tmpdata = data.sel(ncl4=t)
                    tmpdata = tmpdata.to_dataset()
                    indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
                    tmpdata.to_netcdf(f"{self.main_directory}/dl_files/plev_{self.variable}_hgt{self.height}_ID{indx_val}.nc")
        print("Job complete!")