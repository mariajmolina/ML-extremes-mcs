import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import calendar
import pickle
from calendar import monthrange
from config import main_path_003, start_year, end_year

class GenerateTrainData:
    
    """Class instantiation of GenerateTrainData:
    
    Here we will be preprocessing data for deep learning model training.
    The objective is to save files for each time index to subsequently load for training.
    IDs are assigned to each file.
    
    Attributes:
        math_path (str): Path where files are located.
        start_year (int): Start year of analysis.
        end_year (int): Final year of analysis.
        variable (str): Variable for file generation. Defaults to ``None``.
        month (int): Month of analysis. Defaults to ``None``. Required for ens_num: 002.
        instant (boolean): Whether instantaneous variable. Defaults to ``False``, which is the 3-hourly average variable.
        height (int): The respective height index of the file. Defaults to ``None``, which is for single level variables.
        pressure (boolean): Whether to use pressure surfaces variable file. Defaults to ``False`` (e.g., 002 plev files).
        ens_num (str): The CESM CAM ensemble number (can be 002, 003, or era5). Defaults to ``003``.
        era5_directory (str): Location of ERA5 files in RDA (ds: 633.0).
    """
        
    def __init__(self, main_path, start_year, end_year, variable=None, month=None, instant=False, height=None, pressure=False, 
                 ens_num='003', era5_directory=None, mcs_directory=None):
        
        self.main_directory = main_path
        self.year_start = start_year
        self.year_end = end_year
        self.variable = variable
        self.month = month
        self.instant = instant
        if self.instant:
            self.inst_str = 'h4'
        if not self.instant:
            self.inst_str = 'h3'
        self.height = height
        self.pressure = pressure
        self.ens_num = ens_num
        self.era5_directory = era5_directory
        self.mcs_directory = mcs_directory
        
    def make_month(self):
        """
        Returns all caps month string.
        """
        return calendar.month_abbr[self.month].upper()
    
    def make_dict(self, start_str, end_str, frequency='3H', savepath=None):
        """
        Create dictionary of the indices using study time range.
        These indices must be fixed values for the study time range.
        Args:
            start_str and end_str (str): Start and end times for date range.
                For 002 use: start_str=f'04-01-1991', end_str=f'07-31-2005 23:00:00'
                For 003 use: start_str=f'01-01-2000 03:00:00', end_str=f'01-01-2006 00:00:00'
                For era5 use: start_str='2004-01-01 00:00:00', end_str='2016-12-31 23:00:00'
            frequency (str): Spacing for time intervals. E.g., ``3H``.
            savepath (str): Path to save dictionary containing indices.
        """
        alldates = pd.date_range(start=start_str, end=end_str, freq=frequency)
        # cesm doesn't do leap years
        alldates = alldates[~((alldates.month == 2) & (alldates.day == 29))]
        dict_dates = {}
        if self.ens_num == '003' or self.ens_num == '002':
            for i, j in enumerate(alldates):
                dict_dates[j] = i
            return dict_dates
        if self.ens_num == 'era5':
            # file indices for dictionary -- not enumerate since some masks missing
            # this option also provides flexibility for 3H or 1H
            j = 0
            yrs = alldates.strftime('%Y')
            mos = alldates.strftime('%m')
            dys = alldates.strftime('%d')
            hrs = alldates.strftime('%H')
            yrorig = yrs[0]
            for i, yr, mo, dy, hr in zip(alldates, yrs, mos, dys, hrs):
                try:
                    self.open_mask_file(yr, mo, dy, hr)
                    dict_dates[j] = i
                    j += 1
                    if yrorig != yr:
                        print(f"{yrorig} completed")
                        yrorig = yr
                except FileNotFoundError:
                    continue
            if savepath:
                with open(f'{savepath}/mcs_dict_{frequency}.pkl', 'wb') as handle:
                    pickle.dump(dict_dates, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if not savepath:
                return dict_dates
        
    def make_years(self):
        """
        Returns array of years for data generation.
        """
        years = pd.date_range(start=self.year_start+'-01-01', end=self.year_end+'-01-01', freq='AS-JAN').year
        return years
    
    def cesm_last_day(self, year, month):
        """
        Grab last day of the month without leap year option.
        Args:
            year (str): Year for opening file.
            month (str): Month for opening file.
        """
        last_day = monthrange(int(year),int(month))[1]
        if last_day == 29:
            return 28
        if last_day != 29:
            return last_day
    
    def open_variable_file(self, year=None, month=None, era_dir='e5.oper.an.sfc'):
        """
        Returns opened and sliced spatial region of data for respective year.
        Args:
            year (str): Year for opening file. Defaults to ``None``.
            month (str): Month for opening file. Defaults to ``None``.
            era_dir (str): Defaults to ``e5.oper.an.sfc``.
        """
        if self.ens_num == '002':
            if self.pressure:
                data = xr.open_dataset(
                    f'{self.main_directory}/plevs_FV_768x1152.bilinear.{self.make_month()}_b.e13.B20TRC5CN.ne120_g16.002_{self.variable}_{year}.nc')
                data = data.assign_coords(ncl4=("ncl4",data.time), ncl6=("ncl6",data.lat.values), ncl7=("ncl7",data.lon.values))
        if self.ens_num == '003':
            data = xr.open_dataset(
                f'{self.main_directory}/b.e13.B20TRC5CN.ne120_g16.003.cam.{self.inst_str}.{self.variable}.{year}010100Z-{year}123121Z.regrid.23x0.31.nc')
        if self.ens_num == 'era5':
            last_day = self.cesm_last_day(int(year),int(month))
            data = xr.open_mfdataset(
                f'{self.era5_directory}/{era_dir}/{year}{month}/*_{self.variable}.*.{year}{month}0100_{year}{month}{last_day}23.nc')
        return data
    
    def open_mask_file(self, year=None, month=None, day=None, hour=None):
        """
        Open the MCS mask file.
        
        Args:
            year (str): Year. Defaults to ``None``.
            month (str): Month. Defaults to ``None``.
            day (str): Day. Defaults to ``None``.
            hour (str): Hour. Defaults to ``None``.
        """
        if self.ens_num == '002':
            mask = xr.open_dataset(f"{self.main_directory}/mask_camPD_{self.make_month()}.nc")
        if self.ens_num == '003':
            mask = xr.open_dataset(f"{self.main_directory}/mask_camPD_{year}.nc")
        if self.ens_num == 'era5':
            mask = xr.open_dataset(f"{self.mcs_directory}/{year}0101_{year}1231/mcstrack_{year}{month}{day}_{hour}00.nc")
        return mask
    
    def slice_grid(self, data_mask, data, lat='lat', lon='lon'):
        """
        Slice the variable data on pressure level grid to match the mask data.

        Args:
            mask (xarray dataset or dataarray): MCS object mask.
            data (xarray dataset or dataarray): The data to slice.
            lat (str): Latitude dimension name. Defaults to ``lat``.
            lon (str): Longitude dimension name. Defaults to ``lon``.

        Returns:
            Variable data sliced to spatial extent of the mask data.
        """
        if self.ens_num == '002':
            data = data.isel(ncl5=self.height,
                             ncl6=slice(np.where(data[lat].values==data_mask['lat'][0].values)[0][0],
                                        np.where(data[lat].values==data_mask['lat'][-1].values)[0][0]+1),
                             ncl7=slice(np.where(data[lon].values==data_mask['lon'][0].values)[0][0],
                                        np.where(data[lon].values==data_mask['lon'][-1].values)[0][0]+1))
        if self.ens_num == '003':
            data = data.isel(lat=slice(np.where(data[lat].values==data_mask['lat'][0].values)[0][0],
                                       np.where(data[lat].values==data_mask['lat'][-1].values)[0][0]+1),
                             lon=slice(np.where(data[lon].values==data_mask['lon'][0].values)[0][0],
                                       np.where(data[lon].values==data_mask['lon'][-1].values)[0][0]+1))
        if self.ens_num == 'era5':
            lat0_bnd = int(np.around(data_mask['lat'].min(skipna=True).values))
            lat1_bnd = int(np.around(data_mask['lat'].max(skipna=True).values))
            lon0_bnd = int(np.around(data_mask['lon'].min(skipna=True).values))
            lon1_bnd = int(np.around(data_mask['lon'].max(skipna=True).values))
            data = data.isel(latitude=slice(np.where(data[lat].values==lat1_bnd)[0][0],
                                            np.where(data[lat].values==lat0_bnd)[0][0]+1),
                             longitude=slice(np.where(data[lon].values==lon0_bnd+360)[0][0],
                                             np.where(data[lon].values==lon1_bnd+360)[0][0]+1))
        return data
    
    def regrid_mask(self, ds, msk_var='cloudtracknumber', method='bilinear', 
                    lat_coord='lat', lon_coord='lon', 
                    offset=0.125, dcoord=0.25):
        """
        Function to regrid mcs obs mask onto coarser ERA5 grid (0.25-degree).
        Args:
            ds (xarray dataset): Mask file.
            msk_var (str): Variable name in mask file.
            method (str): Regrid method. Defaults to ``bilinear``.
            lat_coord (str): Latitude coordinate name in file. Defaults to ``lat``.
            lon_coord (str): Longitude coordinate name in file. Defaults to ``lon``.
            offset (float): Value to recenter grid by. Defaults to ``0.125``.
            dcoord (float): Distance between lat/lons. Defaults to ``0.25``.
        Returns:
            Regridded mask file for use with machine learning model.
        """
        lat0_bnd = int(np.around(ds[lat_coord].min(skipna=True).values))
        lat1_bnd = int(np.around(ds[lat_coord].max(skipna=True).values))
        lon0_bnd = int(np.around(ds[lon_coord].min(skipna=True).values))
        lon1_bnd = int(np.around(ds[lon_coord].max(skipna=True).values))
        ds_out = xe.util.grid_2d(lon0_b=lon0_bnd-offset, lon1_b=lon1_bnd+offset, d_lon=dcoord, 
                                 lat0_b=lat0_bnd-offset, lat1_b=lat1_bnd+offset, d_lat=dcoord)
        regridder = xe.Regridder(ds, ds_out, method)
        dr_out = regridder(ds[msk_var])
        return dr_out.fillna(0.0)

    def generate_files(self, pre_dict=False, dict_freq='3H', start_str=None, end_str=None, dictsave=None):
        """
        Save the files for each time period and variable with respective ID.
        Args:
            pre_dict (boolean): If ``True``, open pre-saved dictionary file of indices.
                                Defaults to ``False``.
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H``
                             for 3-hourly.
        """
        print("starting file generation...")
        yr_array = self.make_years()
        if not pre_dict:
            indx_array = self.make_dict(start_str=start_str, end_str=end_str, frequency=dict_freq, savepath=dictsave)
        if predict:
            with open(f'{self.main_directory}/mcs_dict_{dict_freq}.pkl', 'rb') as handle:
                indx_array = pickle.load(handle)
        print("opening variable file...")
            
        if self.ens_num == '002':
            #for yr in yr_array:
            #    data = self.open_variable_file(yr)
            #    if self.pressure:
            #        mask = self.open_mask_file()
            #        data = self.slice_grid(mask, data)
            #        print(f"{yr} opened and sliced successfully...")
            #        for t in data.ncl4:
            #            tmpdata = data.sel(ncl4=t)
            #            tmpdata = tmpdata.to_dataset()
            #            indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
            #            tmpdata.to_netcdf(
            #                f"{self.main_directory}/dl_files/plev_{self.variable}_hgt{self.height}_ID{indx_val}.nc")
            print('Training options not yet available for member 002')
            return
                        
        if self.ens_num == '003':
            for yr in yr_array:
                mask = self.open_mask_file(yr)
                data = self.open_variable_file(yr)
                data = self.slice_grid(mask, data)
                print(f"{yr} opened and sliced successfully...")
                for t in data.time:
                    tmpdata = data.sel(time=t)
                    tmpdata = tmpdata.to_dataset()
                    indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
                    tmpdata.to_netcdf(
                        f"{self.main_directory}/dl_files/file003_{self.inst_str}_{self.variable}_ID{indx_val}.nc")
                    
        if self.ens_num == 'era5':
            for indx_val, indx_dt in indx_array.items():
                mask = self.open_mask_file(year=indx_dt.strftime('%Y'), month=indx_dt.strftime('%m'), 
                                           day=indx_dt.strftime('%d'), hour=indx_dt.strftime('%H'))
                data = self.open_variable_file(year=indx_dt.strftime('%Y'), month=indx_dt.strftime('%m'), 
                                               era_dir='e5.oper.an.sfc')
                data = self.slice_grid(mask, data, lat='latitude', lon='longitude')
                # save indx
                tmpdata.to_netcdf(
                        f"{self.main_directory}/dl_files/{dict_freq}/file003_{self.variable}_ID{indx_val}.nc")
                return
                    
        print("Job complete!")

    def generate_masks(self, pre_dict=False, dict_freq='3H', start_str=None, end_str=None, dictsave=None):
        """
        Save the files for each time period and mask with respective ID.
        Args:
            pre_dict (boolean): If ``True``, open pre-saved dictionary file of indices.
                                Defaults to ``False``.
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H``
                             for 3-hourly.
            start_str and end_str (str): Start and end times for date range.
                For 002 use: start_str=f'04-01-1991', end_str=f'07-31-2005 23:00:00'
                For 003 use: start_str=f'01-01-2000 03:00:00', end_str=f'01-01-2006 00:00:00'
                For era5 use: start_str='2004-01-01 00:00:00', end_str='2016-12-31 23:00:00'
            dictsave (str): Path to save dictionary containing indices.
        """
        print("starting mask file generation...")
        yr_array = self.make_years()
        if not pre_dict:
            indx_array = self.make_dict(start_str=start_str, end_str=end_str, frequency=dict_freq, savepath=dictsave)
        if pre_dict:
            with open(f'{self.main_directory}/mcs_dict_{dict_freq}.pkl', 'rb') as handle:
                indx_array = pickle.load(handle)
        if self.ens_num == '002':
            print('Training options not yet available for member 002')
            return
        if self.ens_num == '003':
            for yr in yr_array:
                mask = self.open_mask_file(yr)
                for t in mask.time:
                    tmpmask = mask.sel(time=t)
                    indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
                    tmpmask.to_netcdf(f"{self.main_directory}/dl_files/mask_ID{indx_val}.nc")
        if self.ens_num == 'era5':
            for indx_val, indx_dt in indx_array.items():
                mask = self.open_mask_file(year=indx_dt.strftime('%Y'), month=indx_dt.strftime('%m'), 
                                           day=indx_dt.strftime('%d'), hour=indx_dt.strftime('%H'))
                tmpmask = self.regrid_mask(mask)
                
                tmpmask.to_netcdf(f"{self.main_directory}/dl_files/{dict_freq}/mask_ID{indx_val}.nc")
        print("Job complete!")
