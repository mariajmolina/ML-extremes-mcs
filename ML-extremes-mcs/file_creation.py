import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import calendar
import pickle
import itertools
from calendar import monthrange

class GenerateTrainData:
    """
    Class instantiation of GenerateTrainData:
    
    Here we will be preprocessing data for deep learning model training.
    The objective is to save files for each time index to subsequently load for training.
    IDs are assigned to each file.
    
    Attributes:
        main_path (str): Path where files are located.
        start_year (int): Start year of analysis. For era5 use 2004.
        end_year (int): Final year of analysis. For era5 use 2016.
        variable (str): Variable for file generation. Defaults to ``None``. Options for ``era5`` to 
                        ``2d``, ``2t``, ``10u``, ``10v``, and ``sp``.
        era5_directory (str): Location of ERA5 files in RDA (ds: 633.0).
        mcs_directory (str): Location of MCS obs generated using obs.
        mask_var (str): Variable name in mask file. Defaults to ``cloudtracknumber``. Options also include
                       ``pcptracknumber`` and ``pftracknumber``.
    """
    def __init__(self, main_path, start_year, end_year, variable=None,
                 era5_directory=None, mcs_directory=None, mask_var='cloudtracknumber'):
        """
        Initialization.
        """
        self.main_directory = main_path
        self.year_start = start_year
        self.year_end = end_year
        self.variable = variable
        
        if self.variable == 'w' or self.variable == 'u' or self.variable == 'v' or self.variable == 'z' or self.variable == 'q':
            self.era_dir='e5.oper.an.pl'
            
        if self.variable == '2d' or self.variable == '2t' or self.variable == 'sp' or self.variable == '10v' or self.variable == '10u' or self.variable == 'cape':
            self.era_dir='e5.oper.an.sfc'
            
        self.era5_directory = era5_directory
        self.mcs_directory = mcs_directory
        self.msk_var = mask_var

    def make_dict(self, start_str, end_str, frequency='3H', savepath=None):
        """
        Create dictionary of the indices using study time range.
        These indices must be fixed values for the study time range.
        
        Args:
            start_str and end_str (str): Start and end times for date range.
                For era5 use: start_str='2004-01-01 00:00:00', end_str='2019-12-31 23:00:00'
            frequency (str): Spacing for time intervals. Defaults to ``3H``.
            savepath (str): Path to save dictionary containing indices. Defaults to ``None``.
        """
        alldates = pd.date_range(start=start_str, end=end_str, freq=frequency)
        
        # cesm doesn't do leap years
        alldates = alldates[~((alldates.month == 2) & (alldates.day == 29))]
        dict_dates = {}

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
        return pd.date_range(start=str(self.year_start)+'-01-01', end=str(self.year_end)+'-01-01', freq='AS-JAN').year

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

    def open_variable_file(self, year=None, month=None, day=None):
        """
        Returns opened and sliced spatial region of data for respective year.
        
        Args:
            year (str): Year for opening file. Defaults to ``None``.
            month (str): Month for opening file. Defaults to ``None``.
            day (str): Day for opening file. Defaults to ``None``.
        """
        if self.era_dir == 'e5.oper.an.sfc':
            last_day = monthrange(int(year),int(month))[1]
            data = xr.open_mfdataset(
                f'{self.era5_directory}/{self.era_dir}/{year}{month}/*_{self.variable}.*.{year}{month}0100_{year}{month}{last_day}23.nc')
            
        if self.era_dir == 'e5.oper.an.pl':
            data = xr.open_mfdataset(
                f'{self.era5_directory}/{self.era_dir}/{year}{month}/*_{self.variable}.*.{year}{month}{day}00_{year}{month}{day}23.nc')
            
        return data
    
    def open_mask_file(self, year=None, month=None, day=None, hour=None, era_pctl='3pctl'):
        """
        Open the MCS mask file.
        
        Args:
            year (str): Year. Defaults to ``None``.
            month (str): Month. Defaults to ``None``.
            day (str): Day. Defaults to ``None``.
            hour (str): Hour. Defaults to ``None``.
            era_pctl (str): ERA5 mask precipitation percentile threshold. Defaults to ``3pctl``.
        """
        return xr.open_dataset(f"{self.mcs_directory}/mcstracking_{era_pctl}/{year}/mcstrack_{year}{month}{day}_{hour}00.nc")

    def slice_grid(self, data_mask, data, lat='latitude', lon='longitude'):
        """
        Slice the variable data on pressure level grid to match the mask data.
        
        Args:
            data_mask (xarray dataset or dataarray): MCS object mask.
            data (xarray dataset or dataarray): The data to slice.
            lat (str): Latitude dimension name for variable data. Defaults to ``latitude`` for era5.
            lon (str): Longitude dimension name for variable data. Defaults to ``longitude`` for era5.
            
        Returns:
            Variable data sliced to spatial extent of the mask data.
        """
        lat0_bnd = int(np.around(data_mask['lat'].min(skipna=True).values))
        lat1_bnd = int(np.around(data_mask['lat'].max(skipna=True).values))
        lon0_bnd = int(np.around(data_mask['lon'].min(skipna=True).values))
        lon1_bnd = int(np.around(data_mask['lon'].max(skipna=True).values))
        
        data = data.isel(latitude=slice(np.where(data[lat].values==lat1_bnd)[0][0],
                                        np.where(data[lat].values==lat0_bnd)[0][0]+1),
                         longitude=slice(np.where(data[lon].values==lon0_bnd)[0][0],
                                         np.where(data[lon].values==lon1_bnd)[0][0]+1))
        return data

    def regrid_mask(self, ds, method='nearest_s2d', lat_coord='lat', lon_coord='lon', 
                    offset=0.125, dcoord=0.25, reuse_weights=False):
        """
        Function to regrid mcs obs mask onto coarser ERA5 grid (0.25-degree).
        
        Args:
            ds (xarray dataset): Mask file.
            method (str): Regrid method. Defaults to ``nearest_s2d``. Options include 
                          'bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch'.
            lat_coord (str): Latitude coordinate name in file. Defaults to ``lat``.
            lon_coord (str): Longitude coordinate name in file. Defaults to ``lon``.
            offset (float): Value to recenter grid by. Defaults to ``0.125`` for 0.25-degree grid.
            dcoord (float): Distance between lat/lons. Defaults to ``0.25``.
            reuse_weights (boolean): Whether to use precomputed weights to speed up calculation.
                                     Defaults to ``False``.
        Returns:
            Regridded mask file for use with machine learning model.
        """
        lat0_bnd = int(np.around(ds[lat_coord].min(skipna=True).values))
        lat1_bnd = int(np.around(ds[lat_coord].max(skipna=True).values))
        lon0_bnd = int(np.around(ds[lon_coord].min(skipna=True).values))
        lon1_bnd = int(np.around(ds[lon_coord].max(skipna=True).values))
        
        ds_out = xe.util.grid_2d(lon0_b=lon0_bnd-offset, lon1_b=lon1_bnd+offset, d_lon=dcoord, 
                                 lat0_b=lat0_bnd-offset, lat1_b=lat1_bnd+offset, d_lat=dcoord)
        
        if method == 'conservative':
            latb = np.hstack([(ds['lat']-((ds['lat'][1]-ds['lat'][0])/2)),ds['lat'][-1]+((ds['lat'][1]-ds['lat'][0])/2)])
            lonb = np.hstack([(ds['lon']-((ds['lon'][1]-ds['lon'][0])/2)),ds['lon'][-1]+((ds['lon'][1]-ds['lon'][0])/2)])
            ds = ds.assign_coords({'lat_b':(latb),
                                   'lon_b':(lonb)})
            
        regridder = xe.Regridder(ds, ds_out, method, reuse_weights=reuse_weights)
        
        dr_out = regridder(ds[self.msk_var])
        
        return dr_out.fillna(0.0)
    
    def open_dictionary(self, dict_freq='3H'):
        """
        Open presaved dictionary containing file respective IDs.
        Args:
            pre_dict (boolean): If ``True``, open pre-saved dictionary file of indices. Defaults to ``True``.
        """
        with open(f'{self.main_directory}/mcs_dict_{dict_freq}.pkl', 'rb') as handle:
            indx_array = pickle.load(handle)
        return indx_array

    def generate_files(self, p_height=None, dict_freq='3H', file_indx=None):
        """
        Save the files for each time period and variable with respective ID.
        
        Args:
            p_height (int): Pressure height to grab variable. Defaults to ``None``.
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H`` for 3-hourly.
            file_indx (int): File integer to restart job from. Defaults to ``None``.
        """
        print("starting file generation...")
        indx_array = self.open_dictionary(dict_freq=dict_freq)
                
        # if restarting job of file creation
        if file_indx:
            indx_array = dict(itertools.islice(indx_array.items(), file_indx, len(indx_array), 1))
                
        print("opening variable file...")  # a random creation of a mask coarsened file for slicing era5 files
        mask = self.open_mask_file(year='2005', month='03', day='01', hour='03')
        
        for indx_val, indx_dt in indx_array.items():
            
            if self.era_dir == 'e5.oper.an.sfc':
                data = self.open_variable_file(year=indx_dt.strftime('%Y'), month=indx_dt.strftime('%m'))
                
            if self.era_dir == 'e5.oper.an.pl':
                data = self.open_variable_file(year=indx_dt.strftime('%Y'), 
                                               month=indx_dt.strftime('%m'), 
                                               day=indx_dt.strftime('%d'))
                
            tmpdata = self.slice_grid(mask, data, lat='latitude', lon='longitude')
            data = None
            
            if self.era_dir == 'e5.oper.an.pl':
                assert (p_height), "Please enter a pressure level to extract."
                tmpdata = tmpdata.sel(level=p_height)
                
            tmpdata = tmpdata.sel(time=indx_dt)
            
            if self.era_dir == 'e5.oper.an.sfc':
                tmpdata.to_netcdf(f"{self.main_directory}/dl_files/{dict_freq}/file_{self.variable}_ID{indx_val}.nc")
                
            if self.era_dir == 'e5.oper.an.pl':
                tmpdata.to_netcdf(f"{self.main_directory}/dl_files/{dict_freq}/file_{self.variable}{str(p_height)}_ID{indx_val}.nc")
        print("Job complete!")
        return

    def generate_masks(self, dict_freq='3H', file_indx=None):
        """
        Save the files for each time period and mask with respective ID.
        
        Args:
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H`` for 3-hourly.
            file_indx (int): File integer to restart job from. Defaults to ``None``.
        """
        print("starting mask file generation...")
        indx_array = self.open_dictionary(dict_freq=dict_freq)
        
        # if restarting job of file creation
        if file_indx:
            indx_array = dict(itertools.islice(indx_array.items(), file_indx, len(indx_array), 1))
        
        for indx_val, indx_dt in indx_array.items():
            
            mask = self.open_mask_file(year=indx_dt.strftime('%Y'), month=indx_dt.strftime('%m'), 
                                       day=indx_dt.strftime('%d'), hour=indx_dt.strftime('%H'))
            
            tmpmask = mask.isel(time=0)[self.msk_var].fillna(0.0)
            
        tmpmask.to_netcdf(f"{self.main_directory}/dl_files/{dict_freq}/mask_{self.msk_var}_ID{indx_val}.nc")
        print("Job complete!")
        return

    def generate_train_stats(self, variable, dict_freq='3H',
                             lat='latitude', lon='longitude', author=None):
        """
        Generate statistics for normalizing the training data prior to training ML.
        
        Args:
            variable (str): Variable used to name training file.
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H`` for 3-hourly.
            lat (str): Latitude coordinate name. Defaults to ``latitude``.
            lon (str): Longitude coordinate name. Defaults to ``longitude``.
            author (str): Author of file. Defaults to None.
        """
        indx_array = self.open_dictionary(dict_freq=dict_freq)
        
        ## add more variables here later as training variables change :)
        if variable == 'sp':
            VAR = 'SP'
        if variable == '10v':
            VAR = 'VAR_10V'
        if variable == '10u':
            VAR = 'VAR_10U'
        if variable == '2t':
            VAR = 'VAR_2T'
        if variable == '2d':
            VAR = 'VAR_2D'
        if variable == 'cape':
            VAR = 'CAPE'
        if variable == 'w700':
            VAR = 'W'
        if variable == 'u850' or variable == 'u500':
            VAR = 'U'
        if variable == 'v850' or variable == 'v500':
            VAR = 'V'
        if variable == 'z500':
            VAR = 'Z'
        if variable == 'q1000' or variable == 'q850':
            VAR = 'Q'
        
        assert (VAR), "Please enter an available variable."
        
        # initialize a blank list to use for storing mean values 
        mean_list = []
        stds_list = []
        maxs_list = []
        mins_list = []
        date_list = []
        
        for indx_val in indx_array.keys():
            
            # open and extract variable
            tmp = xr.open_dataset(f"{self.main_directory}/dl_files/{dict_freq}/file_{variable}_ID{indx_val}.nc")
            date_list.append(tmp['utc_date'].values)
            tmp = tmp[VAR]
            
            # compute stats for each file
            mean_list.append(tmp.mean(dim=[lat,lon], skipna=True).values)
            stds_list.append(tmp.std( dim=[lat,lon], skipna=True).values)
            maxs_list.append(tmp.max( dim=[lat,lon], skipna=True).values)
            mins_list.append(tmp.min( dim=[lat,lon], skipna=True).values)
        
        mean_array = np.array([mean_list]).squeeze()
        stds_array = np.array([stds_list]).squeeze()
        maxs_array = np.array([maxs_list]).squeeze()
        mins_array = np.array([mins_list]).squeeze()
        date_array = np.array([date_list]).squeeze()
        
        # create xarray dataset
        xarray_array = xr.Dataset({'averages': (['id'], mean_array),
                                   'sigma':    (['id'], stds_array),
                                   'maxs':     (['id'], maxs_array),
                                   'mins':     (['id'], mins_array),
                                   'dates':    (['id'], date_array)},
                                 coords = {'id': (['id'], np.array(list(indx_array.keys())))},
                                 attrs = {'File Author' : author})
        
        # save file
        xarray_array.to_netcdf(f'{self.main_directory}/dl_files/{dict_freq}/stats_era5_{variable}.nc')
        print(f"File saved and completed for {variable}")
        return
