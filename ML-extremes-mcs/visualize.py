import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cf
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from datetime import timedelta
from shapely.ops import unary_union
from shapely.prepared import prep
import pandas as pd
import calendar
import subprocess

def create_STATES(us_states_location):
    """
    Create shapely files of states.
    
    Args:
        us_states_location (str): Directory location of states shapefiles.
    Returns:
        States data as cartopy feature for plotting.
    """
    proj = ccrs.LambertConformal(central_latitude = 25, 
                                 central_longitude = 265, 
                                 standard_parallels = (25, 25))
    reader = shpreader.Reader(f'{us_states_location}/ne_50m_admin_1_states_provinces_lines.shp')
    states = list(reader.geometries())
    STATES = cfeature.ShapelyFeature(states, ccrs.PlateCarree())
    return STATES

def grid2gif(image_str, output_gif):
    """
    Create a gif using stitched images.
    
    Args:
        image_str (str): String of images and file saved location.
        output_gif (str): String of gif name and file save location.
    """
    str1 = 'convert -delay 8 -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)
    
def print_mask_info(data):
    """
    Print mask info, including locations of not null masks.
    
    Args:
        data (Xarray dataset): File containing masks.
    Returns:
        Printed statements of total number of times (3-hourly for CESM CAM files)
        containing mcss.
    """
    print("number of masks with an MCS:", np.argwhere(data.binary_tag.sum(axis=1).sum(axis=1).values!=0).shape[0])
    print("number of masks without an MCS:", np.argwhere(data.binary_tag.sum(axis=1).sum(axis=1).values==0).shape[0])
    print(np.argwhere(data.binary_tag.sum(axis=1).sum(axis=1).values!=0).reshape(-1))
    return

def create_003onelev_plot(data, data_mask, time_indx, variable, STATES, cmap='viridis', vmin=None, vmax=None):
    """
    Function to plot the variable data.
    These are the 003 member plots that are on one level (e.g., QBOT, VBOT).
    Dimensions of data variables must be named "time, lat, lon".
    
    Args:
        data (Xarray dataset): The file contianing CESM variable.
        data_mask (Xarray dataset): The file containing masks.
        time_indx (int): Integer of time index.
        variable (str): The name of the variable in the file.
        STATES (cartopy feature): US STATES file.
        cmap (str): Colormap name option from matplotlib.
        vmin (float): Set minimum for plot.
        vmax (float): Set maximum for plot.
    Returns:
        Plot of the respective variable.
    """
    fig = plt.figure(figsize=(6.,4.))
    ax = plt.axes([0.,0.,1.,1.], projection=ccrs.PlateCarree())
    data_isel = data.isel(
               time=time_indx,
               lat=slice(np.where(data.lat.values==data_mask.lat[0].values)[0][0],
                         np.where(data.lat.values==data_mask.lat[-1].values)[0][0]+1),
               lon=slice(np.where(data.lon.values==data_mask.lon[0].values)[0][0],
                         np.where(data.lon.values==data_mask.lon[-1].values)[0][0]+1))
    data_isel[variable].plot.pcolormesh(cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_feature(STATES, facecolor='none', edgecolor='k', zorder=30)
    ax.add_feature(cf.BORDERS)
    ax.margins(x=0,y=0)
    ax.coastlines()
    return plt.show()

def create_mask_plot(data, time_indx, STATES, cmap="Reds"):
    """
    Function to plot MCS masks.
    
    Args:
        data (Xarray dataset): The file containing masks.
        time_indx (int): Integer index of mask time.
        STATES (cartopy feature): US STATES file.
        cmap (str): Colormap name option from matplotlib.
    Returns:
        Plot of mask with state borders drawn for a respective time frame.
    """
    fig = plt.figure(figsize=(6.,4.))
    ax = plt.axes([0.,0.,1.,1.], projection=ccrs.PlateCarree())
    data.binary_tag.isel(time=time_indx).plot.pcolormesh(ax=ax, 
                                                         transform=ccrs.PlateCarree(), 
                                                         vmin=0, vmax=1, cmap=cmap)
    ax.add_feature(STATES, facecolor='none', edgecolor='k', zorder=30)
    ax.add_feature(cf.BORDERS)
    ax.margins(x=0,y=0)
    ax.coastlines()
    return plt.show()
