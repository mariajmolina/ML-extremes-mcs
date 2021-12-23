import numpy as np
import calendar
from datetime import timedelta
import subprocess
import matplotlib.pyplot as plt
import cartopy.feature as cf
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
import pandas as pd
import xarray as xr


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
    reader = shpreader.Reader(
        f'{us_states_location}/ne_50m_admin_1_states_provinces_lines.shp')
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


def create_mask_plot(data, time_indx, STATES, cmap="Reds", lat='lat', lon='lon'):
    """
    Function to plot MCS masks.
    ** Works with original tempestextremes files.
    
    Args:
        data (Xarray dataset): The file containing masks.
        time_indx (int): Integer index of mask time.
        STATES (cartopy feature): US STATES file.
        cmap (str): Colormap name option from matplotlib. Defaults to ``Reds``.
    
    Returns:
        Plot of mask with state borders drawn for a respective time frame.
    """
    fig = plt.figure(figsize=(6.,4.))
    ax = plt.axes([0.,0.,1.,1.], projection=ccrs.PlateCarree())
    data.binary_tag.isel(time=time_indx).plot.pcolormesh(lon, lat,
                                                         ax=ax, 
                                                         transform=ccrs.PlateCarree(), 
                                                         vmin=0, vmax=1, cmap=cmap)
    ax.add_feature(STATES, facecolor='none', edgecolor='k', zorder=30)
    ax.add_feature(cf.BORDERS)
    ax.margins(x=0,y=0)
    ax.coastlines()
    return plt.show()

        
def create_training_plot(data, variable, STATES, lat='lat', lon='lon',
                         cmap='viridis', vmin=None, vmax=None, 
                         dpi=200, savedir=None, indx=0):
    """
    Function to plot the variable data already processed for training.
    ** Works with processed tempestextremes MASKs (with ID nums) and 
    ** flextrkr files (if .isel(time=0) and variable=cloudtracknumber).
    
    Args:
        data (Xarray dataset): The file contianing CESM variable.
        variable (str): The name of the variable in the file.
        STATES (cartopy feature): US STATES file.
        cmap (str): Colormap name option from matplotlib. Defaults to ``viridis``.
        vmin (float): Set minimum for plot. Defaults to ``None``.
        vmax (float): Set maximum for plot. Defaults to ``None``.
    
    Returns:
        Plot of the respective variable.
    """
    fig = plt.figure(figsize=(6.,4.))
    ax = plt.axes([0.,0.,1.,1.], projection=ccrs.PlateCarree())
    data[variable].plot.pcolormesh(lon, lat, ax=ax, transform=ccrs.PlateCarree(), 
                                   cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs={'extend': 'both'})
    ax.add_feature(STATES, facecolor='none', edgecolor='k', zorder=30)
    ax.add_feature(cf.BORDERS)
    ax.margins(x=0,y=0)
    ax.coastlines()
    
    if not savedir:
        return plt.show()
    
    if savedir:
        plt.savefig(f"{savedir}/trainplot_{variable}_{indx}.png", bbox_inches='tight', dpi=dpi)
        plt.close()

        
def create_mcs_stat_plot(data, STATES, lat, lon, vmin=0, vmax=10, cmap='BuPu', title=None):
    """
    Function to plot MCS stats on maps.
    ** Works with nontracked_total_grid and tracked_total_grid.
    
    Args:
        data (2d numpy array): The file containing stats.
        STATES (cartopy feature): US STATES file.
        lat (1d or 2d numpy array): Latitude.
        lon (1d or 2d numpy array): Longitude.
        vmin (int): Minimum value for plot. Defaults to ``0``.
        vmax (int): Maximum value for plot. Defaults to ``10``.
        cmap (str): Colormap name option from matplotlib. Defaults to ``BuPU``.
        title (float, int, or str): Title for plot. Defaults to ``None``.
    """
    fig = plt.figure(figsize=(6.,4.))
    fig, axes = plt.subplots(figsize=(6.,4.), nrows=4, ncols=4, sharex=True, sharey=True)
    ax = plt.axes([0.,0.,1.,1.], projection=ccrs.PlateCarree())
    
    if title:
        ax.set_title(title)
        
    ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.add_feature(STATES, facecolor='none', edgecolor='k', zorder=30)
    ax.add_feature(cf.BORDERS)
    ax.margins(x=0,y=0)
    ax.coastlines()
    return plt.show()


def create_mcs_stat_figure(data, STATES, lat, lon, vmin=0, vmax=10, cmap='BuPu', title=None, nrows=5, ncols=5):
    """
    Function to plot MCS stats on maps.
    ** Works with nontracked_total_grid and tracked_total_grid.
    
    Args:
        data (list of 2d numpy array): The files containing stats.
        STATES (cartopy feature): US STATES file.
        lat (1d or 2d numpy array): Latitude.
        lon (1d or 2d numpy array): Longitude.
        vmin (int): Minimum value for plot. Defaults to ``0``.
        vmax (int): Maximum value for plot. Defaults to ``10``.
        cmap (str): Colormap name option from matplotlib. Defaults to ``BuPU``.
        title (float, int, or str): Title for plot. Defaults to ``None``.
    """
    fig, axes = plt.subplots(figsize=(6.,6.), nrows=nrows, ncols=ncols, sharex=True, sharey=True, projection=ccrs.PlateCarree())
    #ax = plt.axes([0.,0.,1.,1.], projection=ccrs.PlateCarree())
    
    for i, ax in enumerate(axes.flat):
        
        #if title:
        #    ax.set_title(title)
        ax.pcolormesh(lon, lat, data[i], transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.add_feature(STATES, facecolor='none', edgecolor='k', zorder=30)
        ax.add_feature(cf.BORDERS)
        ax.margins(x=0,y=0)
        ax.coastlines()
        
    return plt.show()
