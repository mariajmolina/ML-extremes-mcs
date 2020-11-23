from config import us_states_location
from visualize import create_STATES
from id_selector import IDSelector
import mcs_stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
from config import main_path_era, savefig_path

######################################################################
######################################################################
##############
############## Generate MCS stats and figs
##############
######################################################################
######################################################################

def iterate_mcs_stats(month, years, dictpath, mcspath, ens_num='era5', msk_var='cloudtracknumber'):
    """
    Iterate the MCS stats.
    Args:
        month (int): Select one month.
        years (int): List of years to iterate over. 
        dictpath (str): Path to location of dictionary.
        mcspath (str): Path to location of MCSs.
        ens_num (str): The dataset ensemble choice. Defaults to ``era5``.
        msk_var (str): Mask file variable. 
    """
    # create blank lists
    a = []; b = []; c = []; d = []; e = []
    # first loop return lat lon
    return_coords = True
    # loop over years
    for i, yr in enumerate(years):
        # instantiate id selector class
        ids_ = IDSelector(main_path = dictpath, 
                          start_year = 2004, 
                          end_year = 2016, 
                          month_only = [month], year_only = [yr], mcs_only = False, 
                          percent_train = 0.7, ens_num = ens_num)
        # grab ids for corresponding month and year and other settings
        IDlist = ids_.generate_IDarray(pre_dict=True, dict_freq='3H', start_str=None, end_str=None, dictsave=None)
        # instantiate MCS stats class object
        mcsstat = mcs_stats.MCSstats(IDlist=IDlist, mcs_path=mcspath, msk_var=msk_var)
        # return masks (and lat lon once)
        if not return_coords:
            masks = mcsstat.open_masks(return_coords=return_coords)
        if return_coords:
            masks, lat, lon = mcsstat.open_masks(return_coords=return_coords)
        # go thru mcs stat options
        a.append(mcsstat.nontracked_total_grid(masks))
        b.append(mcsstat.tracked_total_grid(masks))
        c.append(mcsstat.tracked_total(masks))
        d.append(mcsstat.total_with_mcs(masks))
        e.append(mcsstat.percentage_with_mcs(d[i]))
        # remove request to return lat lon
        return_coords = False
    return a, b, c, d, e, lat, lon

def create_mcs_stat_figure(data, STATES, lat, lon, vmin=0, vmax=10, cmap='BuPu', 
                           nrows=4, ncols=3, titles=None, suptitle=None, savefig=None):
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
        nrows (int): Number of rows for subplots. Defaults to ``4``.
        ncols (int): Number of columns for subplots. Defaults to ``3``.
        titles (float, int, or str): List of titles for plot. Defaults to ``None``.
        suptitle (str): Title for full figure. Defaults to ``None``.
        savefig (str): Directory and name of figure for saving. Defaults to ``None``.
    """
    if not titles:
        titles = np.zeros(nrows*ncols)
    # create fig
    fig, axes = plt.subplots(figsize=(7.,6.5), nrows=nrows, ncols=ncols, sharex=True, sharey=True, 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    # create plots
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.95)
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        if title:
            ax.set_title(title, fontsize=10)
        ax.pcolormesh(lon, lat, data[i], transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.add_feature(STATES, facecolor='none', edgecolor='k', zorder=30)
        ax.add_feature(cf.BORDERS)
        ax.margins(x=0,y=0)
        ax.coastlines()
    # cbar
    cbar_ax = fig.add_axes([0.345, 0.1, 0.3, 0.0125])
    bounds = [vmin,vmax*0.5,vmax]
    newnorm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=plt.cm.get_cmap(cmap),
                                     norm=newnorm,
                                     ticks=bounds,
                                     orientation='horizontal')
    cbar.set_label('Frequency', fontsize=12) 
    cbar.ax.tick_params(labelsize=12)
    # save fig
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=200)
        return plt.show()
    if not savefig:
        return plt.show()

def main():
    """
    Run statistics and generate associated plots for MCSs.
    """
    ############## set variables
    dictpath = main_path_era
    mcspath = f'{main_path_era}/dl_files/3H/'
    STATES = create_STATES(us_states_location)
    theyears = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    ############## APRIL
    a, b, c, d, e, lat, lon = iterate_mcs_stats(4, theyears, dictpath=dictpath, mcspath=mcspath)
    create_mcs_stat_figure(b, STATES, lat, lon, vmin=0, vmax=15, cmap='BuPu', suptitle='April', titles=theyears,
                           savefig=f'{savefig_path}/april_mcs_flextrkr2.png')
    print(f"Percent of month with MCS: {np.array(e)}")
    plt.bar(theyears, c); plt.title('April'); 
    plt.savefig(f'{savefig_path}/april_mcs_bar.png', bbox_inches='tight', dpi=200); plt.show()
    ############## MAY
    a, b, c, d, e, lat, lon = iterate_mcs_stats(5, theyears, dictpath=dictpath, mcspath=mcspath)
    create_mcs_stat_figure(b, STATES, lat, lon, vmin=0, vmax=15, cmap='BuPu', suptitle='May', titles=theyears,
                           savefig=f'{savefig_path}/may_mcs_flextrkr2.png')
    print(f"Percent of month with MCS: {np.array(e)}")
    plt.bar(theyears, c); plt.title('May'); 
    plt.savefig(f'{savefig_path}/may_mcs_bar.png', bbox_inches='tight', dpi=200); plt.show()
    ############## JUNE
    a, b, c, d, e, lat, lon = iterate_mcs_stats(6, theyears, dictpath=dictpath, mcspath=mcspath)
    create_mcs_stat_figure(b, STATES, lat, lon, vmin=0, vmax=15, cmap='BuPu', suptitle='June', titles=theyears,
                           savefig=f'{savefig_path}/june_mcs_flextrkr2.png')
    print(f"Percent of month with MCS: {np.array(e)}")
    plt.bar(theyears, c); plt.title('June'); 
    plt.savefig(f'{savefig_path}/june_mcs_bar.png', bbox_inches='tight', dpi=200); plt.show()
    ############## JULY
    a, b, c, d, e, lat, lon = iterate_mcs_stats(7, theyears, dictpath=dictpath, mcspath=mcspath)
    create_mcs_stat_figure(b, STATES, lat, lon, vmin=0, vmax=15, cmap='BuPu', suptitle='July', titles=theyears,
                           savefig=f'{savefig_path}/july_mcs_flextrkr2.png')
    print(f"Percent of month with MCS: {np.array(e)}")
    plt.bar(theyears, c); plt.title('July'); 
    plt.savefig(f'{savefig_path}/july_mcs_bar.png', bbox_inches='tight', dpi=200); plt.show()

if __name__ == "__main__":
    main()
