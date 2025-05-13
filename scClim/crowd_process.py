import datetime
import numpy as np
import sys
from pathlib import Path
import datetime as dt
from pyproj import Proj
import pandas as pd
import xarray as xr
import geopandas as gpd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from pyproj import Transformer
from scipy.spatial import cKDTree

from climada import CONFIG
sys.path.append(str(CONFIG.local_data.func_dir))
import scClim as sc
data_dir = str(CONFIG.local_data.data_dir)
crowd_url = str(CONFIG.crowd_url)


def per_category_scaler(X,min_dist_p_category=None):
    """Scale each feature in X by the maximum distance in this dimension which
    should still be considered as the same cluster in DBSCAN. Equivalent to
    eps parameter in DBSCAN, but separate for each dimension.

    Args:
        X (np.ndarray): Data to be scaled (n_samples, n_features)
        min_dist_p_category (np.array, optional): DBSCAN eps parameter for each dimension

    Returns:
        X_scaled (np.ndarray): scaled data
    """
    if min_dist_p_category is None:
        min_dist_p_category = np.ones(X.shape[1])


    X_scaled = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_scaled[:,i] = X[:,i]/min_dist_p_category[i]
    return X_scaled


def process_crowd_data(crowd_data,processed=True):
    """Processing raw crowd sourced data

    Args:
        crowd_data (pd.DataFrame): Dataframe of crowd sourced data
        processed (bool, optional): whether data is already pre processed. Defaults to True.

    Returns:
        crowd_data: pandas.DataFrame with crowd sourced data
    """

    #remove points with default locations
    crowd_data = crowd_data.loc[~((crowd_data['x']==717144) & (crowd_data['y']==95751))]
    crowd_data = crowd_data.loc[~((crowd_data['x']==537969) & (crowd_data['y']==152459))]
    crowd_data = crowd_data.loc[~((crowd_data['x']==683472) & (crowd_data['y']==247852))]

    #Convert time columns to datetime formate
    if processed:
        crowd_data['Time'] = pd.to_datetime(crowd_data['Time'])
        crowd_data['SubmissionTime'] = pd.to_datetime(crowd_data['SubmissionTime'])
        crowd_data['Timer'] = pd.to_datetime(crowd_data['Timer'])
    else:

        crowd_data = crowd_data.rename(columns={'Type':'size'})
        crowd_data = crowd_data.loc[((crowd_data['size']>=10) & (crowd_data['size']<=16))]
        crowd_data['Time'] = pd.to_datetime(crowd_data['Time'],unit='ms')
        crowd_data['SubmissionTime'] = pd.to_datetime(crowd_data['SubmissionTime'],unit='ms')

    #add columns for date, hailday, month and year
    crowd_data['days'] = crowd_data['Time'].dt.date
    crowd_data['hailday'] = (crowd_data['Time']-pd.Timedelta(hours=6)).dt.date #use -6h, to get haildays correctly! (1 hailday: 6UTC-6UTC)
    crowd_data['months'] = crowd_data['Time'].dt.to_period('M')
    crowd_data['years'] = crowd_data['Time'].dt.to_period('Y')


    #Convert hail size categories to size_mm, size_text, size_mm_text
    crowd_data = map_hailsize(crowd_data)


    #inProj = 'epsg:2056' # CH1903+ / LV95, see https://epsg.io/2056
    inProj = 'epsg:21781' # CH1903 / LV03, see https://epsg.io/21781
    outProj = 'epsg:4326' # WGS84, see https://epsg.io/4326

    transformer = Transformer.from_crs(inProj, outProj)
    coords_crowd = transformer.transform(crowd_data['x'],crowd_data['y'])

    crowd_data['Lat'] = coords_crowd[0].tolist()
    crowd_data['Lon'] = coords_crowd[1].tolist()
    return crowd_data

def map_hailsize(crowd_data,overwrite=True):

    conditions = [
        (crowd_data['size'] == 10),(crowd_data['size'] == 11),
        (crowd_data['size'] == 12),(crowd_data['size'] == 13),
        (crowd_data['size'] == 14),(crowd_data['size'] == 15),
        (crowd_data['size'] == 16),(crowd_data['size'] < 10)]

    # create a list of the values we want to assign for each condition
    value_text = ['No hail', '< coffee bean', 'Coffee bean', 'One franc coin',
                'Five franc coin', 'Golf ball', 'Tennis ball', "Old size category"]
    values_size = ['No hail', '>0-5 [mm]', '5-8 [mm]', '23 [mm]',
                '32 [mm]', '43 [mm]', '68 [mm]',"Old size category"]
    values_num = [0,2.5,6.5,23,32,43,68,np.nan]

    #map hail/no hail for both old and new categories
    condition_no_hail = ((crowd_data['size'] == 10) | (crowd_data['size'] == 0))
    if "no_hail" not in crowd_data.columns or overwrite:
        crowd_data['no_hail'] = condition_no_hail

    # create a new column and use np.select to assign values to it using our lists as arguments
    if "size_text" not in crowd_data.columns or overwrite:
        crowd_data['size_text'] = np.select(conditions, value_text,default="")
    if "size_mm_text" not in crowd_data.columns or overwrite:
        crowd_data['size_mm_text'] = np.select(conditions, values_size,default="")
    if "size_mm" not in crowd_data.columns or overwrite:
        crowd_data['size_mm'] = np.select(conditions, values_num,default=np.nan)

    return crowd_data



def setup_crowd_plot(dpi=200,relief=True,figsize=(11,10)):
    """Set up crowd sourced plot

    Args:
        dpi (int, optional): DPI of figure. Defaults to 200.
        relief (bool, optional): Whether to plot relief. Defaults to True.

    Returns:
        fig, ax: Figure and axes object
    """


    #plot data
    fig = plt.figure(figsize=figsize, dpi=dpi)
    prj = ccrs.AlbersEqualArea(8.222665776, 46.800663464)
    prj2 = ccrs.PlateCarree()
    ax = plt.axes(projection=prj)
    Ticino = [8.55, 9.4, 45.8, 46.3]
    Central = [7.6, 8.5, 46.5, 47.2]
    ZRH = [8.5,8.6,47.3,47.45]
    CH = [5.8, 10.7, 45.7, 47.95]

    extent = CH
    ax.set_extent(extent)

    if relief:
        plot_relief(ax,lakes=True)

    return fig, ax

def plot_relief(ax,lakes=True,borders=False):
        # read relief and data
        try:
            da_relief = xr.open_rasterio(data_dir+'/ch_shapefile/relief_georef_clipped_swiss.tif')
        except AttributeError: #AttributeError: module 'xarray' has no attribute 'open_rasterio'
            da_relief = xr.open_dataset(data_dir+'/ch_shapefile/relief_georef_clipped_swiss.tif',engine='rasterio').band_data

        # Compute the lon/lat coordinates with rasterio.warp.transform
        ny, nx = len(da_relief['y']), len(da_relief['x'])
        x, y = np.meshgrid(da_relief['x'], da_relief['y'])
        # Rasterio works with 1D arrays
        outProj = 'epsg:4326' # WGS84, see https://epsg.io/4326
        try:
            crs_relief = da_relief.crs
        except AttributeError: #AttributeError: 'Dataset' object has no attribute 'crs'
            crs_relief = '+init=epsg:21781'
        transformer = Transformer.from_crs(crs_relief, outProj)
        lat, lon = transformer.transform(x.flatten(), y.flatten())
        lon = np.asarray(lon).reshape((ny, nx))-0.01
        lat = np.asarray(lat).reshape((ny, nx))
        da_relief.coords['lon'] = (('y', 'x'), lon)
        da_relief.coords['lat'] = (('y', 'x'), lat)

        # get band
        da_relief = da_relief.isel(band=0, drop=True)
        da_relief = da_relief.where(da_relief > 1, drop=True)

        # add Relief
        da_relief.plot(ax=ax, x='lon', y='lat', cmap="Greys_r",
                    norm=colors.Normalize(vmin=110, vmax=255),
                    add_colorbar=False, transform=ccrs.PlateCarree())

        # let's load Lakes from Swisstopo
        gdf_lakes = gpd.read_file(data_dir+"/ch_shapefile/swiss-lakes-maps.json")



        # add lakes
        gdf_lakes.plot(ax=ax, edgecolor='none', color="cornflowerblue", transform=ccrs.PlateCarree())

        if borders:
            # add country border
            ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.5)
        # add urban areas
        #gdf_urban.boundary.plot(ax=ax,transform=prj2,zorder=1.4,linewidth=1,alpha=0.4, color='black')


def plot_crowd(meas_crowd,dpi=200,relief=True,figsize=(11,10),fig_ax=None,
               gridlines=True,zorder=4,cmap=None,legend=True,all_lables=False,
               **legend_kws):
    """Plotting function derived from Jerome Kopp

    Args:
        meas_crowd (pd.Dataframe): DF with crowd sourced data
        dpi (int, optional): DPI of figure. Defaults to 200.
        relief (bool, optional): Whether to plot relief. Defaults to True.
        figsize (tuple, optional): Figure size. Defaults to (11,10).
        fig_ax (tuple, optional): Tuple of figure and axes object. Defaults to None.
        gridlines (bool, optional): Whether to plot gridlines. Defaults to True.
        zorder (int, optional): Zorder of plot. Defaults to 4.
        cmap (str, optional): Colormap. Defaults to None.
        legend (bool, optional): Whether to plot legend. Defaults to True.
        all_lables (bool, optional): Whether to plot all labels (no matter whether or not the size categories appear). Defaults to False.
    Returns:
        fig,ax: Figure and axes object
    """

    #Set up figure
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = setup_crowd_plot(dpi=dpi,relief=relief,figsize=figsize)

    fs = 14

    # Prepare plotting of crowdsourced data
    if cmap is None:
        cmap = cm.get_cmap('Purples')
    meas_crowd = meas_crowd.loc[meas_crowd['size'].between(11,16)].sort_values('size')
    scat_kw = dict(marker='o', zorder=zorder, s=16, linewidth=0.4,
                   cmap=cmap ,edgecolors="purple", alpha=1, transform=ccrs.Geodetic(),
                   norm = plt.Normalize(11,16)) #10=no hail, 11= '<coffee bean', 16=tennis ball
    unique_sizes = np.sort(meas_crowd['size'].unique())
    size_map_dict = {11:'< coffee bean', 12:'Coffee bean', 13:'One franc coin',
                        14:'Five franc coin', 15:'Golf ball', 16:'Tennis ball'}

    #Plot data
    if 'Lon' in meas_crowd.columns:
        crowd = ax.scatter(meas_crowd['Lon'],meas_crowd['Lat'],c=meas_crowd['size'],**scat_kw)
    elif 'chx' in meas_crowd.columns:
        scat_kw.update({'transform':ccrs.epsg(2056)})
        crowd = ax.scatter(meas_crowd['chx'],meas_crowd['chy'],c=meas_crowd['size'],**scat_kw)

    if legend:

        if all_lables:
            figEmpty, axEmpty = plt.subplots(figsize=(1,1),subplot_kw={'projection': ccrs.PlateCarree()})
            plot = axEmpty.scatter(np.zeros(6), np.zeros(6), c=np.arange(11,16+1), **scat_kw)
            handles, labels = plot.legend_elements()
            labels = [size_map_dict[s] for s in np.arange(11,16+1)]
            plt.close(figEmpty)
        else:
            handles, labels = crowd.legend_elements()
            # labels = ['< coffee bean', 'Coffee bean', 'One franc coin', 'Five franc coin', 'Golf ball', 'Tennis ball']
            labels = [size_map_dict[s] for s in unique_sizes]


        for ha in handles:
            ha.set_markeredgecolor("purple")

        loc = legend_kws.pop('loc','upper left')
        legend1 = ax.legend(handles, labels, loc=loc, title="Report size",
                            prop={'size': fs-2}, framealpha=1, markerscale=2, **legend_kws)
        legend1.get_title().set_fontsize(fs)
        ax.add_artist(legend1)

    # format gridlines
    if gridlines:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray',
                        alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': fs}
        gl.ylabel_style = {'size': fs}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    plt.title('')
    return fig, ax

#%%

def grid_crowd_source_TS(crowd_source_data,pop_path,date_sel='20210628',
                      k_size=7,gauss_sigma=1.5,min_rep_density=0.001,min_rep=2,min_crowd_count=25,
                      w_pop=1,w_n_rep=1,size_agg='mean',rep_density_var = "rep_density_within_kernel",
                      use_empty_events=False,directional_interpolation=False):
    """Function to grid crowd-sourced data (Version by Timo)

    Args:
        crowd_source_data (Path,str,pd.DataFrame): dataframe of crowd-source reports
        pop_path (Path,str): path for population data source
        date_sel (str, optional): 'all' for all dates, otherwise '%Y%m%d'. Defaults to '20210628'.
        k_size (int, optional): size of kernel (must be odd). Defaults to 7.
        gauss_sigma (float, optional): sigma for gaussian kernel. Defaults to 1.5.
        min_rep_density (float, optional): minimum reporting density within kernel. Defaults to 0.001.
        min_rep (int, optional): minimum number of reports within kernel. Defaults to 2 (one report in addition to itself)
        min_crowd_count (int, optional): min count of crowd-source reports.
            Only used if date_sel='all'. Defaults to 100.
        w_pop (int, optional): weight for population. Defaults to 1.
        w_n_rep (int, optional): weight for number of reports. Defaults to 1.
        size_agg (str, optional): aggregation method for sizes within 1 gridcell (1km). Defaults to 'mean'.
        rep_density_var (str, optional): variable to use for filtering based on
          'min_rep_density' parameter. Defaults to "rep_density_within_kernel"
        use_empty_events (bool, optional): whether to use empty events (no reports).
            (To plot get emtpty fields for operational plots)
        directional_interpolation (bool, optional): whether to use directional
            interpolation. Defaults to False.

    Returns:
        ds: xr.Dataset of gridded crowd-sourced data
    """

    assert(k_size%2 == 1), 'kernel size must be an odd number'

    # read crowd source data
    if isinstance(crowd_source_data,(str,Path)):
        path = crowd_source_data
        crowd = pd.read_csv(path,sep=',')
    elif isinstance(crowd_source_data,pd.DataFrame):
        crowd = crowd_source_data

    #filter out filtered_out points
    if 'FILTEREDOUT' in crowd.columns and crowd['FILTEREDOUT'].sum()>0:
        raise UserWarning('Filtered out points detected. Please filter out before processing')

    #population pre-processing
    population = pd.read_csv(pop_path,sep=';')

    #Round coordinates to 1km resolution (center of 1km grid)
    population['chx'] = np.round(population['E_KOORD']//1000)*1000 +500
    population['chy'] = np.round(population['N_KOORD']//1000)*1000 +500

    pop_sum = population.groupby(['chx','chy']).agg({'B21BTOT': 'sum'})

    #Create meshgrid for population data
    chx_range = [2484, 2838]
    chy_range = [1073, 1299]
    chx = np.arange(chx_range[0],chx_range[1])*1000+500
    chy = np.arange(chy_range[0],chy_range[1])*1000+500

    # Assign population values to grid
    pop = np.zeros((len(chy),len(chx)))
    for i in range(len(pop_sum)):
        x_coord = chx==pop_sum.index.get_level_values('chx')[i]
        y_coord = chy==pop_sum.index.get_level_values('chy')[i]
        pop[y_coord, x_coord] = pop_sum.iloc[i]['B21BTOT']


    #crowd source pre processing
    crowd['Time'] = pd.to_datetime(crowd['Time'], format='%Y-%m-%d %H:%M:%S')
    # round to nearest minute
    crowd['Time'] = crowd['Time'].dt.round('min')

    #create copy of crowd dataframe with all values
    crowd_all = crowd.copy(deep=True)

    #Determine over which dates to loop
    if date_sel == 'all':
        # use -6h, to get haildays correctly (1 hailday: 6UTC-6UTC)
        crowd['hail_day'] = (crowd['Time'] - pd.Timedelta('6h')).dt.date
        grpy = crowd.groupby('hail_day').ID.count()
        #Select only dates with more than min_crowd_count reports
        sel_dates = grpy.index[grpy>min_crowd_count]
        #only select dates after 2017 (when reporting categories changed)
        dates = [d.strftime('%Y%m%d') for d in sel_dates if d.year >=2017]
    else:
        dates = [date_sel]

    #For operational purposes, create empty dataset
    if len(dates)==0 and use_empty_events:
        #create empty dataset (for operational plots)
        dates = ["20000101"]

    #loop over dates
    for date in dates:
        print(date)

        #Select data within the selected hail day (from 6UTC to 6UTC)
        date_plus_one = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime('%Y%m%d')
        datelist_range = [f'{date}0600',f'{date_plus_one}0600']
        crowd = crowd_all.loc[(crowd_all['Time'] >= datelist_range[0]) &
                              (crowd_all['Time'] <= datelist_range[1])]

        # # round to nearest km (for later groupby operation)
        crowd['chx'] = (crowd['chx'] // 1000) * 1000 + 500 #center of 1km grid
        crowd['chy'] = (crowd['chy'] // 1000) * 1000 + 500

        # group by chx, chy to speed up array loop
        crowd_count = crowd.groupby(['chx','chy']).agg({'size_mm': 'count'})
        if size_agg=='mean':
            crowd_size = crowd.groupby(['chx','chy']).agg({'size_mm': 'mean'})
        elif size_agg=='max':
            crowd_size = crowd.groupby(['chx','chy']).agg({'size_mm': 'max'})
        elif type(size_agg)==float:
            crowd_size = crowd.groupby(['chx','chy']).agg({'size_mm': lambda x: np.quantile(x,q=size_agg)})

        #Initialize arrays from crowdsourced hail size and number of reports
        crowd_sizes = np.zeros((len(chy),len(chx)))
        crowd_sizes[:] = np.nan
        crowd_counts = np.zeros((len(chy),len(chx)))

        # assign grouped point values to array
        for i in range(len(crowd_size)):
            x_coord = chx ==crowd_size.index.get_level_values('chx')[i]
            y_coord = chy ==crowd_size.index.get_level_values('chy')[i]
            crowd_sizes[y_coord, x_coord] = crowd_size.iloc[i]['size_mm']
            crowd_counts[y_coord, x_coord] = crowd_count.iloc[i]['size_mm']


        # Initalize additional variables
        crowd_estimate = {}
        crowd_estimate['size'] = crowd_sizes.copy()
        crowd_estimate['count'] = crowd_counts.copy()
        crowd_estimate['chx'] = chx
        crowd_estimate['chy'] = chy
        crowd_estimate['pop'] = pop
        crowd_estimate['size_raw'] = np.nan_to_num(crowd_sizes.copy(),nan=0)
        crowd_estimate['size_interp'] = np.zeros((len(chy),len(chx)))
        crowd_estimate['n_rep_within_kernel'] = np.zeros((len(chy),len(chx)))
        crowd_estimate['rep_density_within_kernel'] = np.zeros((len(chy),len(chx)))
        crowd_estimate['rep_density_within_kernel_weigh'] = np.zeros((len(chy),len(chx)))

        #Create gaussian kernel
        k_radius = k_size // 2 # 1 corresponds to 3x3 kernel, 2 to 5x5,
        gauss_kernel = np.fromfunction(lambda i,j: np.exp(-((i - k_size//2) ** 2 + (j - k_size//2) ** 2) / (2 * gauss_sigma ** 2)),(k_size,k_size))
        radius_arr = np.fromfunction(lambda i,j: np.sqrt((i - k_size//2) ** 2 + (j - k_size//2) ** 2),(k_size,k_size))
        gauss_kernel[radius_arr>k_radius] = 0 #set values outside of radius to 0
        #normalize gaussian kernel
        gauss_kernel = gauss_kernel/gauss_kernel.sum()

        #If directional interpolation is used, calculate the angle of movement
        if directional_interpolation:
            max_dist=5000
            crowd_estimate['angle'] = np.zeros((len(chy),len(chx)))
            tree = cKDTree(crowd[['chx', 'chy']])
            df_cluster_all = sc.crowd_process.create_hailswath_dataframe(crowd)

        #Loop over grid cells
        for i in np.arange(k_radius,crowd_estimate['size_raw'].shape[0]-k_radius): # loop over y (skip edges)
            for j in np.arange(k_radius,crowd_estimate['size_raw'].shape[1]-k_radius): # loop over x

                #Population weight
                weight_pop = crowd_estimate['pop'][i-k_radius:i+k_radius+1,j-k_radius:j+k_radius+1] # always +1 because of python indexing
                #Number of reports weight
                weight_n_rep = crowd_estimate['count'][i-k_radius:i+k_radius+1,j-k_radius:j+k_radius+1]

                #Calculate the reporting density
                crowd_estimate['n_rep_within_kernel'][i,j] = np.sum(weight_n_rep)
                crowd_estimate['rep_density_within_kernel'][i,j] = np.nan if np.sum(weight_pop)==0 else np.sum(weight_n_rep)/np.sum(weight_pop)
                crowd_estimate['rep_density_within_kernel_weigh'][i,j] = np.nan if np.sum(weight_pop*gauss_kernel)==0 else np.sum(weight_n_rep*gauss_kernel)/np.sum(weight_pop*gauss_kernel)

                #Normalize weights
                weight_pop_norm = np.zeros_like(weight_pop) if np.sum(weight_pop)==0 else weight_pop/np.sum(weight_pop)
                weight_n_rep_norm = np.zeros_like(weight_n_rep) if np.sum(weight_n_rep)==0 else weight_n_rep/np.sum(weight_n_rep)

                #Calculate total weight
                weight_total = (w_pop*weight_pop_norm + w_n_rep*weight_n_rep_norm) * gauss_kernel

                #Directional interpolation (not used in published version)
                if directional_interpolation and df_cluster_all.shape[0]>0:
                    y = crowd_estimate['chy'][i]
                    x = crowd_estimate['chx'][j]
                    dist,idx = tree.query((x,y))
                    if dist < max_dist:
                        #get the average time of all reports within 5km
                        dist = np.sqrt((crowd['chx']-x)**2 + (crowd['chy']-y)**2)
                        #select cluster with most reports
                        sel_cluster = crowd['cluster'][dist<max_dist].value_counts().index[0]
                        df_cluster_sel = df_cluster_all[df_cluster_all['cluster']==sel_cluster]

                        #get the nearest centroid of the selected cluster
                        if df_cluster_sel.shape[0]>0:
                            tree2 = cKDTree(df_cluster_sel[['chx', 'chy']])
                            dist,idx = tree2.query((x,y))

                            #get the movement angle of the nearest centroid
                            angle = df_cluster_sel.iloc[idx]['dir_running']
                            crowd_estimate['angle'][i,j] = angle

                            if ~np.isnan(angle):
                                #stretch gauss kernel in direction of angle
                                stretched_gauss = sc.E.stretch_array_any_direction2(gauss_kernel,angle)
                                stretched_gauss = stretched_gauss/stretched_gauss.sum()

                                #Overwrite weight_total and rep_density_within_kernel_weigh
                                weight_total = (w_pop*weight_pop_norm + w_n_rep*weight_n_rep_norm) * stretched_gauss
                                crowd_estimate['rep_density_within_kernel_weigh'][i,j] = np.nan if np.sum(weight_pop*stretched_gauss)==0 else np.sum(weight_n_rep*stretched_gauss)/np.sum(weight_pop*stretched_gauss)

                #Interpolate hail size with weighted kernel
                if weight_total.sum() == 0:
                    crowd_estimate['size_interp'][i,j] = 0
                else:
                    crowd_estimate['size_interp'][i,j] = np.average(crowd_estimate['size_raw'][i-k_radius:i+k_radius+1,j-k_radius:j+k_radius+1],
                                                                   weights=weight_total)
                #np.nanmean(crowd_estimate['size'][i-kernel_size:i+kernel_size,j-kernel_size:j+kernel_size])

        # -------------- convert to xarray DataSet ----------------
        #Define data variables
        data_vars = {'h_raw': (['chy', 'chx'], crowd_estimate['size']),
                    'n_rep': (['chy', 'chx'], crowd_estimate['count']),
                    'pop': (['chy', 'chx'], crowd_estimate['pop']),
                    'h_smooth': (['chy', 'chx'], crowd_estimate['size_interp']),
                    'n_rep_within_kernel': (['chy', 'chx'], crowd_estimate['n_rep_within_kernel']),
                    'rep_density_within_kernel': (['chy', 'chx'], crowd_estimate['rep_density_within_kernel']),
                    'rep_density_within_kernel_weigh': (['chy', 'chx'], crowd_estimate['rep_density_within_kernel_weigh'])}

        if directional_interpolation:
            data_vars['angle'] = (['chy', 'chx'], crowd_estimate['angle'])

        #Create xarray DataSet
        ds = xr.Dataset(data_vars=data_vars,
                    coords={'chx': (['chx'], crowd_estimate['chx']),'chy': (['chy'], crowd_estimate['chy'])})
        #Set values below min_rep_density and min_rep to 0
        ds['h_smooth'] = ds['h_smooth'].where((ds[rep_density_var]>min_rep_density)& (ds['n_rep_within_kernel']>=min_rep),0)

        ds = ds.expand_dims({'time': [dt.datetime.strptime(date, '%Y%m%d')]})

        #Merge different dates
        if date == dates[0]:
            ds_all = ds.copy(deep=True)
        else:
            ds_all = xr.concat([ds_all, ds], dim='time')

    ## Assign lat/lon coordinates
    projdef = ccrs.epsg(2056)
    #create meshgrid
    meshX,meshY = np.meshgrid(ds_all.chx, ds_all.chy)
    p = Proj(projdef)
    lon, lat = p(meshX,meshY, inverse=True)
    ds_all=ds_all.assign_coords({'lon':(('chy','chx'),lon)})
    ds_all=ds_all.assign_coords({'lat':(('chy','chx'),lat)})

    return ds_all




def create_hailswath_dataframe(crowd, min_dist_m=10e3, min_reports=30, dt_sec=10*60,
                               running_anlge_windowsize=3,max_angle_diff=45):
    """calculate direction of each hailswath of crouwdsourced reports

    Args:
        crowd (pd.DataFrame): Crowdsourced reports clustered with DBSCAN
        min_dist_m (float, optional): Minimum distance of hail swath to be considered. Defaults to 10e3.
        min_reports (int, optional): minimum number of reports per hail swath to be considered. Defaults to 30.
        dt_sec (float, optional): Timesteps in seconds. Defaults to 10*60.
        running_anlge_windowsize (int): windowsize for running angle calculation (must be uneven)
        max_angle_diff (int): Maximum angle difference to previous timestep to be considered (in running angle)

    Returns:
        df_cluster_all: DataFrame with direction of each hailswath. Each timestep is one row.
    """
    df_cluster_all = pd.DataFrame(dtype=float)
    df_columns = ['cluster','chx','chy','dx','dy','dx_running','dy_running','n_rep',
                'avg_direction','avg_dir_weigh','dx_avg_weigh','dy_avg_weigh',
                'dir_running']

    #Loop over each cluster
    for i,cluster in enumerate(crowd.cluster.value_counts().index):
        sel = crowd[crowd.cluster==cluster]

        #Cluster with less than min_reports are not considered
        if sel.shape[0]<min_reports:
            continue

        #Loop over each  timestep (default = 10min)
        tmin = sel.time_int.min()
        tmax = sel.time_int.max()
        df_cluster = pd.DataFrame(index = np.arange(tmin,tmax,dt_sec),columns=df_columns,dtype=float)
        df_cluster['cluster'] = cluster
        for t in df_cluster.index:
            sel_t = sel.loc[(sel.time_int>=t) & (sel.time_int<t+dt_sec)]

            #get average location (chx,chy) of all reports in this timestep
            df_cluster.loc[t,'chx'] = sel_t.chx.mean()
            df_cluster.loc[t,'chy'] = sel_t.chy.mean()
            df_cluster.loc[t,'n_rep'] = sel_t.shape[0]

        #Calculate dx,dy for each timestep
        df_cluster['dx'] = df_cluster['chx'].diff()
        df_cluster['dy'] = df_cluster['chy'].diff()
        #cluster with a start-to-end distance below min_dist (in m) are not considered
        if np.sqrt(df_cluster.dx.sum()**2 + df_cluster.dy.sum()**2) < min_dist_m:
            continue

        #Average direction
        dx_avg = df_cluster['dx'].sum()
        dy_avg = df_cluster['dy'].sum()
        df_cluster['avg_direction'] =   np.arctan2(dy_avg,dx_avg)*180/np.pi

        #Weighted average direction (by how many reports are in each 10min timestep)
        df_cluster['dx_avg_weigh'] = np.average(a=df_cluster['dx'][1:],weights = (df_cluster['n_rep']+df_cluster['n_rep'].shift(1))[1:])
        df_cluster['dy_avg_weigh']  = np.average(a=df_cluster['dy'][1:],weights = (df_cluster['n_rep']+df_cluster['n_rep'].shift(1))[1:])
        df_cluster['avg_dir_weigh'] = np.arctan2(df_cluster['dy_avg_weigh'] ,df_cluster['dx_avg_weigh'])*180/np.pi

        #Running direction (weighted average of 3 timesteps)
        assert running_anlge_windowsize % 2 == 1
        half_wsize = running_anlge_windowsize//2
        for t in df_cluster.index[:]:
            idx = df_cluster.index.get_loc(t)
            idx_min_run = max(1,idx-half_wsize) #must be at least 1, because the first timestep does not have a delta x/y yet (diff)
            idx_max_run = idx+half_wsize+1 #second +1 is because of python indexing

            #for first timestep consider an average until +2, otherwise only one
            # delta x/y would be considered, which may lead to unstable directions
            if t == df_cluster.index[0] and half_wsize==1:
                idx_max_run +=1

            # calculate running average
            dx_run = np.average(a=df_cluster['dx'][idx_min_run:idx_max_run], #average from -1 to +1
                                    weights = (df_cluster['n_rep']+df_cluster['n_rep'].shift(1))[idx_min_run:idx_max_run])
            dy_run = np.average(a=df_cluster['dy'][idx_min_run:idx_max_run],
                                        weights = (df_cluster['n_rep']+df_cluster['n_rep'].shift(1))[idx_min_run:idx_max_run])
            df_cluster.loc[t,'dir_running'] = np.arctan2(dy_run,dx_run)*180/np.pi
            df_cluster.loc[t,'dx_running'] = dx_run
            df_cluster.loc[t,'dy_running'] = dy_run

        #Filter out directions with a difference of more than max_angle_diff to the previous timestep
        remove_sel = df_cluster.dir_running.diff().abs()>max_angle_diff
        df_cluster.loc[remove_sel,'dir_running'] = np.nan

        df_cluster_all = pd.concat([df_cluster_all,df_cluster])

    # (Optional) Calculate further summary statistics
    # df_cluster_all['magnitude'] = np.sqrt(df_cluster_all['dx']**2 + df_cluster_all['dy']**2)
    # df_cluster_all['dx_norm'] = df_cluster_all['dx']/df_cluster_all['magnitude']
    # df_cluster_all['dy_norm'] = df_cluster_all['dy']/df_cluster_all['magnitude']
    # df_cluster_all['direction'] = np.arctan2(df_cluster_all['dy'],df_cluster_all['dx'])*180/np.pi

    return df_cluster_all