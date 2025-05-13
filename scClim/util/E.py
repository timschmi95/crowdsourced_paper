"""
Utility function for subproject E (Timo)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import pandas as pd
from climada.engine import Impact
from scipy import sparse
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import random
import cartopy.crs as ccrs
import datetime as dt
import sys
from climada import CONFIG
sys.path.append(str(CONFIG.local_data.func_dir))
import scClim as sc
import xarray as xr
import json
from scipy.ndimage import affine_transform



def stretch_array_any_direction2(arr, angle, primary_factor=1.5, secondary_factor=1/1.5):
    """
    Stretch a 2D numpy array along a given direction (angle) with two defined factors.

    Parameters:
    - arr: Input n x n 2D numpy array
    - angle: Direction to stretch in degrees (0 = horizontal, 90 = vertical, etc.)
    - primary_factor: Stretching factor along the specified angle.
    - secondary_factor: Stretching factor perpendicular to the specified angle.

    Returns:
    - Stretched 2D numpy array
    """
    # Convert angle to radians
    theta = np.deg2rad(angle)

    # Create the affine transformation matrix for rotation (clockwise to align x-axis with the angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Rotation matrix for counterclockwise rotation
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Stretching matrix (applies scaling along x-axis for primary_factor and y-axis for secondary_factor)
    stretch_matrix = np.array([[1/secondary_factor, 0],
                               [0, 1/primary_factor]])

    # Inverse of the rotation matrix to rotate back after stretching
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    # The combined affine transformation matrix (rotate -> stretch -> rotate back)
    combined_matrix = inv_rotation_matrix @ stretch_matrix @ rotation_matrix

    # Calculate the center of the image
    # Assert that the array has an uneven number of rows and columns (and is square)
    assert arr.shape[0] % 2 == 1 and arr.shape[0] == arr.shape[1]
    center = np.array(arr.shape) // 2

    # Apply affine transformation with the computed matrix and center
    transformed_array = affine_transform(arr, combined_matrix, offset=center - center @ combined_matrix)

    return transformed_array


def js_r(filename: str):
    """Read json file into dictionary

    Args:
        filename (str): path

    Returns:
        dict: json content as dictionary
    """
    with open(filename) as f_in:
        return json.load(f_in)

def get_hazard_files_TS(haz_var,years,event_def_version,data_dir):
    """get Hazard files (specific to subproject E, Timo's PC)

    Args:
        haz_var (str): hazard
        years (np.array): considered years
        event_def_version (int): Event definition version
        data_dir (str,path): data directory

    Raises:
        NotImplementedError: for haz_var= HKE

    Returns:
        paths (paths or xr.Dataset): Data (or path) to use in get_hazard_from_radar()
    """
    if haz_var == 'MESHS' or haz_var == 'MESHS_smooth' or haz_var=='MESHS_4km' or haz_var == 'MESHS_20to23':
        # for event_definition version 5, the old POH values were used, which
        # are only 6-0UTC for 2021, and have a shifted grid
        if event_def_version == 5:
            raise DeprecationWarning('event_def_version 5 is deprecated')
            # paths = [data_dir+'/V0/MESHS/MZC/MZC_X1d66_%d.nc' %
            #         year for year in years]
        else:
            paths = [data_dir+'/V5/MZC/MZC_X1d66_%d.nc' % year for year in years]

        if haz_var == 'MESHS_4km':
            paths = xr.open_dataset(data_dir+'/V5/MZC_4km/MZC_4km_2002_2021.nc')
    elif haz_var == 'MESH':
        paths = [data_dir+'/V5/SHI/SHI_6t6_%d.nc' % year for year in years]
    elif haz_var == 'MESHS_opt_corr':
        paths = [f"{data_dir}/V5/MZC/shifted/MESHS_corr_opt_min10.nc"]
    elif 'MESHSdBZ' in haz_var:
        if 'MESHSdBZ'==haz_var:
            paths = xr.open_dataset(data_dir+'/V5/MESHSdBZ/MESHSdBZ_2013-2021.nc')
        elif 'MESHSdBZ_p3'==haz_var:
            paths = xr.open_dataset(data_dir+'/V5/MESHSdBZ/MESHSdBZ_2013-2021_p3.nc')

    elif haz_var == 'dBZ':
        paths = [data_dir+ '/V5/CZC/CZC_6t6_%d.nc' % year for year in years] #non-filtered data, only use after 2012

    elif haz_var == 'POH':
        paths = [data_dir+'/V5/BZC/BZC_X1d66_%d.nc' % year for year in years]
    elif haz_var == 'HKE': #MESHS based hail kinetic energy
        raise NotImplementedError('define common version with Raphi for thresholds')
        paths = xr.open_dataset(data_dir+'/V5/HKE_MESHS.nc')
        paths['HKE'] = paths.HKE.round(decimals=0) #(decimals=-1) #V1 was rounded to 10j/m2
    elif haz_var == 'durPOH':
        paths = [data_dir+'/V5/duration/BZC/BZC_DURd66_V5_%d.nc' %
                year for year in years]
    elif haz_var == 'MESHSweigh':
        pathsMESHS = [data_dir+'/V5/MZC/MZC_X1d66_%d.nc'%year for year in years]
        pathsDur = [data_dir+'/V5/duration/BZC/BZC_DURd66_V5_%d.nc'%year for year in years]
        dsMESHS = xr.open_mfdataset(pathsMESHS,concat_dim = 'time',combine='nested',coords = 'minimal')
        dsDur = xr.open_mfdataset(pathsDur,concat_dim = 'time',combine='nested',coords = 'minimal').sel(time=dsMESHS.time)
        paths = dsMESHS.MZC*(1+0.2*dsDur.BZC80_dur).round(decimals=0).rename('MESHSweigh').to_dataset()

    elif haz_var == 'crowd' or haz_var == 'crowdFiltered':
        paths = xr.open_dataset(data_dir+'/crowd-source/Reports_min100_2017-22.nc')
        if haz_var == 'crowdFiltered':
            paths = xr.open_dataset(f'{data_dir}/crowd-source/Reports_min100_2017-21_MESHSfiltered.nc')

        #filter hazard data by date
        paths = paths.sel(time=slice(f'{years[0]}-01-01',f'{years[-1]}-12-31'))
        #round to integer values
        paths = paths.round(decimals=0)

    elif haz_var=='E_kin': #E_kin from Waldvogel 1978
        paths = data_dir+'/V5/E_kin/WF/E_kin_proc_2013_2021.nc'
    elif haz_var== 'E_kinCC': #E_kin from Cecchini 2022
        paths = data_dir+'/V5/E_kin/CC/E_kin_proc_CC_2013_2021.nc'

    elif haz_var == 'VIL':
        paths = [data_dir+ '/V5/VIL/dLZC_6t6_%d.nc' % year for year in years]

    return paths


def calc_and_save_skill_scores(imp_df,dmg_thresh,imp_var,expHazImp_str):
    """
    Calculate skill scores and save them to a csv file

    Args:
        imp_df (pd.DataFrame): modelled impact, observed impact
        dmg_thresh (float): damage threshold
        imp_var (str): name of impact variable (MDR or PAA)
        expHazImp_str (str): String of Exposure Hazard ImpactFunction

    Returns:

    """
    imp_var_str = imp_var.replace('MDR','').replace('PAA','_PAA') #remove MDR from imp_var string (as it is the default)
    #read df_scores from csv
    df_scores = pd.read_csv(str(CONFIG.local_data.out_dir)+'/skill_plots/_scores.csv',index_col=0)

    for y_filter in ['_2012plus','']:
        if y_filter=='_2012plus':
            df_now=imp_df.loc[imp_df.prePost2012,:]
        else:
            df_now=imp_df.copy()

        #caculate skill scores
        rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,considered_events = calc_skill_scores(df_now,dmg_thresh)

        #Save error_measures in csv
        df_scores.loc[f'{expHazImp_str}{imp_var_str}_t{dmg_thresh:.0e}{y_filter}',:] = [
            rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,considered_events]
        if y_filter=='':
            #save df_scores to csv
            df_scores= df_scores.sort_index()
            df_scores.to_csv(str(CONFIG.local_data.out_dir)+'/skill_plots/_scores.csv')
            return rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,considered_events

def create_imp_df(imp_now,imp_obs_now,xmin):
    """create a dataframe with modelled and observed impacts

    Args:
        imp_now (climada.engine.Impact): impact modelled
        imp_obs_now (climada.engine.Impact): impact observed
        xmin (float): minimum total impact to consider

    Returns:
        imp_df: Dataframe with modelled and observed impacts
    """
    ord_dates_nonZero = np.sort(np.unique(np.concatenate((imp_now.date[imp_now.at_event>xmin],
                                                        imp_obs_now.date[imp_obs_now.at_event>xmin]))))
    imp_df = pd.DataFrame(index=ord_dates_nonZero,data={
        'date':[dt.datetime.fromordinal(d) for d in ord_dates_nonZero],
        "prePost2012":[dt.datetime.fromordinal(d).year>2012 for d in ord_dates_nonZero]})

    imp_dfMod = pd.DataFrame(data={"imp_modelled":imp_now.at_event},index = imp_now.date)
    imp_dfObs = pd.DataFrame(data={"imp_obs":imp_obs_now.at_event},index = imp_obs_now.date)

    imp_df= imp_df.join(imp_dfMod,how='left') #join on index
    imp_df= imp_df.join(imp_dfObs,how='left') #join on index
    imp_df = imp_df.fillna(0) #fill NaN
    return imp_df


def calc_skill_scores(df_now,dmg_thresh,calc_additional_scores=False,imp_modelled="imp_modelled",imp_obs="imp_obs"):

        if imp_modelled is not None:
            df_now

        #get number of points
        considered_events = sum((df_now[imp_modelled]>dmg_thresh) | (df_now[imp_obs]>dmg_thresh))

        neither_is_zero = (df_now[imp_modelled]!=0) & (df_now[imp_obs]!=0)
        rmse = np.sqrt(np.mean((df_now[imp_modelled][neither_is_zero]-df_now[imp_obs][neither_is_zero])**2))
        rmsf = np.exp(np.sqrt(np.mean(np.log(np.divide(
            df_now[imp_modelled][neither_is_zero],df_now[imp_obs][neither_is_zero]))**2)))
        rmsf_weighted = np.exp(np.sqrt(np.mean(np.log(np.divide(
            df_now[imp_modelled][neither_is_zero],df_now[imp_obs][neither_is_zero]))**2
            *df_now[imp_obs][neither_is_zero])/np.sum(df_now[imp_obs][neither_is_zero])))

        above_either_thresh = (df_now[imp_obs]>dmg_thresh) | (df_now[imp_modelled]>dmg_thresh)
        hits = sum(above_either_thresh & (df_now[imp_obs]>=df_now[imp_modelled]/10) &
                   (df_now[imp_obs]<df_now[imp_modelled]*10))
        misses = sum((df_now[imp_obs]>dmg_thresh) & (df_now[imp_modelled]<df_now[imp_obs]/10))
        false_alarms = sum((df_now[imp_modelled]>dmg_thresh) & (df_now[imp_modelled]>df_now[imp_obs]*10) )

        FAR = (sum((df_now[imp_modelled]>dmg_thresh) & (df_now[imp_modelled]>df_now[imp_obs]*10)))/sum((df_now[imp_modelled]>dmg_thresh))
        POD = sum((df_now[imp_obs]>dmg_thresh) & (df_now[imp_obs]>df_now[imp_modelled]/10) &
                  (df_now[imp_obs]<df_now[imp_modelled]*10))/sum((df_now[imp_obs]>dmg_thresh))
        # print(f"POD: {POD}, POD2 = {hits/(hits+misses)}")
        # print(f"FAR: {FAR}, FAR2 = {false_alarms/(hits+false_alarms)}")
        p_within_OOM = sum(above_either_thresh & (df_now[imp_obs]>df_now[imp_modelled]/10) &
                           (df_now[imp_obs]<df_now[imp_modelled]*10))/sum(above_either_thresh)

        if calc_additional_scores:
            #calculate RMSF only for events above the damage threshold
            rmsf_above_thresh = np.exp(np.sqrt(np.mean(np.log(np.divide(
            df_now[imp_modelled][above_either_thresh & neither_is_zero],df_now[imp_obs][above_either_thresh & neither_is_zero]))**2)))

            return rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,considered_events,rmsf_above_thresh

        return rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,considered_events

def get_mod_obs_ratio(imp_df,PAA_thresh,MDR_thresh,imp_obs='imp_observed',
                      imp_mod='imp_MESHS',imp_PAA_obs='n_count',
                      imp_PAA_mod='n_buildings_MESHS',q=0.75,plot=True):
    """Calculate the model-to-observed ratio for for monetary damages and number of affected buildings
    The model-to-observed ratio is calculated for events above a certain threshold (PAA_thresh, MDR_thresh)

    Args:
        imp_df (pd.DataFrame): Dataframe with modelled and observed impacts, with columns
        as indicated in the function arguments
        PAA_thresh (float): Threshold for minimum number of affected buildings
        MDR_thresh (float): Threshold for minimum monetary damages (modelled or observed)
        imp_obs (str, optional): column name of observed damages. Defaults to 'imp_observed'.
        imp_mod (str, optional): Column name of modelled damages. Defaults to 'imp_MESHS'.
        imp_PAA_obs (str, optional): Columns name of observed number of damaged assets. Defaults to 'n_count'.
        imp_PAA_mod (str, optional): Columns name of modelled number of damages assets. Defaults to 'n_buildings_MESHS'.
        q (float, optional): Quantile for the secondary evaluatgion metric. Defaults to 0.75.

    Returns:
        tuple: all PAA ratios, all MDR ratios
    """


    imp_sel = imp_df.loc[(imp_df[imp_obs]>=MDR_thresh) | (imp_df[imp_mod]>=MDR_thresh)]
    imp_sel['model-obs-ratio'] = imp_sel[imp_mod]/imp_sel[imp_obs]

    imp_selPAA = imp_df.loc[(imp_df[imp_PAA_obs]>=PAA_thresh) | (imp_df[imp_PAA_mod]>=PAA_thresh)]
    imp_selPAA['model-obs-ratio'] = imp_selPAA[imp_PAA_mod]/imp_selPAA[imp_PAA_obs]

    #invert negative ratios
    with np.errstate(divide='ignore'): # Suppress warnings for divide-by-zero. They are correctly handled as np.exp(np.log(0).abs()) = np.inf
        MDR_ratio_factor = np.exp(np.log(imp_sel['model-obs-ratio']).abs())
        PAA_ratio_factor = np.exp(np.log(imp_selPAA['model-obs-ratio']).abs())

    if plot:
        plot_log = False
        if plot_log:
            bins = np.logspace(0,2,20)
            xscale = 'log'
        else:
            bins = np.arange(1,11,0.5)
            xscale = 'linear'

        fig,ax = plt.subplots()
        ax.hist(MDR_ratio_factor,bins=bins,label='MDR') # note 10**(np.log10()) leads to equal result
        ax.hist(PAA_ratio_factor,bins=bins,
                label='PAA',color='orange',alpha=0.5) # note 10**(np.log10()) leads to equal result

        ymax = ax.get_ylim()[1]
        #add quantiles
        for q in [0.5,q]:
            q_mdr = MDR_ratio_factor.quantile(q)
            if q_mdr <= bins.max():
                ax.axvline(q_mdr,color='blue',linestyle='--')#,label=f'Q{q} MDR')
                ax.text(q_mdr,ymax*0.9,f'Q {q}',color = 'blue')
            q_paa = PAA_ratio_factor.quantile(q)
            if q_paa <= bins.max():
                ax.axvline(q_paa,color='orange',linestyle='--')#,label=f'Q{q} PAA')
                ax.text(q_paa,ymax*0.9,f'Q {q}',color = 'darkorange')
            # ax.axvline(MDR_ratio_factor.quantile(0.5),color='blue',linestyle='--')

        ax.set(xscale=xscale,xlabel = 'Modelled-to-observed factor(+/-)',ylabel='No. of events',
                xlim=[bins.min(),bins.max()])
        ax.legend(loc='center right')

    #print key numbersprint(f"PAA mean factor: {PAA_ratio_factor.mean():.2f} | MDR mean factor: {MDR_ratio_factor.mean():.2f}")
    print(f"PAA median factor: {PAA_ratio_factor.median():.1f} | MDR median factor: {MDR_ratio_factor.median():.1f}")
    print(f"PAA {q} quantile: {PAA_ratio_factor.quantile(q):.1f} | MDR {q} quantile: {MDR_ratio_factor.quantile(q):.1f}")

    return PAA_ratio_factor,MDR_ratio_factor


def ds_to_df(ds_all,window_size=30,vars=None):
    """Short helper function to transform an xr.Dataset containing per-grid cell values
    of crowdsourced hail size and observed damages to a pandas DataFrame

    Args:
        ds_all (xr.Dataset): dataset containing per-grid cell values
        window_size (int, optional): windowsize (Number of points, not mm) for
            rolling output. Defaults to 30.
        vars (list, optional): list of variables. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe with same data
    """
    if vars is None:
        vars = ['h_smooth','PAA','n_count_exposure']
    df = ds_all[vars].to_dataframe()
    df = df.dropna(how='any')
    # df.plot.scatter(x='h_smooth',y='PAA',c='n_count_exposure',cmap='Blues',vmax=100)


    df['h_smooth_r'] = df['h_smooth'].round(0)
    df = df.sort_values('h_smooth_r')
    if window_size is not None:
        df['PAA_rolling'] = df['PAA'].rolling(window_size,center=True,min_periods=1).mean()
        if 'MDR' in vars:
            df['MDR_rolling'] = df['MDR'].rolling(window_size,center=True,min_periods=1).mean()
    return df


def random_points_in_polygon(number, polygon):
    """Assign random points to polygon

    Args:
        number (int): number of points to assign
        polygon (shapely.polygon): polygon

    Returns:
        Point/MultiPoint: radnomly assingned points
    """
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append(point)
            i += 1
    if number==1:
        return points[0]
    if number>1:
        return MultiPoint(points)# returns list of shapely point

def poly_to_pts(gdf,n_pts,epsg,mode='random'):
    """assign random points to geodataframe with polgon geometry

    Args:
        gdf (gpd.GeoDataFrame): gdf
        n_pts (str): 'always_one' to calculate 1 point per polygon
                    otherwise name of column that stores the number
                    of points to calculate (e.g. n_claims)
        epsg (int): epsg code
        mode (str, optional): Defaults to 'random'.
"""
    gdf_out = gdf.copy()
    #assert that index corresponds to row number
    gdf_out = gdf_out.reset_index().drop(columns='index')

    if not isinstance(gdf_out,gpd.GeoDataFrame):
        gdf_out = gpd.GeoDataFrame(gdf_out,geometry='geometry')
        gdf_out = gdf_out.set_crs(epsg=epsg)


    if n_pts == 'always_one':
        gdf_out['n_points'] = 1
    # elif n_pts == 'n_claims': #n_claims is given at 'c_claims' columns
    #     print('number of points given by n_claims column')
    else:
        gdf_out['n_points'] = gdf_out[n_pts]


    if mode=='random':
        rand_points = [random_points_in_polygon(gdf_out.loc[row,'n_points'],gdf_out.loc[row,'geometry'])
                       for row in range(gdf_out.shape[0])]
        gdf_out['random_points'] = rand_points
        gdf_out=gdf_out.set_geometry('random_points')
        gdf_out=gdf_out.drop(columns=['geometry'])
    else:
        NotImplementedError(f'mode {mode} not implemented')
    return(gdf_out)


def filter_imp(imp,sel_ev):
    """Filter impact object by certain events

    Parameters
    ----------
    imp : climada.impact
        imact object to filter
    sel_ev : np.array, dtype=bool
        boolean array of length of event_id with the events to keep

    Returns
    -------
    imp_filtered : climada.impact

    """
    #sel_ev = np.isin(imp.date, dates)
    imp_out = Impact()

    for (var_name, var_val) in imp.__dict__.items():
        if var_name == 'eai_exp':
            setattr(imp_out, var_name, var_val) #will be adjusted at end of function
        elif isinstance(var_val, np.ndarray) and var_val.ndim == 1 \
                and var_val.size > 0:
            print(var_name)

            setattr(imp_out, var_name, var_val[sel_ev])
        elif isinstance(var_val, sparse.csr_matrix):
            setattr(imp_out, var_name, var_val[sel_ev, :][:, :])
        elif isinstance(var_val, list) and var_val:
            setattr(imp_out, var_name, [var_val[idx] for idx in sel_ev])
        elif var_name == 'centroids':
            setattr(imp_out, var_name, var_val)
        else:
            setattr(imp_out, var_name, var_val)

        #define number of years only if frequency is equal, otherwise it will throw an error
        if len(np.unique(imp.frequency))==1: n_years = 1/imp.frequency[0]

        imp_out.eai_exp = imp_out.imp_mat.sum(axis=0).getA1()/n_years
        imp_out.aai_agg = imp_out.eai_exp.sum()
    return imp_out


def cut_xarr(da, extent = [5.8, 10.6, 45.7, 47.9]):
    """cut netcdf xarray object to rectangular lat/lon

    Parameters
    ----------
    da : xarray.Dataset or xarray.DataArray
        dataset
    extent : list or str
        lon_min, lon_max, lat_min, lat_max, default = Switzerland

    Returns
    -------
    da : xarray.DataArray, xarray.Dataset

    """
    lon_min, lon_max, lat_min, lat_max = extent
    lon_cond = np.logical_and(da.lon >= lon_min, da.lon <= lon_max)
    lat_cond = np.logical_and(da.lat >= lat_min, da.lat <= lat_max)
    cut_da = da.where(np.logical_and(lat_cond,lon_cond),drop=True)
    return cut_da


def plot_dmg_bin(df,ax,bin_size=5,ymin=0,color='green',alpha=0.3,new_axis=True,
                 pl_var = 'count_dmg',ymax=None,relative=False,**kwargs):

    #if 0 in df.index:
    bins = pd.cut(df.index,right=False,
                  bins = np.arange(min(df.index),max(df.index)+bin_size,bin_size))
    bin_mids = bins.categories.mid.values
    bin_right = bins.categories.right.values
    bin_l = bins.categories.left.values

    if df.index[0]==0 and df.index[1]>bin_size: #first index is 0 and only contains 0 values (no rolling!)
        bin_mids[0] = 0 #change to zero
        bin_l[0] -= bin_size/2 #center first bin at 0
    df_bins=df.groupby(bins).sum()
    if relative:
        df_bins[pl_var] = df_bins[pl_var]/df_bins[pl_var].sum()
    if new_axis:
        ax2 = ax.twinx()
    else:
        ax2 = ax

    #ax2.step(bin_mids,df_bins[pl_var],color = 'green',label = '#damages',alpha=0.2)
    ax2.bar(bin_l,df_bins[pl_var],width=bin_size,color = color,label = '#damages',
            alpha=alpha,align='edge',**kwargs)
    # ax2.set(ylabel="Number of damage reports",ylim=[ymin,max(df_bins[pl_var])*1.1])

    if pl_var == 'count_all': #create axis break
        ylabel = "Number of exposed assets"
        ymax = max(df_bins[pl_var][1:])*1.3 if ymax is None else ymax
        # ax2.set_ylim([ymin,ymax])
        ax2.annotate(f'Out of axis: {(df_bins[pl_var][0]):.1e}',xy=(bin_mids[0],ymax),
                     xytext=(bin_right[0]*1.1,ymax*0.9),
                     arrowprops=dict(facecolor=color,shrink=0.05),color=color)

    elif pl_var == 'count_dmg':
        ylabel = "Fraction of damage reports" if relative else "Number of damage reports"
        ymax=max(df_bins[pl_var])*1.1 if ymax is None else ymax
    elif pl_var == 'count_cells':
        ylabel = "Number of grid cells"
        ymax=max(df_bins[pl_var])*1.1 if ymax is None else ymax

    ax2.set(ylabel=ylabel,ylim=[ymin,ymax])
    return ax2

def plot_monotone_fit(df,var,ax,color='black',label='monotonic fit',**kwargs):

    if df.index[0]==0:# skip_zero
        assert(np.where(df.index==0) == np.array([0]))
        x=df.index[1:]
        y=df[var][1:]
    else:
        x=df.index
        y=df[var]
    #avoid nan values
    no_nan = ~np.isnan(x) & ~np.isnan(y)
    x = x[no_nan]
    y = y[no_nan]

    monotone_fit = sc.smooth_monotonic(x,y,plot=False)
    ax.plot(x, monotone_fit, color=color,label = label,**kwargs)


####### plotting function for specific plots
#scatter plot of modelled vs observed damages
def make_cut_axis(a,b,c,d):

    #top bottom separation
    a.spines.bottom.set_visible(False)
    b.spines.bottom.set_visible(False)
    c.spines.top.set_visible(False)
    d.spines.top.set_visible(False)
    a.xaxis.tick_top()
    b.xaxis.tick_top()
    a.tick_params(labeltop=False) # don't put tick labels at the top
    b.tick_params(labeltop=False) # don't put tick labels at the top
    c.xaxis.tick_bottom()
    d.xaxis.tick_bottom()

    #left right separation
    a.spines.right.set_visible(False)
    c.spines.right.set_visible(False)
    b.spines.left.set_visible(False)
    d.spines.left.set_visible(False)

    a.yaxis.tick_left()
    c.yaxis.tick_left()
    b.yaxis.tick_right()
    d.yaxis.tick_right()
    b.tick_params(labelleft=False) # don't put tick labels left
    d.tick_params(labelleft=False,labelright=False) # don't put tick labels left

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    v_h = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -v_h), (1, v_h)], markersize=10,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    a.plot([0], [0], transform=a.transAxes, **kwargs)
    b.plot([1], [0], transform=b.transAxes, **kwargs)
    c.plot([0], [1], transform=c.transAxes, **kwargs)
    d.plot([1], [1], transform=d.transAxes, **kwargs)

    kwargs = dict(marker=[(-v_h,-1), (v_h,1)], markersize=10,
                linestyle="none", color='r', mec='k', mew=1, clip_on=False)
    a.plot([1], [1], transform=a.transAxes, **kwargs)
    b.plot([0], [1], transform=b.transAxes, **kwargs)
    c.plot([1], [0], transform=c.transAxes, **kwargs)
    d.plot([0], [0], transform=d.transAxes, **kwargs)

    #additional changes
    a.set(xlabel='',xticks=[0])
    b.set(xlabel='',ylabel='',yticks=[0])
    c.set(xlabel='',ylabel='',yticks=[0],xticks=[0])
    d.set(ylabel='',yticks=[0])

def grey_out_dmg_thresh(a,b,c,d,dmg_thresh=100,alpha=0.4):
    b.add_patch(Rectangle((0, 0), dmg_thresh, dmg_thresh, color="grey", alpha=alpha))
    d.add_patch(Rectangle((0, d.get_ylim()[0]), dmg_thresh, dmg_thresh, color="grey", alpha=alpha))
    a.add_patch(Rectangle((a.get_xlim()[0], 0), dmg_thresh, dmg_thresh, color="grey", alpha=alpha))
    c.add_patch(Rectangle((c.get_xlim()[0], c.get_ylim()[0]), (c.get_ylim()[1]- c.get_ylim()[0]),
                          c.get_xlim()[1]- c.get_xlim()[0], color="grey", alpha=alpha))

def skill_background_cols(a,b,c,d,min_plot_val,max_plot_val,dmg_thresh=100,alpha=0.3,
                          true_pos='tab:green',false_alarm='tab:red',missed_event='tab:orange'):
    """add background colors for skill score metrics

    Args:
        a-d (axes): axes to add background colors to
        min_plot_val (int): minimum value of "main" plot (axis b)
        dmg_thresh (int, optional): _description_. Defaults to 100.
    """

    #false alarms
    a.add_patch(Rectangle((a.get_xlim()[0], dmg_thresh), 1, 1e18, color=false_alarm, alpha=alpha,zorder=0))
    b.add_patch(Polygon([(min_plot_val,dmg_thresh), (dmg_thresh*0.1, dmg_thresh),
                         (max_plot_val*0.1, max_plot_val),(min_plot_val,max_plot_val)],
                         color=false_alarm, alpha=alpha,zorder=0))

    #true positives
    b.add_patch(Polygon([(dmg_thresh, dmg_thresh*0.1), (max_plot_val, max_plot_val*0.1),
                         (max_plot_val, max_plot_val),(max_plot_val*0.1, max_plot_val),
                         (dmg_thresh*0.1, dmg_thresh), (dmg_thresh, dmg_thresh)],
                         color=true_pos, alpha=alpha,zorder=0))

    #missed events
    d.add_patch(Rectangle((dmg_thresh, d.get_ylim()[0]), 1e10, 1, color=missed_event, alpha=alpha,zorder=0))
    b.add_patch(Polygon([(dmg_thresh, min_plot_val), (max_plot_val, min_plot_val),
                         (max_plot_val, max_plot_val*0.1),(dmg_thresh, dmg_thresh*0.1)],
                         color=missed_event, alpha=alpha,zorder=0))



    #############Pixel-wise statistics#########################
def get_and_plot_quantile(values,q_all=0.999,q_nonzero=0.95):
    quantile = np.quantile(values[~np.isnan(values)],q_all)
    quantile_nonzero = np.quantile(values[values>0],q_nonzero)
    fig,axes = plt.subplots(1,2,figsize=(10,3.5))

    axes[0].hist(values,bins=70)
    axes[0].set(yscale='log')
    axes[1].hist(values[values>0],bins=70)
    for ax in axes:
        ax.axvline(quantile,color='red',label=f'{q_all:.2%} quantile')
        ax.axvline(quantile_nonzero,color='green',label=f'{q_nonzero:.2%} quantile of non-zero values')
        ax.legend()

    return quantile,quantile_nonzero



def plot_values_above_quantile(da,quantile_value,time_dim='date',
                               spatial_dims=('chx','chy'),is_impact=False):
    """Analyse the distribution of high-intnesity values of a DataArray

    Args:
        da (xr.DataArray): data to analyse
        quantile_value (float): data value corresponding to a given quantile
                                (not the quantile itself)
        time_dim (str, optional): Time dimension. Defaults to 'date'.
        spatial_dims (tuple, optional): Spatial dimensions. Defaults to ('chx','chy').
        is_impact (bool, optional): If the data contains impacts (rather than hazard data).
                                    Defaults to False.
    """

    fig = plt.figure(figsize=(10, 4.5), layout="constrained")
    spec = fig.add_gridspec(ncols=2, nrows=1,wspace=0.15)

    proj = ccrs.AlbersEqualArea(central_longitude=6, central_latitude=47)
    ax0 = fig.add_subplot(spec[0, 0],projection=proj)
    ax1 = fig.add_subplot(spec[0, 1],projection=proj)
    # ax2 = fig.add_subplot(spec[0, 2])

    #Select values above quantile
    ds_q95 = da.where(da>quantile_value)
    #drop timesteps without values above quantile
    ds_q95 = ds_q95.dropna(dim=time_dim,how='all')

    #rank dates according to the number of high-impact pixels
    date_rank = ds_q95.sum(dim=spatial_dims).rank(dim=time_dim)
    max_rank = date_rank.max()
    date_rank_inv = max_rank-date_rank+1

    #select the top 10 events
    rank_top10 = date_rank_inv.copy()
    rank_top10 = rank_top10.where(rank_top10<=10,10)

    #create a mask for each of the 10 events
    ds95_mask = ds_q95.copy()
    for i in range(len(ds_q95[time_dim])):
        ds95_mask[i,:,:] = ((ds_q95[i,:,:]>0)*rank_top10[i]).where(ds_q95[i,:,:]>0)
    ds95_mask = ds95_mask.rename('event_ID(sorted)')

    # ds95_mask.min(dim=time_dim).plot(cmap='tab10',extend='max',ax=ax0)
    sc.plot_nc(ds95_mask.min(dim=time_dim),cmap='tab10',extend='max',ax=ax0,borders=False,cbar_lbl='Event ID (sorted)')
    ax0.set(title=f"All grid cells with >{quantile_value:.3f} {da.name} in at least one event")

    # ds_q95.max(dim=time_dim).plot(ax=ax1)
    sc.plot_nc(ds_q95.max(dim=time_dim),ax=ax1,borders=False,cbar_lbl='Max. value')
    timeseries = (ds_q95>0).sum(dim=spatial_dims)

    for ax in [ax0,ax1]:
        sc.plot_canton(ax=ax,canton=sc.constants.CH_SEL_CANTONS,lakes=False)


    #get number of events that contribute to 90% of the total high-impact pixels
    sorted_rel_contribution = np.sort(timeseries)/timeseries.sum().data
    cum_sum = np.cumsum(sorted_rel_contribution)
    n_events_90perc = len(np.argwhere(cum_sum>0.1))
    #assert actual percentage is close to 90%
    assert(abs(0.9-np.sort(timeseries)[-n_events_90perc:].sum()/timeseries.sum())<2)
    perc90_value = np.sort(timeseries)[-n_events_90perc]

    #get corresponding dates
    extr_dates = timeseries[time_dim][(timeseries>=perc90_value)]

    label = (f'90% of high-impact gridcells are \n contained in the {n_events_90perc} events '
             f'with >{perc90_value} gridcells.')

    #"define equivalent potential damage as sum of PAA in all high-impact pixels"
    if is_impact:
        eq_potential_dmg = da.sum(dim=spatial_dims)
        #contribution
        eq_dmg_contr = (eq_potential_dmg.sel({time_dim:extr_dates}).sum()/eq_potential_dmg.sum()).data
        label = f'{label}\nThis is {eq_dmg_contr*100:.1f}% of the total equivalent damage'

    _,ax2=plt.subplots()
    ax2.scatter(x=timeseries[time_dim],y=timeseries.values,c=timeseries.values>perc90_value,
            marker='o',edgecolors='black',cmap='viridis_r')

    ax2.axhline(perc90_value,color='red',label=label)
    ax2.set(ylabel="No. of grid cells (2.2x2.2 km)", xlabel='Date')
    ax2.legend(loc='upper left')


def cut_xr_to_canton(ds,sel_cantons=('Zürich','Bern','Luzern','Aargau'),buffer=0):
    """Cut an xarray dataset to the extent of the selected cantons

    Args:
        ds (xr.DataSet): dataset to cut
        sel_cantons (tuple, optional): Selected cantons.
            Defaults to ('Zürich','Bern','Luzern','Aargau').
        buffer (int, optional): Buffer around the canton borders in meters (unit of EPSG:2056).

    Returns:
        ds_out: cut xr.DataSet
    """
    ds_out = ds.copy()
    ch_shp_path = str(CONFIG.ch_shp_path)
    if sel_cantons == 'all':
        countries = gpd.read_file("%s/swissTLMRegio_LANDESGEBIET_LV95.shp"%ch_shp_path)
        sel_polys = countries.loc[countries.ICC=='CH']
    else:
        cantons = gpd.read_file("%s/swissTLMRegio_KANTONSGEBIET_LV95.shp"%ch_shp_path)
        sel_polys = cantons.loc[cantons.NAME.isin(sel_cantons)]
    if buffer>0:
        sel_polys = sel_polys.buffer(buffer)
        sel_polys = gpd.GeoDataFrame(geometry=sel_polys)
    sel_polys = sel_polys.to_crs(epsg=4326)

    ds_coords = gpd.GeoSeries(gpd.points_from_xy(np.ravel(ds.lon), np.ravel(ds.lat)),crs='EPSG:4326')

    coords_within = ds_coords.within(sel_polys.dissolve().geometry.values[0])

    ds_out['coords_within'] = (ds_out.lon.dims,coords_within.values.reshape(ds_out.lon.shape))
    ds_out = ds_out.where(ds_out.coords_within,drop=True)

    return ds_out

def cut_xr_to_country(ds,country='Switzerland',buffer_lat_lon=0,res="50m"):
    """Cut an xarray dataset to the extent of the selected country

    Args:
        ds (xr.DataSet): dataset to cut
        country (str, optional): Selected country. Defaults to 'Switzerland'.
        buffer_lat_lon (int, optional): Buffer around the country borders in lat/lon.
        res (str, optional): Resolution of the country borders: "50m" or "low".
            Defaults to "50m" (i.e. 1:50mio).

    Returns:
        ds_out: cut xr.DataSet
    """
    ds_out = ds.copy()
    if res == 'low':
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        country = world.loc[world.name==country]
    elif res == '50m':
        world = gpd.read_file(sc.constants.NE_BORDERS)
        country = world.loc[(world.SOVEREIGNT==country)&
                            ((world.TYPE=='Country')|(world.TYPE=='Sovereign country'))]

    if buffer_lat_lon>0:
        country = country.buffer(buffer_lat_lon)
        country = gpd.GeoDataFrame(geometry=country)
    assert country.crs == 'EPSG:4326'

    ds_coords = gpd.GeoSeries(gpd.points_from_xy(np.ravel(ds.lon), np.ravel(ds.lat)),crs='EPSG:4326')

    coords_within = ds_coords.within(country.dissolve().geometry.values[0])

    ds_out['coords_within'] = (ds_out.lon.dims,coords_within.values.reshape(ds_out.lon.shape))
    ds_out = ds_out.where(ds_out.coords_within,drop=True)

    return ds_out

def filter_land_only(ds,diag_plot=False):
    """
    Filter the dataset to keep only values on land. Works only on the
    climate simulation (CORDEX) domeain

    Parameters:
    ds (xr.Dataset): Input dataset.
    diag_plot (bool): If True, plot a diagnostic figure of filtered data

    Returns:
    xr.Dataset: Filtered dataset with only land values.
    """

    data_dir = str(CONFIG.local_data.data_dir)
    #Read land-sea mask
    land_sea_mask = pd.read_csv(f"{data_dir}/HailCast/land_sea_sandro/land_sea_mask.csv",
                                 header=None).values.astype(bool)

    # Check that the land-sea mask has the correct shape
    assert 'rlat' in ds.coords and 'rlon' in ds.coords
    assert land_sea_mask.shape == (ds.sizes['rlat'], ds.sizes['rlon'])


    # Apply the land-sea mask to the dataset
    ds['land_sea_mask'] = (('rlat', 'rlon'), land_sea_mask)
    ds_filtered = ds.where(ds.land_sea_mask)

    # Plot diagnostic figure
    if diag_plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        ds['DHAIL_MX'].max(dim='time').plot(ax=axs[0])
        ds_filtered['DHAIL_MX'].max(dim='time').plot(ax=axs[1])
        return ds_filtered, fig

    return ds_filtered



################## quantile calibration visualization ##########################

def get_dmg_df(intensity,param_dicts,haz_sorted,imp_sorted,haz_vals_all,
               bootstrap_quantiles=None,center_bins=True,
               #haz_counts=None
               ):
    """Calculate a dataframe with equivalent damage, based on the given impact functions and hazard intensities.

    Args:
        intensity (np.arrayx): intensity values
        param_dicts (dict of dicts): all fitted impact function parameters to display
        haz_sorted (np.array): sorted haz values from quantile mapping
        imp_sorted (np.array): sorted imp values from quantile mapping§
        haz_vals_all (np.array): all hazard values in the hazard dataset
        bootstrap_quantiles (dict of arrays): arrays with same shape as imp_sorted
        center_bins (bool, optional): center bins at the intensity values. Defaults to False, meaning the values refer to the left bin edge.
        # haz_counts (np.array): counts of each haz value per intensity bin.
        #                         Optional. (same shape as haz_sorted,iimp_sorted)

    Returns:
        df_impf (pd.DataFrame): dataframe with equivalent damages and mapped impact function values
    """

    #calculate the impact functions for the given intensities
    impf_vals_dict = {}
    for impf_name in param_dicts.keys():
        impf_vals_dict[impf_name] = sc.calib_opt.get_emanuel_vals(
            intensity=intensity,**param_dicts[impf_name],)

    impf_vals_dict= {**impf_vals_dict, **{'quant':np.nan}}

    #Optinal: add 2 bootstrap quantiles
    if bootstrap_quantiles is not None:
        assert(len(bootstrap_quantiles.keys()))==2
        q1 = sorted(list(bootstrap_quantiles.keys()))[0]
        q2 = sorted(list(bootstrap_quantiles.keys()))[1]
        for k in bootstrap_quantiles.keys():
            assert imp_sorted.shape == bootstrap_quantiles[k].shape

        impf_vals_dict= {**impf_vals_dict, **{f'q{q1}':np.nan,f'q{q2}':np.nan}}


    #initialize dataframe
    df_impf = pd.DataFrame(index=intensity,data=impf_vals_dict)

    #Add value counts to dataframe
    count,hc_intensity = np.histogram(haz_vals_all,bins=list(df_impf.index)+[np.inf])
    df_impf['value_counts'] = count


    if center_bins:
        #assert that the bins are equally spaced
        assert np.allclose(np.diff(intensity),np.diff(intensity)[0])
        dx = np.diff(intensity)[0]

    #Add quantile impact function
    for i in range(len(df_impf.index)):
        idx_val = df_impf.index[i]
        # idx_val_min1 = df_impf.index[i-1]
        haz_upperBound = df_impf.index[i+1] if i+1<len(df_impf.index) else np.inf
        if center_bins:
            indices = np.where((haz_sorted>=idx_val-dx/2) & (haz_sorted<idx_val+dx/2))[0]
        else:
            indices = np.where((haz_sorted>=idx_val) & (haz_sorted<haz_upperBound))[0]
        if len(indices)==0:
            continue
            #forward fill (not used now, instead NaNs are filled with interpolation below)
            df_impf.loc[idx_val,'quant'] = df_impf.loc[idx_val_min1,'quant']
            if bootstrap_quantiles is not None:
                df_impf.loc[idx_val,f'q{q1}'] = df_impf.loc[idx_val_min1,f'q{q1}']
                df_impf.loc[idx_val,f'q{q2}'] = df_impf.loc[idx_val_min1,f'q{q1}']
        else:
            #Optianally calculate weighted mean instead
            # if haz_counts is not None:
            #     df_impf.loc[idx_val,'quant'] = np.average(imp_sorted[indices],weights=haz_counts[indices])
            # else:
            df_impf.loc[idx_val,'quant'] = imp_sorted[indices].mean()


            if bootstrap_quantiles is not None:
                df_impf.loc[idx_val,f'q{q1}'] = bootstrap_quantiles[q1][indices].mean()
                df_impf.loc[idx_val,f'q{q2}'] = bootstrap_quantiles[q2][indices].mean()


    #fill NaNs with interpolation
    df_impf = df_impf.interpolate(method='linear',limit_direction='both')


    #calculate equivalent damage
    for impf_name in param_dicts.keys():
        df_impf[f'dmg_{impf_name}'] = count*df_impf[impf_name]
    df_impf['dmg_quant'] = count*df_impf['quant']
    if bootstrap_quantiles is not None:
        df_impf[f'dmg_q{q1}'] = count*df_impf[f'q{q1}']
        df_impf[f'dmg_q{q2}'] = count*df_impf[f'q{q2}']
        # # df_impf['dmg_q95'] = count*df_impf['q95']


    return df_impf

def plot_df_impf(df_impf,impf_names=['flex','p3','quant'],full_impf_names = None):

    """Plotting function for df_impf as create in function above

    Args:
        df_impf (pd.DataFrame): impact function dataframe
        impf_names (list, optional): List of impact function opitions. Defaults to ['flex','p3','quant'].
        full_impf_names (list, optional): Full names of impact function options. Defaults to None.

    Returns:
        fig: plt.Figure
    """
    colors = ['tab:green','tab:orange','tab:blue','tab:red'][:len(impf_names)]
    if full_impf_names is None:
        full_impf_names = impf_names

    fig, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(14,4))
    for i,impf_name in enumerate(impf_names):
        ax.plot(df_impf.index, df_impf[impf_name], color=colors[i],label = f'{full_impf_names[i]}')
    ax.set(title='Impact functions (PAA)')

    ax.legend()
    ax_right = ax.twinx()
    bin_widths = list(np.diff(df_impf.index))+[np.diff(df_impf.index)[-1]]
    ax_right.bar(df_impf.index,df_impf['value_counts'],color='grey',alpha=0.5,width=bin_widths,align='edge')
    ax_right.set(yscale='log')

    dmg_colnames = [f"dmg_{n}" for n in impf_names]
    df_impf[dmg_colnames].plot(ax=ax2,color=colors)

    if "dmg_q5" in df_impf.columns:
        ax2.fill_between(df_impf.index,df_impf['dmg_q5'],df_impf['dmg_q95'],color='tab:blue',alpha=0.2)
    ax2.set(xlim=(0,None),title='Damage per intensity')

    #plot cumulative damage
    df_impf[dmg_colnames].cumsum().plot(ax=ax3,color=colors,legend=False)
    if "dmg_q5" in df_impf.columns:
        ax3.fill_between(df_impf.index,df_impf['dmg_q5'].cumsum(),df_impf['dmg_q95'].cumsum(),color='tab:blue',alpha=0.2)
    ax3.set(title='Cumulative damage',xlim=(0,None))
    cumsums =  '\n'.join([ f"{var}: {df_impf[var].cumsum().iloc[-1]:.1f}" for var in dmg_colnames])
    ax3.annotate(cumsums,(0.1,0.8),xycoords='axes fraction')

    return fig