# -*- coding: utf-8 -*-
"""
Collection of functions for reading and processing data for the use in Climada
in the framework of the scClim project (scclim.ethz.ch)
 - reading hazard data from radar
 - reading exposure data from .csv/.gpkg files

"""

import datetime as dt
import geopandas as gpd
import numpy as np
import pandas as pd
import sys
import xarray as xr
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Hazard
from climada import CONFIG
from scipy import sparse
import warnings
try:
    import climada.util.lines_polys_handler as u_lp
    from climada.engine import Impact, ImpactCalc
except: #exception for old climada version on n2o
    from climada.engine import Impact

import matplotlib.pyplot as plt

sys.path.append(str(CONFIG.local_data.func_dir))
from scClim.constants import BAUINDEX,BAUJAHR_DICT

# Hazard
def hazard_from_radar(files, varname='MESHS', time_dim='time', forecast_init=None,
                      ensemble_dim=None, spatial_dims = None, country_code=None,
                      extent=None, subdaily = False, month=None, ignore_date=False,
                      n_year_input=None, get_xarray=False):
    """Create a new Hail hazard from MeteoCH radar data
    or COSMO HAILCAST ouput (single- or multi-member)

    Parameters
    ----------
    files : list of str or xarray Dataset
        list of netcdf filenames (string) or xarray Dataset object
    varname : string
        the netcdf variable name to be read from the file
    time_dim : str
        Name of time dimension, default: 'time'
    forecast_init : datetime object
        List with datetimes of forecast initializations,
        needs to have same length as time, default: None
    ensemble_dim : str
        Name of ensemble dimension, default: None
    spatial_dims : list of str
        Names of spatial dimensions
    country_code : int
        ISO 3166 country code to filter the data
    extent : list / array
        [lon_min, lon_max, lat_min, lat_max]
    ignore_date : boolean
        If True: ignores netcdf dates (e.g. for synthetic data).
    n_year_input : int
        Number of years: will only be used if ignore_date=True
    Returns
    -------
    haz : Hazard object
        Hazard object containing radar data with hail intensity (MESHS)
    """

    #Initialize default values
    if spatial_dims is None: spatial_dims = ['chy','chx']

    #read netcdf if it is given as a path
    if type(files) == xr.core.dataset.Dataset:
        netcdf = files
    else:
        netcdf = xr.open_mfdataset(files, concat_dim=time_dim, combine='nested',
                                   coords='minimal')

    #select month of the year if given
    if month:
        grouped=netcdf.groupby("time.month")
        netcdf=grouped[int(month)]

    #Cut data to selected country/area only
    if extent:
        lon_min, lon_max, lat_min, lat_max = extent
        lon_cond = np.logical_and(netcdf.lon >= lon_min, netcdf.lon <= lon_max)
        lat_cond = np.logical_and(netcdf.lat >= lat_min, netcdf.lat <= lat_max)
        cond = np.logical_and(lat_cond,lon_cond).compute()
        netcdf = netcdf.where(cond,drop=True)


    # #stack data
    # stacked = netcdf.stack(new_dim=spatial_dims)

    # if country_code:
    #     c_code = util.coordinates.get_country_code(stacked.lat,stacked.lon)
    #     stacked = stacked.assign_coords(country_code=("new_dim",c_code))
    #     stacked = stacked.where(stacked.country_code==country_code,drop=True)

    #Select variable and set units
    varname_xr = varname #by default the varname corresponds to the xr name
    if varname in ['MESHS', 'MESHS_4km', 'MESHS_opt_corr','MESHS_20to23']:
        varname_xr = 'MZC'
        unit = 'mm'
    elif varname == 'MESHSdBZ' or varname == 'MESHSdBZ_p3':
        varname_xr = 'MESHSdBZ'
        unit = 'mm'
    elif varname == 'POH':
        varname_xr = 'BZC'
        unit = '%'
    elif 'DHAIL' in varname:
        unit = 'mm'
    elif varname == 'MESH':
        #calculate MESH from SHI
        netcdf = 2.54*(netcdf)**0.5 #Witt et al 1998
        netcdf = netcdf.rename_vars({'SHI':'MESH'})
        #round to steps of 0.5mm
        netcdf['MESH'] = np.round(netcdf['MESH'] / 0.5) * 0.5  # round to steps of 0.5mm (for calibration)
        netcdf['MESH'] = netcdf['MESH'].where(netcdf['MESH'] < 80, 0)  # outlier values to zero
        unit = 'mm'

    elif varname == 'dBZ' or varname =='dBZfiltered':
        varname_xr = 'CZC'
        unit = 'dBZ'
        # Filter values for efficient calculation. dBZ<40 are set to zero
        netcdf = netcdf.where(netcdf[varname_xr]>40,0)
    elif varname == 'possible_hail':
        unit = '[ ](boolean)'
    elif varname == 'durPOH':
        varname_xr = 'BZC80_dur'
        netcdf[varname_xr] = netcdf[varname_xr]*5 #times 5 to get minutes
        unit = '[min]'
    elif varname == 'MESHSweigh':
        unit = 'mm (scaled by duration)'
    elif varname == 'HKE':
        unit = 'Jm-2'
    elif varname == 'crowd' or varname=='crowdFiltered':
        #warnings.warn('use smoothed data for crowd-sourced data')
        varname_xr = 'h_smooth'
        unit = 'mm'
    elif varname == 'E_kin' or varname=='E_kinCC': #E_kin from Waldvogel 1978, or Cecchini 2022
        varname_xr = 'E_kin'
        unit = 'Jm-2'
    elif varname == 'VIL':
        unit = 'g/m2'
        varname_xr = 'dLZC'
        # Filter values for efficient calculation. VIL<10g/m2 are set to zero
        netcdf = netcdf.where(netcdf[varname_xr]>10,0).round(0)
    else:
        raise ValueError(f'varname "{varname}" is not implemented at the moment')

    #prepare xarray with ensemble dimension to be read as climada Hazard
    if ensemble_dim:
        # omit extent if ensemble_dim is given
        if extent:
            warnings.warn("Do not use keyword extent in combination with "
                          "ensemble_dim. Plotting will not work.")
        # omit igonore_date if ensemble_dim is given
        if ignore_date:
            warnings.warn('Do not use keyword ignore_date in combination with '
                          'ensemble_dim. Event names are set differently.')
        # stack ensembles along new dimension
        netcdf = netcdf.stack(time_ensemble=(time_dim, ensemble_dim))

        # event names
        if forecast_init: #event_name = ev_YYMMDD_ensXX_init_YYMMDD_HH
            n_member, = np.unique(netcdf[ensemble_dim]).shape
            forecast_init = np.repeat(forecast_init, n_member)
            if netcdf[time_dim].size != len(forecast_init):
                warnings.warn("Length of forecast_init doesn't match time.")
            event_name = np.array([f"{pd.to_datetime(ts).strftime('ev_%y%m%d')}_ens{ens:02d}_{init.strftime('init_%y%m%d_%H')}"
                                   for (ts,ens),init in zip(netcdf.time_ensemble.values, forecast_init)])
        else: #event_name = ev_YYMMDD_ensXX
            event_name = np.array([f"{pd.to_datetime(ts).strftime('ev_%y%m%d')}_ens{ens:02d}"
                                   for ts,ens in netcdf.time_ensemble.values])
        #convert MultiIndex to SingleIndex
        netcdf = netcdf.reset_index('time_ensemble')
        netcdf = netcdf.assign_coords({'time_ensemble':netcdf.time_ensemble.values})

        if 'time_ensemble' in netcdf.lat.dims:
            # remove duplicates along new dimension for variables that are identical across members (only if lat/lon are saved as variables rather than coordinates)
            netcdf['lon'] = netcdf['lon'].sel(time_ensemble=0, drop=True)
            netcdf['lat'] = netcdf['lat'].sel(time_ensemble=0, drop=True)

    # get number of events and create event ids
    n_ev = netcdf[time_dim].size
    event_id = np.arange(1, n_ev+1, dtype=int)

    if ignore_date:
        n_years = n_year_input
        if 'year' in netcdf.coords:
            event_name = np.array(['ev_%d_y%d'%i for i in zip(event_id,netcdf.year.values)])
        else:
            event_name = np.array(['ev_%d'%i for i in event_id])
    elif ensemble_dim:
        n_years = netcdf[time_dim].dt.year.max().values-netcdf[time_dim].dt.year.min().values + 1
    else:
        n_years = netcdf[time_dim].dt.year.max().values-netcdf[time_dim].dt.year.min().values + 1
        if subdaily:
            event_name = netcdf[time_dim].dt.strftime("ev_%Y-%m-%d_%H:%M").values
        else:
            event_name = netcdf[time_dim].dt.strftime("ev_%Y-%m-%d").values

    #Create Hazard object
    event_dim = time_dim
    coord_vars = dict(event=event_dim,longitude='lon',latitude='lat')
    haz = Hazard.from_xarray_raster(netcdf,'HL',unit,intensity=varname_xr,
                                    coordinate_vars=coord_vars)
    #set correct event_name, frequency, date
    haz.event_name = event_name
    haz.frequency = np.ones(n_ev)/n_years
    if ignore_date:
        haz.date = np.array([], int)
    if ensemble_dim:
        haz.date = np.array([pd.to_datetime(ts).toordinal() for ts in netcdf[time_dim].values])

    netcdf.close()
    haz.check()

    if get_xarray:
        return haz,netcdf
    else:
        return haz


def read_zurich_exposure(shapefile,input_type = 'random_points',
                         return_type = 'exp_pnt'):
    """Read Zurich exposure shapefile (.gpkg) to exposure object

    Parameters
    ----------
    shapefile : csv
        file with GVZ exposure (or damage) data
    input_type : string
        'random_points' for point gdf, 'poly' for polygon gdf
    return_type : string
        'exp_pnt' for point exposure, 'exp_shp' for polygon exposure
    Returns
    -------
    exp : Exposure object
        Exposure object
    """

    exp_gdf = gpd.read_file(shapefile)
    exp = Exposures(exp_gdf.to_crs(epsg=4326),value_unit='CHF')


    if return_type == 'exp_pnt' and input_type=='poly':
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None
        )
        exp_pnt.check()
        return(exp_pnt)
    elif return_type == 'exp_shp' and input_type=='random_points':
        ValueError('Cannot return shapefile with points as input')
    elif return_type == 'exp_shp' and input_type=='poly':
        return(exp)
    elif return_type == 'exp_pnt' and input_type=='random_points':
        return(exp)
    else:
        raise ValueError('Invalid input_type / return_type - Combination')

def read_zurich_dmg(gdf_path,years=None):
    gdf = gpd.read_file(gdf_path)

    #remove rows without geometries
    no_geom = gdf.geometry.isna()
    if sum(no_geom)>0.01*len(no_geom):
        ValueError('More than 1% of geometries are invalid. Please check!')
    gdf = gdf[~no_geom]

    #add date
    gdf['date_dt']=pd.to_datetime(gdf.CLAIMS_DATE)

    #rename damage as 'value'
    gdf = gdf.rename(columns={'PAID':'value'})

    #filter years
    if years:
        year_dt = gdf.date_dt.dt.year
        gdf = gdf.loc[(year_dt>=years[0]) & (year_dt<=years[1]),:]

    #filter months (only hail season April-September)
    month_sel = (gdf.date_dt.dt.month>=4) & (gdf.date_dt.dt.month<=9)
    if (sum(gdf.loc[~month_sel,'value'])/sum(gdf['value']))>0.05:
        ValueError('More than 5% of reported damages are not in the \
                    hail season. Please check input!')
    gdf = gdf.loc[month_sel,:]

    assert(gdf.crs =='EPSG:4326')
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y

    imp_out = gdf_to_imp(gdf,id_col='POLNR',exp_val_col='VEHVALUE')
    return(imp_out)

def read_gvz_exposure(csv_file,on_grid = False,exp_n_assets=False,
                      exterior_only=False,crs = 'EPSG:2056'):
    """Read CSV file

    Parameters
    ----------
    csv_file : csv
        file with GVZ exposure data
    exp_n_assets : bool
        if True: read number of assets as exposure (rather than their value)
    on_grid : bool
        whether or not data should be interpolated on regular grid (not implemented yet)
    exterior_only : bood
        if yes: estimate value for building exterior (which is easily damaged) from total value
    Returns
    -------
    exp : Exposure object
        Exposure object
    """
    if crs.upper() == 'EPSG:2056':
        x_coord='KoordinateOst'
        y_coord= 'KoordinateNord'
    elif crs.upper() == 'EPSG:4326':
        x_coord='longitude'
        y_coord= 'latitude'

    gvz_df = pd.read_csv(csv_file,sep=";")

    if exp_n_assets:
        unit = ''
        gvz_df['value'] = 1 #set value for each exposure to 1
    else:
        unit = 'CHF'
        gvz_df=gvz_df.rename({'Versicherungssumme':'value'},axis =1)

        if exterior_only:
            # values for all cantons (see grid_cantonal_data.py (5.4.2023))
            # 0.27 is empirical powerlaw, 1560 corresponds to 99% of damages below new estimate
            gvz_df['value'] = np.minimum(gvz_df['value'],1560*gvz_df['value']**0.27)
            # Note that in theory the scaling from volume to surface would be
            # a powerlaw with 0.66 as exponent (2/3)
    gdf = gpd.GeoDataFrame(gvz_df, geometry = gpd.points_from_xy(gvz_df[x_coord],
                                                                 gvz_df[y_coord],
                                                                 crs = crs))

    #Create Exposure
    exp = Exposures(gdf,value_unit = unit)
    exp.set_lat_lon()
    exp = exp.to_crs(epsg=4326)
    exp.check()

    return exp

def read_gvz_dmg(csv_file, cant_insurance ,min_dmg = 0,w_rel_dmg=False,
                return_type='imp',exp_n_assets=False,haz=None,years: tuple=None,
                index_dmgs=True,crs = 'EPSG:2056',id_col='VersicherungsID',
                baujahr_filter=''):
    """Read CSV file

    Parameters
    ----------
    csv_file : csv
        file with GVZ exposure (or damage) data
    cant_insurance : str
        Name of cantonal building insurance company (e.g. 'GVZ')
    on_grid : bool
        whether or not data should be interpolated on regular grid
        (not implemented yet)
    w_rel_dmg : bool
        if relative damage should be saved in gdf too
    return_type: str
        if 'imp': return imp object
        if 'gdf': return gpd.GeoDataFrame
        if 'imp_df': return pd.Dataframe to be used for calib_optimize
        if 'imp_df_yearly': same, but with yearly damages
    exp_n_assets : bool
        if True: read number of assets as exposure (rather than their value)
    haz: climada.hazard
        climada.hazard object to get event_ids. Only needed if return_type='imp_df'
    years: tuple
        tuple of (yearMin,yearMax), to only select the years in this range
    baujahr_filter: str
        if not '', filters damages only for assets with certain year of construction
    Returns
    -------
    xxx : *see "return type" input
    """
    if crs.upper() == 'EPSG:2056':
        x_coord='KoordinateOst'
        y_coord= 'KoordinateNord'
    elif crs.upper() == 'EPSG:4326':
        x_coord='longitude'
        y_coord= 'latitude'

    gvz_dmg = pd.read_csv(csv_file,sep=";") #,parse_dates=['Schadendatum'])
    if 'date_dt' in gvz_dmg.columns:
        gvz_dmg['date_dt'] = pd.to_datetime(gvz_dmg.date_dt)
    else:
        gvz_dmg['Schadendatum'] = gvz_dmg['Schadendatum'].astype(str).str.zfill(8)
        gvz_dmg['date_dt']=pd.to_datetime(gvz_dmg.Schadendatum,format='%d%m%Y')

    #drop 'Schadendatum' column to avoid further confusion
    if 'Schadendatum' in gvz_dmg.columns:
        gvz_dmg.drop(columns=['Schadendatum'])

    #filter years
    if years:
        year_dt = gvz_dmg.date_dt.dt.year
        gvz_dmg = gvz_dmg.loc[(year_dt>=years[0]) & (year_dt<=years[1]),:]

    #filter by year of construction
    if not baujahr_filter=='':
        yearMin, yearMax = BAUJAHR_DICT[baujahr_filter]
        if sum(gvz_dmg.Baujahr.isna())/len(gvz_dmg.Baujahr)>0.01:
            raise ValueError('Over 1% of exposure points without valid Baujahr')
        # Missing entries  for 'Baujahr' are mainly from Bern, when buildings
        # were renovated! i.e. they were already standing before 2002,
        # when the radar data begins ->thus set to 2000
        gvz_dmg.Baujahr[gvz_dmg.Baujahr.isna()] = 2000
        gvz_dmg = gvz_dmg.loc[(gvz_dmg.Baujahr>=yearMin) & (gvz_dmg.Baujahr<=yearMax),:]

    if exp_n_assets:
        unit = ''
        gvz_dmg['value'] = 1 #set value for each impact to 1
    else:
        unit = 'CHF'

        if index_dmgs:#Index damages to year 2021
            gvz_dmg['Bauindex'] = gvz_dmg.date_dt.dt.year.map(BAUINDEX[cant_insurance])
            gvz_dmg['value'] = BAUINDEX[cant_insurance].loc[2021]/gvz_dmg['Bauindex']*gvz_dmg['Schadensumme']
        else:
            print('Warning: No indexing of damages')
            gvz_dmg['value'] = gvz_dmg['Schadensumme']

    grpy = gvz_dmg.groupby('date_dt').sum(numeric_only=True)
    grpy = grpy.reset_index()
    dates= grpy['date_dt'][grpy.value>min_dmg]
    #dates = grpy.index[grpy.value>min_dmg] #.strftime('%Y-%m-%d')
    if w_rel_dmg:
        gvz_dmg['rel_dmg'] = gvz_dmg['value'] / gvz_dmg['Versicherungssumme']

    if return_type == 'gdf':
        gdf_dmg = gpd.GeoDataFrame(gvz_dmg,geometry = gpd.points_from_xy(gvz_dmg[x_coord],
                               gvz_dmg[y_coord],crs = crs))
        gdf_dmg = gdf_dmg.to_crs(4326)
        gdf_dmg['latitude'] = gdf_dmg.geometry.y
        gdf_dmg['longitude'] = gdf_dmg.geometry.x
        gdf_dmg['id_col'] = gdf_dmg[id_col]
        return gdf_dmg

    elif return_type=='imp':
        imp_out = gdf_to_imp(gvz_dmg,id_col=id_col,dates=dates,exp_val_col='Versicherungssumme',
                            x_coord=x_coord,y_coord=y_coord,crs=crs,unit=unit)
        return(imp_out)

    elif return_type=='id_col':
        coord_df = gvz_dmg.groupby(id_col).first()[[y_coord,x_coord]]
        return(np.array(coord_df.index))

    elif return_type == 'imp_df_yearly':
        years = np.array(dates.dt.year)
        grpy['year']=years
        grpy_year = grpy.groupby('year').sum()
        impact_data_source = grpy_year.reset_index()[['year','value','Schadensumme']].rename(columns={'value':'impact_scaled','Schadensumme':'impact'})
        for col,val in zip(['ISO','regiond_id','reference_year'],['CHE',756,2021]):
            impact_data_source[col]=val
        return impact_data_source

    elif return_type == 'imp_df':
        #get ordinal date and year to add to grpy_df
        ord_dates = np.array(dates.map(dt.datetime.toordinal))
        grpy['date'] = ord_dates
        grpy_sel = grpy[['value','Schadensumme','date']].rename(columns={'value':'impact_scaled','Schadensumme':'impact'})

        #initialize dataframe with rows for each event from haz object
        impact_data_source=pd.DataFrame({'event_id':haz.event_id,
                    'event_name':haz.event_name,
                    'year':[dt.datetime.fromordinal(d).year for d in haz.date],
                    'date':haz.date})
        impact_data_source = impact_data_source.merge(grpy_sel,how='outer',on='date')
        return impact_data_source

def gdf_to_imp(gdf,id_col,dates=None,exp_val_col=None,x_coord='lon',
               y_coord='lat',crs = 'EPSG:4326',unit=''):

        #assign dates if not given
        if dates is None:
            dates = pd.Series(gdf.date_dt.unique())

        #In case one has to group by Coordinates (2 columns), use these 2 lines
        # imp_df=gvz_dmg.groupby(['KoordinateNord','KoordinateOst']).size().rename('count')
            # temp=df_sel.set_index(['KoordinateNord','KoordinateOst'])['value'].rename(date)

        #create coordinate df
        coord_df = gdf.groupby(id_col).first()[[y_coord,x_coord]]
        coord_df = gpd.GeoDataFrame(coord_df,geometry = gpd.points_from_xy(coord_df[x_coord],
                               coord_df[y_coord],crs = crs))
        if not coord_df.crs == 'EPSG:4326':
            coord_df = coord_df.to_crs(epsg=4326)
        coord_df['latitude'] = coord_df.geometry.y
        coord_df['longitude'] = coord_df.geometry.x

        #Initialize impact matrix as dataframe
        imp_df = pd.DataFrame(index=coord_df.index)
        #Initialize matrix to store affected asset values
        aff_value_df = pd.DataFrame(index=coord_df.index)
        for date in dates:
            df_sel =  gdf.loc[gdf.date_dt==date,:]
            temp = df_sel.set_index(id_col)['value'].rename(date)
            #In rare cases there are 2 damages for the same house and date,
            # then add them together
            if len(temp)!=len(np.unique(temp.index.astype(str))):
                temp = temp.groupby(level=0).sum()
            #add the damages of 'date' to the impact matrix
            imp_df=pd.concat([imp_df,temp],axis=1)

            #Add values to affected_asset_value matrix:
            if exp_val_col != None:
                temp = df_sel.set_index(id_col)[exp_val_col].rename(date)
                #In rare cases there are 2 damages for the same house and date,
                #just select the Versicherungsumme *1
                if len(temp)!=len(np.unique(temp.index.astype(str))):
                    temp = temp.groupby(level=0).first()
                aff_value_df=pd.concat([aff_value_df,temp],axis=1)

        imp_df = imp_df.fillna(0)
        imp_mat = sparse.csr_matrix(imp_df.T.values)


        imp_out = Impact()
        imp_out.coord_exp = coord_df[['latitude','longitude']].values
        imp_out.crs = coord_df.crs

        #Date and frequency
        ord_dates = np.array(dates.map(dt.datetime.toordinal))
        ev_names = np.array([date.strftime('ev_%Y-%m-%d') for date in dates])
        imp_out.date = ord_dates
        imp_out.event_name = ev_names
        n_years = np.ceil((max(ord_dates)-min(ord_dates))/365) #alternative: count unique years
        imp_out.frequency = np.ones(imp_df.shape[1])/n_years
        imp_out.event_id = np.arange(imp_df.shape[1])+1

        #Damages
        imp_out.imp_mat = imp_mat
        imp_out.at_event = imp_mat.sum(axis=1).getA1()
        imp_out.eai_exp = imp_mat.sum(axis=0).getA1()/n_years
        imp_out.aai_agg = imp_mat.sum()/n_years

        #Exposed asset value
        if exp_val_col != None:
            aff_value_df = aff_value_df.fillna(0)
            aff_mat = sparse.csr_matrix(aff_value_df.T.values)
            imp_out.aff_mat = aff_mat
            #NOTE: aff_mat (affected exposure) is a matrix with the total value of affected assets
        #Others
        imp_out.unit = unit
        imp_out.affected_total_value = np.nan
        return imp_out

def read_xr_exposure(nc_file,var_name,val_unit='CHF'):
    """Read exposure from netCDF file"""
    if isinstance(nc_file,str):
        nc = xr.open_dataset(nc_file)[var_name]
    elif isinstance(nc_file,xr.Dataset):
        nc = nc_file[var_name]
    nc = nc.rename('value')
    df_exp = nc.to_dataframe()
    #CORRECT WAY
    if 'lon' in df_exp.columns:
        df_exp = df_exp.rename(columns={'lon':'longitude','lat':'latitude'})
    gdf_exp = gpd.GeoDataFrame(df_exp,geometry = gpd.points_from_xy(df_exp.longitude,
                               df_exp.latitude,crs = 'EPSG:4326'))
    exp = Exposures(gdf_exp,value_unit = val_unit)
    # add 'impf_' columns if it was not initialized already
    if not any(column.startswith('impf_') for column in exp.gdf.columns):
        exp.gdf['impf_'] = np.full_like(exp.gdf.shape[0], 1)
    exp.check()

    # # CLIMADA WAY in order to have raster data
    # df_exp=df_exp.reset_index()
    # Note: Climada requires x/y coordinates to be called longitude/latitude
    # df_exp = df_exp.rename(columns={'chx':'longitude','chy':'latitude'})
    # print(df_exp.head())
    # gdf_exp = gpd.GeoDataFrame(df_exp,geometry = gpd.points_from_xy(df_exp.longitude,
    #                            df_exp.latitude,crs = 'EPSG:2056'))
    # exp = Exposures(gdf_exp,value_unit = val_unit)
    # exp.check()
    return exp

def read_xr_impact(nc_file,var_name,spatial_dims=['chy','chx'],time_dim='date',
                   unit='CHF',years=None):
    """Read impact from netCDF file"""
    if isinstance(nc_file,str):
        nc = xr.open_dataset(nc_file)[var_name]
    elif isinstance(nc_file,xr.Dataset):
        nc = nc_file[var_name]
    nc = nc.rename('value')
    stacked = nc.stack(new_dim=spatial_dims).fillna(0)

    # filter by years
    if years is not None:
        stacked = stacked.sel({time_dim:slice(str(years[0]),str(years[-1]))})

    # df_imp = nc.to_dataframe()
    n_ev = len(stacked[time_dim])
    n_years = len(np.unique(stacked[time_dim].dt.year))

    if 'lon' in stacked.coords:
        coord_exp = np.array([stacked.lat.values,stacked.lon.values]).T
        crs = 'EPSG:4326'
    else:
        raise NotImplementedError('Only lat/lon coordinates are supported')

    imp = Impact(
        event_id = np.arange(n_ev)+1,
        event_name = np.array([d.strftime('ev_%Y-%m-%d') for d in stacked[time_dim].dt.date.values]),
        date = np.array([d.toordinal() for d in stacked[time_dim].dt.date.values]),
        coord_exp = coord_exp,
        imp_mat = sparse.csr_matrix(stacked),
        crs = crs,
        eai_exp = stacked.sum(dim=[time_dim]).values/n_years,
        at_event = stacked.sum(dim='new_dim').values,
        frequency = np.ones(n_ev)/n_years,
        aai_agg = float(stacked.sum(dim=["new_dim",time_dim]).values)/n_years,
        unit = unit
    )

    return imp


def impf_from_csv(csv,smooth,emanuel_fit=False,PAA_only=False,plot=False,
                  return_impf=False):

    if smooth:
        mdr,mdd,paa= 'MDR_smooth','MDD_smooth','PAA_smooth'
    else:
        mdr,mdd,paa= 'MDR','MDD','PAA'
    if emanuel_fit:
        if smooth: raise ValueError('Cannot have smooth and emanuel_fit=True')
        mdr,mdd,paa='MDR_emanuel','MDD_emanuel','PAA_emanuel'

    df_impf = pd.read_csv(csv,index_col = 0)
    if PAA_only: #Impact function for PAA only (MDD=1)
        imp_fun1 = ImpactFunc()
        imp_fun1.id = 1
        imp_fun1.intensity = df_impf.index.values
        imp_fun1.mdd = np.ones_like(df_impf[paa].values)
        imp_fun1.paa = df_impf[paa]
        imp_fun1.haz_type = 'HL'
        imp_fun1.check()

        imp_fun_set = ImpactFuncSet()
        imp_fun_set.append(imp_fun1)
        if plot:
            ax = imp_fun1.plot()
            ax.set_ylim([0,max(df_impf[paa].values)*100*1.1])
    else:

        imp_fun1 = ImpactFunc()
        imp_fun1.id = 1
        imp_fun1.intensity = df_impf.index.values
        imp_fun1.mdd = df_impf[mdr].values
        imp_fun1.paa = np.ones_like(df_impf[mdr].values)
        imp_fun1.haz_type = 'HL'
        imp_fun1.check()

        # second impact function based on PAA*MDD Note that it has a bias
        # because of expensive buildings are damaged more often!
        imp_fun2 = ImpactFunc()
        imp_fun2.id = 2
        imp_fun2.intensity = df_impf.index.values
        imp_fun2.mdd = df_impf[mdd].values
        imp_fun2.paa = df_impf[paa]
        imp_fun2.haz_type = 'HL'
        imp_fun2.check()
        if plot:
            ax = imp_fun1.plot()
            ax.set_ylim([0,max(df_impf[mdr].values)*100*1.1])
            imp_fun2.plot()

        imp_fun_set = ImpactFuncSet()
        imp_fun_set.append(imp_fun1)
        imp_fun_set.append(imp_fun2)
    return imp_fun_set

# %% Agriculture specific functions
def split_hazard_to_time_subsets(hazard, startdates, enddates,
                                 subset_names=['pre-heading','heading','post-heading','post-harvest']):

    """Split a hazard object into a hazard subsets based on n fixed (e.g. phenological) time windows during the year. To be used for separate impact computation in each phase e.g. due to changing vulnerability
    Parameters
    ----------
    hazard : climada.hazard
        hazard object
    startdates : list of n strings
        List of date strings in the format "mm-dd" with the startdates of the time window
    enddates : list of n strings
        List of date strings in the format "mm-dd" with the enddates of the time window
    subset_names : list of n strings
        List of name strings that describe the names of the subsets.
        Default is phenological stages: ['pre-heading','heading','post-heading','post-harvest']

    Returns
    -------
    hazard_subset : dict
        Dictionary with hazard object for each of the n subsets
    """

    startdates=list(startdates)
    enddates=list(enddates)
    subset_names=list(subset_names)
    hazard_subsets={}

    #get start year and end year from hazard object
    startyear=int(hazard.get_event_date()[0][0:4])
    endyear=int(hazard.get_event_date()[-1][0:4])

    #loop over all phases
    for i,startdate in enumerate(startdates):

        enddate=enddates[i]
        subset_name=subset_names[i]

        #create list of event dates
        dr=pd.DatetimeIndex([],dtype='datetime64[ns]',freq='D')

        #loop over all years, create date slice and add to list of event dates
        for y in range(startyear,endyear+1):

            year=str(y)
            startdateyear=year+'-'+startdate
            enddateyear=year+'-'+enddate

            #create list of dates in the specified window
            dr_new=pd.date_range(start=startdateyear,end=enddateyear, freq='D')
            dr=dr.union(dr_new)

        #create event names from event dates; contains all days of a given time window of the year (for all years)
        event_names=['ev_'+dat.strftime('%Y-%m-%d') for dat in dr]

        #get all event names that are actually present in hazard
        ev_names=[ev for ev in event_names if ev in hazard.event_name]

        if len(event_names) != len(ev_names):
            warnings.warn("Not all events of time period in hazard object. Continue nevertheless.")

        #select subset of hazard based on event names
        if len(ev_names)>0:
            hazard_subsets[subset_name]=hazard.select(event_names=ev_names) #date=(startdate,enddate))
        else:
            hazard_subsets[subset_name]=None

    return hazard_subsets

def exposure_from_gpd(filename, description, value_colname = 'area_ha', value_unit = 'ha',):

    """
    read exposure from a file that is readable as geopandas.GeoDataFrame.
    Here used to read crop-specific exposure data gridded on a 1x1km grid

    Parameters
    ----------
        filename:  string
            Name of the file to read as GeoDataFrame and create Exposure from
        description: string
            String describing the exposure. Is added to the Exposure.tag.
        value_column: string
            Name of the column in the GeoDataFrame that is interpreted as 'value' column of the Exposure (default: 'area_ha')
        value_unit: string
            unit of the exposure values (default: ha)
        return_cell_gdf: boolean
            whether or not to return the original GeoDataFrame with geometry of the gridcell saved as polygons (which is neat for plotting)
            default: False
    Returns
    -----------
        exp: climada.entity.exposures.base.Exposures
            Climada exposure
        cell_gdf: GeoDataFrame
            orignal GeoDataFrame with gridcells explicityl specified in geometry
    """

    filename = str(filename)
    value_colname = str(value_colname)
    description = str(description)

    #read file
    gdf = gpd.read_file(filename)

    #gdf=gdf.fillna(0)
    #create column for impact function
    gdf['impf_'] = 1

    #define the value column
    gdf=gdf.rename(columns = {value_colname : "value"})

    #set up exposure
    exp = Exposures(gdf, value_unit = value_unit)

    #set latitude longitude columns from geometry points
    exp.set_lat_lon()

    #tag exposure (not used after CLIMADA 4.0)
    #exp.tag.file = filename
    #exp.tag.description = description

    #check
    exp.check()

    #remove exposure points with value zero
    exp.gdf = exp.gdf[exp.gdf['value']>0]

    return exp

def read_shv_dmg(excel_file, croptype, return_type = 'exp_dict', exposure = None, hazard = None):
    """Read Excel file

    Parameters
    ----------
    csv_file : csv
        file with GVZ exposure (or damage) data
    croptype : string
        string denoting croptype, for several croptypes use underscore (_) to separate between names
    on_grid : bool
        whether or not data should be interpolated on regular grid (not implemented yet)
    w_rel_dmg : bool
        if relative damage should be saved in gdf too
    return_type: str
        if 'imp': return imp object
        if 'exp_dict': returns dict of exposures
    haz: climada.hazard
        climada.hazard object to get event_ids. Only needed if return_type='imp'
    Returns
    -------
    exp_dict : dict
        Dictionary with Exposure objects for each event
    """

    #### start helper functions ####
    def get_damage_dict(shv_dmg, return_type):

           out_dict = {}
           dates=shv_dmg.groupby('Schadendatum').groups.keys()

           #Loop over all dates
           for date in dates:
               df_sel =  shv_dmg.loc[shv_dmg.Schadendatum==date]

               if return_type == 'exp_dict':
                   gdf_dmg = gpd.GeoDataFrame(df_sel,geometry = gpd.points_from_xy(df_sel['lon'],
                                          df_sel['lat'],crs = 'EPSG:4326'))

                   exp_dmg = Exposures(gdf_dmg,value_unit = '%')
                   exp_dmg.set_lat_lon()
                   exp_dmg.check()
                   out_dict.update({date:exp_dmg})
               elif return_type == 'gdf_dict':
                   gdf_dmg = gpd.GeoDataFrame(df_sel,geometry = gpd.points_from_xy(df_sel['X'],
                                          df_sel['Y'],crs = 'EPSG:2056'))
                   out_dict.update({date: gdf_dmg})

           return out_dict

    def adjust_croptype(croptype):
        if croptype == "Weizen":
            croptype = "Winterweizen"
        if croptype == "Gerste":
            croptype = "Wintergerste"
        if croptype == "Reben":
            croptype = "Wein"
        if croptype == "Aepfel":
            croptype = "Tafeläpfel"
        if croptype == "Mais":
            croptype = "Mais|mais"
        if croptype == "Raps":
            croptype = "Raps|raps"
        return croptype

    def read_damage_dataframe(excel_file, croptype):
        shv_dmg = pd.read_excel(excel_file) #,parse_dates=['Schadendatum'])
        # select croptype(s)
        #split if several croptypes
        croptypes=croptype.split("_")
        #adjust names to make sure all of them are found
        croptypes_corrected=[adjust_croptype(c) for c in croptypes]
        shv_dmg = shv_dmg.loc[shv_dmg['Kultur'].str.contains('|'.join(croptypes_corrected))]
        # remove zero values
        shv_dmg=shv_dmg.loc[shv_dmg['Ertragsverlust (%)']!=0]
        # set value
        shv_dmg["value"] = shv_dmg["Ertragsverlust (%)"].values

        return shv_dmg

    def get_impact_mats_for_imp_dict(shv_dmg,exposure,hazard):

            #Group by coordinates and create copys for output
            imp_df=shv_dmg.groupby(['lat','lon']).size().rename('count')
            nfields_df=imp_df.copy(deep=True)
            area_df=imp_df.copy(deep=True)
            ha_loss_df=imp_df.copy(deep=True)

            #make coordinate array from array of tuples
            coord_map=map(np.array,imp_df.index.values)
            coord_exp=np.array(list(coord_map))

            dates=shv_dmg.groupby('Schadendatum').groups.keys()

            if exposure:
                exp_dmgs=get_damage_dict(shv_dmg, return_type = 'exp_dict')

            for date in dates:

                if exposure and hazard:
                    #get dataframe with field area for each coordinate
                    exp_dmg = exp_dmgs[date]
                    exp_dmg.assign_centroids(hazard)
                    exposure.assign_centroids(hazard)
                    exp_dmg = compute_affected_area_from_dmgs(exp_dmg, exposure)
                    area_temp = exp_dmg.gdf.groupby(['lat','lon'])['field_area'].mean().rename(date)

                #get damages (loss in percent) dataframe at date
                df_sel =  shv_dmg.loc[shv_dmg.Schadendatum==date]

                #get damages in ha by multiplying with field area
                #get all damage values and set coordinates as index
                temp = df_sel.set_index(['lat','lon'])['Ertragsverlust (%)'].rename(date)


                if exposure and hazard:
                   #get total area lost (multiply harvest loss in % with area exposed)
                    col=[]
                    for i in temp.index:
                        col.append(area_temp.loc[area_temp.index==i].values[0])
                    temp_ha=temp.copy(deep=True)*np.array(col)/100

                #count number of damage claims at each coordinate for this date (if larger than 1 this indicates a coordinate related to community centroid rather than a specific field)
                nfields_temp=temp.groupby(['lat','lon']).count()

                #If several damages for same coordinate, then average them together
                if len(temp)!=len(np.unique(temp.index)):
                    temp = temp.groupby(['lat','lon']).mean()
                    if exposure and hazard:
                        temp_ha = temp_ha.groupby(['lat','lon']).sum().rename(date)

                #add the damages of 'date' to the impact matrix
                imp_df=pd.concat([imp_df,temp],axis=1)
                nfields_df=pd.concat([nfields_df,nfields_temp],axis=1)

                if exposure and hazard:
                    area_df=pd.concat([area_df,area_temp],axis=1)
                    ha_loss_df=pd.concat([ha_loss_df,temp_ha],axis=1)



            imp_df = imp_df.fillna(0)
            imp_df=imp_df.drop(columns='count')
            nfields_df = nfields_df.fillna(0)
            nfields_df=nfields_df.drop(columns='count')

            imp_mat = sparse.csr_matrix(imp_df.T.values)
            nfields_mat = sparse.csr_matrix(nfields_df.T.values)

            if exposure and hazard:
                area_df = area_df.fillna(0)
                area_df=area_df.drop(columns='count')
                area_mat = sparse.csr_matrix(area_df.T.values)
                ha_loss_df = ha_loss_df.fillna(0)
                ha_loss_df = ha_loss_df.drop(columns='count')
                ha_loss_mat = sparse.csr_matrix(ha_loss_df.T.values)
            else:
                ha_loss_mat=sparse.csr_matrix(imp_mat.shape())
                area_mat=sparse.csr_matrix(imp_mat.shape())

            return imp_mat, nfields_mat, ha_loss_mat, area_mat, coord_exp

    def get_impact_mats_for_imp(shv_dmg,exposure,hazard):

            #Group by ID and create copys for output
            coord_df = shv_dmg.groupby('ID').first()[['lat','lon']]

            #Initialize impact matrix as dataframe
            imp_df = pd.DataFrame(index=coord_df.index)
            #Initialize matrix to store exposed asset values
            aff_value_df = pd.DataFrame(index=coord_df.index)

            #make coordinate array from array of tuples
            coord_exp=coord_df[['lat','lon']].values

            #get list of dates
            dates=list(shv_dmg.groupby('Schadendatum').groups.keys())

            #get damage exposure dict (to get field area)
            exp_dmgs=get_damage_dict(shv_dmg, return_type = 'exp_dict')


            for date in dates:

                #get dataframe with field area for each exposure point
                exp_dmg = exp_dmgs[date]
                exp_dmg.assign_centroids(hazard)
                exposure.assign_centroids(hazard)
                exp_dmg = compute_affected_area_from_dmgs(exp_dmg, exposure)
                area_temp = exp_dmg.gdf.groupby(['ID'])['field_area'].mean().rename(date)

                #get damages (loss in percent) dataframe at date
                df_sel =  shv_dmg.loc[shv_dmg.Schadendatum==date]
                #get all damage values and set ID as index
                temp = df_sel.set_index(['ID'])['Ertragsverlust (%)'].rename(date)

                #get total area lost (multiply harvest loss in % with area exposed)
                col=[]
                for i in temp.index:
                        col.append(area_temp.loc[area_temp.index==i].values[0])
                temp_ha=temp.copy(deep=True)*np.array(col)/100

                #add the damages of 'date' to the impact matrix
                imp_df=pd.concat([imp_df,temp_ha],axis=1)

                #add affected area to impact matrix for affected area
                aff_value_df=pd.concat([aff_value_df,area_temp],axis=1)


            imp_df = imp_df.fillna(0)
            imp_mat = sparse.csr_matrix(imp_df.T.values)

            #create matrix of exposed asset values
            aff_value_df = aff_value_df.fillna(0)
            aff_mat = sparse.csr_matrix(aff_value_df.T.values)

            return imp_mat, aff_mat, coord_exp

    #### end helper functions ####

    if (return_type in ['imp','imp_dict']) and (not exposure):
        raise ValueError('For return_type {} an exposure need to be passed.'.format(return_type))

    #adjust croptype names if necessary
    croptype = adjust_croptype(croptype)

    #read damage data
    shv_dmg = read_damage_dataframe(excel_file,croptype)
    dates=shv_dmg.groupby('Schadendatum').groups.keys()

    if return_type in ['imp','imp_dict']:

        #get impact matrices
        if return_type == 'imp_dict':
            imp_mat, nfields_mat, ha_loss_mat, area_mat, coord_exp = \
               get_impact_mats_for_imp_dict(shv_dmg,exposure,hazard)
        elif return_type ==  'imp':
            imp_mat, aff_mat, coord_exp = \
                    get_impact_mats_for_imp(shv_dmg,exposure,hazard)

        #Date and frequency
        ord_dates = np.array([d.toordinal() for d in dates])
        ev_names = np.array([d.strftime('ev_%Y-%m-%d') for d in dates])
        n_years = np.floor((max(ord_dates)-min(ord_dates))/365)
        frequency = np.ones(imp_mat.shape[0])
        event_id = np.arange(imp_mat.shape[0])+1

        #Others
        tot_value = np.nan

        #Damages
        imp_at_event = imp_mat.sum(axis=1).getA1()
        imp_eai_exp = imp_mat.sum(axis=0).getA1()/n_years
        imp_aai_agg = imp_mat.sum()/n_years

        if return_type == 'imp_dict':
            nfields_at_event = nfields_mat.sum(axis=1).getA1()
            nfields_eai_exp = nfields_mat.sum(axis=0).getA1()/n_years
            nfields_aai_agg = nfields_mat.sum()/n_years

            #aggregated damages for area measures
            ha_loss_at_event = ha_loss_mat.sum(axis=1).getA1()
            ha_loss_eai_exp = ha_loss_mat.sum(axis=0).getA1()/n_years
            ha_loss_aai_agg = ha_loss_mat.sum()/n_years

            area_at_event = area_mat.sum(axis=1).getA1()
            area_eai_exp = area_mat.sum(axis=0).getA1()/n_years
            area_aai_agg = area_mat.sum()/n_years



            #initialize impact objects
            imp_out = Impact(
                imp_mat = imp_mat,
                coord_exp = coord_exp,
                crs = 'EPSG:4326',
                date = ord_dates,
                event_name = ev_names,
                frequency = frequency,
                event_id = event_id,
                at_event = imp_at_event,
                eai_exp = imp_eai_exp,
                aai_agg= imp_aai_agg,
                unit ='%',
                tot_value=tot_value)

            nfields_out = Impact(
                imp_mat = nfields_mat,
                coord_exp = coord_exp,
                crs = 'EPSG:4326',
                date = ord_dates,
                event_name = ev_names,
                frequency = frequency,
                event_id = event_id,
                at_event = nfields_at_event,
                eai_exp = nfields_eai_exp,
                aai_agg = nfields_aai_agg,
                unit='',
                tot_value=tot_value)

            ha_loss_out = Impact(
                    imp_mat = ha_loss_mat,
                    coord_exp = coord_exp,
                    crs = 'EPSG:4326',
                    date = ord_dates,
                    event_name = ev_names,
                    frequency = frequency,
                    event_id = event_id,
                    at_event = ha_loss_at_event,
                    eai_exp = ha_loss_eai_exp,
                    aai_agg= ha_loss_aai_agg,
                    unit='ha',
                    tot_value=tot_value)

            area_out = Impact(
                    imp_mat = area_mat,
                    coord_exp = coord_exp,
                    crs = 'EPSG:4326',
                    date = ord_dates,
                    event_name = ev_names,
                    frequency = frequency,
                    event_id = event_id,
                    at_event = area_at_event,
                    eai_exp = area_eai_exp,
                    aai_agg= area_aai_agg,
                    unit='ha',
                    tot_value=tot_value)

            #create output dict
            imp={'loss (%)': imp_out, 'fields affected': nfields_out, 'loss (ha)': ha_loss_out, 'area affected (ha)': area_out}

        elif return_type == 'imp':
                imp = Impact(
                    imp_mat = imp_mat,
                    coord_exp = coord_exp,
                    crs = 'EPSG:4326',
                    date = ord_dates,
                    event_name = ev_names,
                    frequency = frequency,
                    event_id = event_id,
                    at_event = imp_at_event,
                    eai_exp = imp_eai_exp,
                    aai_agg= imp_aai_agg,
                    unit ='ha',
                    tot_value=tot_value)
                #ad matrix of affected area
                imp.aff_mat=aff_mat

        return imp

    elif return_type in ['exp_dict', 'gdf_dict']:

        out_dict=get_damage_dict(shv_dmg, return_type = return_type)

        return out_dict

def read_gridded_damages_as_impact(filename_dmg,impact_metric='n_fields_dmg', binary_damage=False):

    damages_data=xr.open_dataset(filename_dmg)

    #Group by coordinates and get df
    df_all=damages_data.to_dataframe().reset_index().fillna(0)
    df_all_new=df_all[df_all['n_fields_exp']>0]
    coord_df=df_all_new.groupby(['chy','chx']).first()[['lat','lon']]

    #make coordinate array from array of tuples
    coord_exp=coord_df[['lat','lon']].values

    #Initialize impact matrix as dataframe with coordinates as index
    imp_df = pd.DataFrame(index=coord_df.index)

    #get all damage dates
    dates=list(df_all_new.groupby('time').groups.keys())

    #loop over dates and extend concatenate impacts for individual dates as new columns to impact dataframe (imp_df)
    for date in dates:
        temp_df=df_all_new[df_all_new['time']==date].set_index(['chy','chx'])[impact_metric]
        if binary_damage == True:   #if binary damage is True set impact to 1 at each grid cell
            temp_df=temp_df/temp_df
        imp_df=pd.concat([imp_df,temp_df],axis=1)

    #fill Nan values and reset coordinate index to integer index
    imp_df = imp_df.fillna(0).reset_index().drop(['chx','chy'],axis=1)

    #create a sparse matrix from the impact dataframe
    imp_mat = sparse.csr_matrix(imp_df.T.values)

    #Prepare climada Impact attributes
    #Date and frequency
    ord_dates = np.array([d.toordinal() for d in dates])
    ev_names = np.array([d.strftime('ev_%Y-%m-%d') for d in dates])
    n_years = np.floor((max(ord_dates)-min(ord_dates))/365)
    frequency = np.ones(imp_mat.shape[0])
    event_id = np.arange(imp_mat.shape[0])+1

    #Others
    tot_value = np.nan

    #Damages
    imp_at_event = imp_mat.sum(axis=1).getA1()
    imp_eai_exp = imp_mat.sum(axis=0).getA1()/n_years
    imp_aai_agg = imp_mat.sum()/n_years

    #initialize impact object
    imp_out = Impact(
                    imp_mat = imp_mat,
                    coord_exp = coord_exp,
                    crs = 'EPSG:4326',
                    date = ord_dates,
                    event_name = ev_names,
                    frequency = frequency,
                    event_id = event_id,
                    at_event = imp_at_event,
                    eai_exp = imp_eai_exp,
                    aai_agg= imp_aai_agg,
                    unit = '',
                    tot_value=tot_value)
    return imp_out

def drop_events(drop_dict, damages):
    """
    Drop events from damage impact objects for different croptypes


    Parameters
    ----------
    drop_dict: dict
        Dictionary with events to be dropped per croptype
    damages : dict
        Dictionary of damages (damages as impact objects) from which to drop events

    Returns
    -------
    damages : dict
        Dictionary of damages with dropped events

    """
    for croptype in damages.keys():
        events=list(damages[croptype].event_name)
        for ev in drop_dict[croptype]:
            if ev in events:
                events.remove(ev)
        damages[croptype]=damages[croptype].select(event_names=events)
    return damages
# %% Impact computation

def log_func_PVA(x,croptype,which='mean'):

    if croptype == 'wheat':
        if which == 'mean':
            a, b, c, d = [ 0.10754041,  0.06987181,  0.14708824, 44.93387726]
        elif which == 'min':
            a, b, c, d = [ 0.05829311,  0.05422006,  0.3901072 , 43.1887867 ]
        elif which == 'max':
            a, b, c, d = [ 0.13695882,  0.08222636,  0.09417312, 42.61603538]
    else:
        raise ValueError('Crop type {} not implemented'.format(croptype))

    return a / (1.0 + np.exp(-c * (x - d))) + b

def log_func_PAA(x,croptype):

    if croptype == 'wheat':
        a, b, c, d = [1.90534634e-01, 3.45333516e-02, 6.25820365e-02, 4.93882516e+01]
    elif croptype == 'grapevine':
        a, b, c, d = [ 0.29478159,  0.12742784,  0.1954461 , 44.31808603]
    else:
        raise ValueError('Crop type {} not implemented'.format(croptype))

    return a / (1.0 + np.exp(-c * (x - d))) + b

def sigmoid_impact_func_stages(stage, kind='PAA', which='mean', croptype='wheat',sep_hazard=True):

    """ Create a logistic impact function for hail hazard for croptype  at a given crop stage (coarse stages)

    ----------
    stage: integer
       integer denoting stage of the crop (coarse stages)
    sep_hazard: boolean
        if True define separate hazard types for each stage if False use same hazard type for each stage (default: True)
    Returns
    -------
    climada.entity.ImpactFunc()

    """

    #set impact function attributes
    #tag
    tag=str(croptype)+str(stage)+'_dummy_coarse'
    name=str(croptype)+' stage '+str(stage)

    if sep_hazard:
        ID=1
    else:
        ID=stage

    #set params
    #dummy numbers to illustrate changing vulnerability

    if stage==0: # pre-heading #Bestockung bis Aehrenschieben/Heading
        haz_type='HL_pre_head'
        factor=1
    elif stage==1: #heading / Flowering
        factor=1
        haz_type='HL_head'
    elif stage==2: #post-heading, Fruchtbildung bis Reife
        factor=1
        haz_type='HL_post_head'
    elif stage==3: #post-harvest, nach der Ernte
        factor=0
        haz_type='HL_post_harv'
    else:
        raise ValueError("Crop stage {} not implemented.".format(stage))

    #create impact function
    imp_fun = ImpactFunc()

    # create a sigmoid impact function
    #imp_fun=imp_fun.from_sigmoid_impf((0,100,2),L,k,x0,if_id=ID)
    intensity_range=np.arange(20,121,1)
    intensity_range_0=np.concatenate((np.arange(0,20,1),intensity_range))
    imp_fun.intensity=intensity_range_0
    imp_fun.mdd=np.ones((len(intensity_range_0)))

    if kind == 'PAA': #percent assets affected
        imp_fun.paa=np.concatenate((np.zeros(20),log_func_PAA(intensity_range, croptype)*factor))
    elif kind == 'PVA': #percent value affected
        imp_fun.paa=np.concatenate((np.zeros(20),log_func_PVA(intensity_range, croptype, which = which)*factor))

    imp_fun.id=ID

    #set attributes

    #(maybe can remove this in the future)
    if sep_hazard:
        imp_fun.haz_type = 'HL'
    else:
        imp_fun.haz_type = 'HL'

    imp_fun.intensity_unit = 'mm'
    imp_fun.tag = tag
    imp_fun.name = name

    # check if the all the attributes are set correctly
    imp_fun.check()

    return imp_fun


def get_impact_functions_crops(stages,stage_names,croptype, kind='PAA', which = 'mean',plot=False, figdir=None):

        """get impact functions"""

        #create impact function for each stage and add to ImpactFuncSet
        imp_func_sets={stage_names[s]: ImpactFuncSet() for s in stages}
        for s in stages:
            imp_func_sets[stage_names[s]].append(sigmoid_impact_func_wheat_coarse(s, kind = kind, which = which, sep_hazard=True))

        # make plot if True
        if plot == True:
            #plot impact functions and save figure
            f,axs=plt.subplots(2,2)
            axs=axs.flatten()
            for s in stages:
                ax=axs[s]
                imp_func_sets[stage_names[s]].plot(axis=ax)
                ax.get_legend().remove()
                ax.set_ylim([0,20])
                if s in [1,4]: #2,4,5,7,8]:
                    ax.set_ylabel('')
                    ax.set_yticklabels('')
                if s in [0,1]: #,2,3,4,5]:
                    ax.set_xticklabels('')
                    ax.set_xlabel('')
            axs[0].legend()

        figname='impact_functions_'+croptype+'_coarse_calibrated.png'

        if figdir:
            f.savefig(figdir+figname,bbox_inches='tight',pad_inches=0)
            print('Figure ' + figdir + figname + ' saved.')
        else:
            print('Figure not saved.')

        return imp_func_sets


def make_hazard_phenology_subsets(hazard,pheno_dat,stage_names,stages):
    #make hazard subsets
    stages_dat=pheno_dat['stage']
    startdates=pheno_dat['start']
    enddates=pheno_dat['end']

    if not np.array_equal(stages,stages_dat):
        raise ValueError('Defined stage numbers not same as the ones in phenology dataset. Defined {}. \n Phenology Data: {}'.format(stages,stages_dat))
    hazard_subsets=split_hazard_to_time_subsets(hazard,startdates,enddates,subset_names=stage_names)

    return hazard_subsets

def calc_impact_from_subsets(hazard_subsets, imp_func_sets, exposure):

    impacts={}
    for stage_name in hazard_subsets.keys():

        if hazard_subsets[stage_name]:
            imp=ImpactCalc(exposure,imp_func_sets[stage_name], hazard_subsets[stage_name]).impact(save_mat=True)
            impacts[stage_name]=imp


    # concatenate together to combined impact object
    if len(list(impacts.values())) > 1:
        impact_concat=Impact.concat(list(impacts.values()))

    return [impact_concat, impacts]

def compute_affected_area_from_dmgs(exposure_damages_object, exposure):

    """ compute percent area affected from damages and exposure
    Parameters
    ---------

    exposure_damages_object: climada.entity.exposure
                damage claims of a single event stored as exposure object
    exposure:  climada.entity.exposure
                exposure data for a given croptype

    Returns
    -------

    exposure_damages_object_new: climada.entity.exposure
                a copy of exposure_damages_object but with values for PAA and total area affected"""


    exp_dmg=exposure_damages_object

    #get field area
    exposure.gdf['field_area']=exposure.gdf['value']/exposure.gdf['n_fields']
    #compute Swiss average field area to use if no field area can be defined
    field_area_mean=exposure.gdf['value'].sum()/exposure.gdf['n_fields'].sum()

    #add field area column
    exp_dmg.gdf['field_area']=np.nan

    #loop over all damages and add field area
    for index in exp_dmg.gdf.index:

           #get centroids
           centroid=exp_dmg.gdf.loc[index,'centr_HL']
           #read field area from exposure
           field_area=exposure.gdf.loc[exposure.gdf[exposure.gdf['centr_HL']==centroid].index,'field_area'].values
           if field_area.size == 1:
               exp_dmg.gdf.loc[index,'field_area']=field_area[0]
           elif field_area.size == 0:
               #issue warning if damage is recorded where no exposure is present
               exp_dmg.gdf.loc[index,'field_area']=field_area_mean
               warnings.warn('damage reported in region without exposure. swiss average field size used.')
           else:

               raise ValueError('more than one field area found.')



    return exp_dmg





