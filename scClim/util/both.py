# -*- coding: utf-8 -*-
"""
Utility functions for both/all Subprojects
"""
import datetime as dt
import climada.util.coordinates as u_coord
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt



def add_imp_to_xr(imp,ds,xdim='chx',ydim='chy',varname='imp'):
    """Write the impact matrix to an existing xarray dataset. The existing
    dataset must have the same dimensions as the exposure used to create the
    impact object (i.e. also the same as the impact matrix).

    Args:
        imp (climada.Impact): Impact object containing the impact matrix.
        ds (xr.Dataset): xarray dataset to which the impact matrix should be added.
    """

    #Create 2d arrays of lat/lon and assert that they match the existing dataset
    lat = np.reshape(imp.coord_exp[:,0],(len(ds[ydim]),len(ds[xdim])))
    latOrig = ds.lat.values
    np.testing.assert_array_almost_equal(lat,latOrig)
    lon = np.reshape(imp.coord_exp[:,1],(len(ds[ydim]),len(ds[xdim])))
    lonOrig = ds.lon.values
    np.testing.assert_array_almost_equal(lon,lonOrig)


    #Create array of dates in imp object
    imp_sel_dates = np.array([dt.datetime.fromordinal(int(d)) for d in imp.date])

    #Reshape the impact data to 3d array and create xarray dataset
    impResh= np.reshape(imp.imp_mat.toarray(),(imp.imp_mat.shape[0],len(ds[ydim]),len(ds[xdim])))
    imp_xr = xr.Dataset({varname: (('date',ydim,xdim),impResh)},
                            coords={'date':imp_sel_dates,ydim:ds[ydim],xdim:ds[xdim]})

    #Add the impact data to the existing dataset
    ds[varname] =imp_xr[varname]

    return ds



def assign_centroids_gdf(gdf, hazard, distance='euclidean',
                        threshold=u_coord.NEAREST_NEIGHBOR_THRESHOLD):
    """Assign for each exposure coordinate closest hazard coordinate.
    -1 used for disatances > threshold in point distances. If raster hazard,
    -1 used for centroids outside raster.
    Parameters
    ----------
    hazard : Hazard
        Hazard to match (with raster or vector centroids).
    distance : str, optional
        Distance to use in case of vector centroids.
        Possible values are "euclidean", "haversine" and "approx".
        Default: "euclidean"
    threshold : float
        If the distance (in km) to the nearest neighbor exceeds `threshold`,
        the index `-1` is assigned.
        Set `threshold` to 0, to disable nearest neighbor matching.
        Default: 100 (km)
    See Also
    --------
    climada.entity.exposures.base for details: same function for exposure.gdf's
    climada.util.coordinates.assign_coordinates: method to associate centroids to
        exposure points
    """

    if not u_coord.equal_crs(gdf.crs, hazard.centroids.crs):
        raise ValueError('Set hazard and exposure to same CRS first!')
    if hasattr(hazard.centroids,'meta') and  hazard.centroids.meta:
        assigned = u_coord.assign_grid_points(
            gdf.longitude.values, gdf.latitude.values,
            hazard.centroids.meta['width'], hazard.centroids.meta['height'],
            hazard.centroids.meta['transform'])
    else:
        assigned = u_coord.assign_coordinates(
            np.stack([gdf.latitude.values, gdf.longitude.values], axis=1),
            hazard.centroids.coord, distance=distance, threshold=threshold)
    try:
        haz_type = hazard.tag.haz_type
    except AttributeError: #CLIMADA 4.0+
        haz_type = hazard.haz_type

    gdf['centr_' + haz_type] = assigned


def assign_centroids_imp(imp,hazard,distance='euclidean',
                        threshold=u_coord.NEAREST_NEIGHBOR_THRESHOLD):
    """
    same as assign_centroids_gdf, but for impact object

    """
    #make sure the imp crs is epsg:4326, as it is required by the u_coord methods
    assert(imp.crs == 'EPSG:4326')
    coord_df = pd.DataFrame(imp.coord_exp,columns=['latitude', 'longitude'])
    gdf = gpd.GeoDataFrame(coord_df,geometry = gpd.points_from_xy(imp.coord_exp[:,1],imp.coord_exp[:,0],crs = 'EPSG:4326'))
    assign_centroids_gdf(gdf,hazard,distance,threshold)

    try:
        haz_type = hazard.tag.haz_type
    except AttributeError:
        haz_type = hazard.haz_type
    setattr(imp,f'centr_{str(haz_type)}' , gdf['centr_' + haz_type] )




def smooth_monotonic(x,y,plot=False):
    """
    monotonic smoother based on https://stats.stackexchange.com/questions/467126/monotonic-splines-in-python
    x must be ordered increasing!

    """
    assert(len(y)==len(x))
    N=len(y)
    # Prepare bases (Imat) and penalty
    dd = 3
    E  = np.eye(N)
    D3 = np.diff(E, n = dd, axis=0)
    D1 = np.diff(E, n = 1, axis=0)
    la = 100
    kp = 10000000

    # Monotone smoothing
    ws = np.zeros(N - 1)

    for _ in range(30):
        Ws      = np.diag(ws * kp)
        mon_cof = np.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, y)
        ws_new  = (D1 @ mon_cof < 0.0) * 1
        dw      = np.sum(ws != ws_new)
        ws      = ws_new
        if(dw == 0): break

    if plot:
        # Monotonic and non monotonic fits
        z  = mon_cof
        z2 = np.linalg.solve(E + la * D3.T @ D3, y)
        # Plots
        plt.scatter(x, y, linestyle = 'None', color = 'gray', s = 0.5, label = 'raw data')
        plt.plot(x, z, color = 'red', label = 'monotonic smooth')
        plt.plot(x, z2, color = 'blue', linestyle = '--', label = 'unconstrained smooth')
        plt.legend(loc="lower right")
        plt.show()
    return mon_cof


