# -*- coding: utf-8 -*-
"""
Ploting functions for climada and non-climada objects used in scClim

"""

import numpy as np
import pandas as pd
import xarray as xr
import sys
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, ArrowStyle, Patch, Rectangle
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize,ListedColormap
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.lines import Line2D
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import FuncFormatter

from scipy.ndimage import gaussian_filter as g_filter
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union

# from climada.entity import ImpactFunc, ImpactFuncSet
from climada.engine import Impact
from climada import CONFIG
from climada.util.constants import CMAP_IMPACT

from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
import cartopy.io.img_tiles as cimgt
import cartopy.geodesic as cgeo
import geopandas
sys.path.append(str(CONFIG.local_data.func_dir))
import scClim as sc
import colorcet as cc
cmap_imp = CMAP_IMPACT
cmap_imp.set_under('white',alpha=0)
from scClim.plot_cmaps import CMAP_IMPACT_CC, CMAP_IMPACT_CHF
from scClim.constants import CH_EXTENT_EPSG2056, CMAP_VIR

data_dir = str(CONFIG.local_data.data_dir)

# %% General plotting functions
def plot_canton(ax,canton='Zürich',edgecolor = 'black',facecolor = "none",
                lakes=True,ch_border=None,zorder=0.5,lakeEdgeColor='none',
                lakeFaceColor='lightblue',union=False,**kwargs):

    ch_shp_path = str(CONFIG.ch_shp_path)
    reader = shpreader.Reader("%s/swissTLMRegio_KANTONSGEBIET_LV95.shp"%ch_shp_path)

    if canton=='all':
        sel_cantons = [place for place in reader.records() if
                       place.attributes["OBJEKTART"]=='Kanton' and
                       place.attributes["ICC"]=='CH']
        for sel_canton in sel_cantons:
            shape_feature = cf.ShapelyFeature([sel_canton.geometry],
                                               ccrs.epsg(2056), edgecolor=edgecolor,
                                               facecolor = facecolor,**kwargs)
            ax.add_feature(shape_feature,zorder=zorder)
    elif type(canton)==list:
        sel_cantons = [place for place in reader.records() if
                       place.attributes["NAME"] in canton]
        if union:
            union_geom = unary_union([sel_canton.geometry for sel_canton in sel_cantons])
            shape_feature = cf.ShapelyFeature([union_geom], ccrs.epsg(2056),
                                              edgecolor=edgecolor, facecolor=facecolor,
                                               **kwargs)
            ax.add_feature(shape_feature, zorder=zorder)
        else:
            for sel_canton in sel_cantons:
                shape_feature = cf.ShapelyFeature([sel_canton.geometry],
                                                ccrs.epsg(2056), edgecolor=edgecolor,
                                                facecolor = facecolor,**kwargs)
                ax.add_feature(shape_feature,zorder=zorder)
    elif type(canton)==str:
        sel_canton = [place for place in reader.records() if
                      place.attributes["NAME"]==canton][0]
        shape_feature = cf.ShapelyFeature([sel_canton.geometry],
                                           ccrs.epsg(2056), edgecolor=edgecolor,
                                           facecolor = facecolor,**kwargs)
        ax.add_feature(shape_feature)

    # add lake
    if lakes:
        add_lakes(ax,facecolor=lakeFaceColor,edgecolor=lakeEdgeColor,zorder=zorder)

    if ch_border:

        reader3 = shpreader.Reader("%s/swissTLMRegio_LANDESGEBIET_LV95.shp"%ch_shp_path)
        sel_ctry = [place for place in reader3.records() if
                    place.attributes["ICC"]=='CH'][0]
        shape_feature3 = cf.ShapelyFeature([sel_ctry.geometry],
                                           ccrs.epsg(2056), edgecolor=ch_border,
                                           facecolor = "none")
        ax.add_feature(shape_feature3,zorder = zorder-1)
        shape_feature4 = cf.ShapelyFeature([sel_ctry.geometry],
                                           ccrs.epsg(2056), edgecolor='None',
                                           facecolor = 'white')
        ax.add_feature(shape_feature4,zorder=0)

def add_lakes(ax,facecolor='lightblue',edgecolor='none',zorder=0.5,
              crs=ccrs.epsg(2056),min_size=1e7,**kwargs):
    reader2 = shpreader.Reader("%s/Hydrography/swissTLMRegio_Lake.shp"%str(CONFIG.ch_shp_path))
    geometry = reader2.geometries()
    geometry = np.array([g for g in geometry])
    lakesize = np.array([a.area for a in reader2.geometries()])
    geometry = geometry[lakesize>min_size]
    shape_feature2 = cf.ShapelyFeature(geometry,crs, edgecolor=edgecolor,
                                        facecolor = facecolor,**kwargs)
    ax.add_feature(shape_feature2,zorder=zorder)#0.5,2

def plot_country(ax, country='all', edgecolor = 'white', facecolor = 'none',
                   linewidth = 1, zorder = 0.5,**kwargs):
    """plot one or several countries provided in the shapefile

    Parameters
    ----------
    country: string, list of country iso code; available values: all, LI, AT, DE, CH, FR, IT
    """


    reader3 = shpreader.Reader("%s/swissTLMRegio_LANDESGEBIET_LV95.shp"%str(CONFIG.ch_shp_path))

    # all countries
    if country=='all':
        country = ['CH', 'DE', 'AT', 'LI', 'IT', 'FR']

    # list of countries
    if type(country)==list:
        for cntry in country:
            sel_country = [place for place in reader3.records() if
                      place.attributes["ICC"]==cntry][0]
            shape_feature3 = cf.ShapelyFeature([sel_country.geometry],
                                               ccrs.epsg(2056),
                                               edgecolor = edgecolor,
                                               facecolor = facecolor,
                                               linewidth = linewidth)
            ax.add_feature(shape_feature3, zorder=zorder,**kwargs)

    # one country only
    elif type(country) == str:
        sel_country = [place for place in reader3.records() if
                       place.attributes["ICC"]==country][0]
        shape_feature3 = cf.ShapelyFeature([sel_country.geometry],
                                           ccrs.epsg(2056),
                                           edgecolor = edgecolor,
                                           facecolor = facecolor,
                                           linewidth = linewidth)
        ax.add_feature(shape_feature3, zorder=zorder,**kwargs)


def init_fig(projection=ccrs.PlateCarree,figsize=None,ch_border = True,title=''):

    if not figsize: figsize = (9,6)
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':projection})
    ax.set(title=title)
    plot_canton(ax,edgecolor="none",ch_border='grey')
    return fig, ax

def plot_nc(nc,ax=None,fillna='none',extent = None,title='',discrete='none',vmax=None,
            vmin=None,borders=True,cbar_lbl='',pl_type='',crs='default',canton=None,
            logScale=False,cbar_horizontal=False,extend=None,cbar=True,return_plot=False,
            border_color='black',**kwargs):
    """
     Parameters
    ----------
    nc : xr.DataArray
        with 1 timestep only
    """


    if crs == 'default': #PlateCaree()
        transform = ccrs.PlateCarree()
        x_dim = 'lon'
        y_dim = 'lat'
        if 'clat' in nc.coords:
            nc = nc.rename({'clat':'lat','clon':'lon'})
    elif crs=='EPSG:2056':
        transform = ccrs.epsg(2056)
        x_dim = 'chx'
        y_dim = 'chy'
    elif crs=='EPSG:21781':
        transform = ccrs.epsg(21781)
        x_dim = 'chx'
        y_dim = 'chy'
    elif type(crs) == ccrs.AlbersEqualArea: #crs given by rxr object
        transform = crs #nc.rio.crs
        x_dim = 'x'
        y_dim = 'y'

    if not fillna=='none':
        nc = nc.fillna(fillna)
    if not discrete=='none':
        nc = nc>discrete
    if not ax:
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':transform})
    if extent: #
        ax.set_extent(extent,crs=transform)
        # lon_min, lon_max, lat_min, lat_max = extent
        # lon_cond = np.logical_and(nc.lon >= lon_min, nc.lon <= lon_max)
        # lat_cond = np.logical_and(nc.lat >= lat_min, nc.lat <= lat_max)
        # nc = nc.where(np.logical_and(lat_cond,lon_cond),drop=True)

    if logScale:
        vmin = vmin if vmin else nc.min().values
        vmax = vmax if vmax else nc.max().values
        vmin = max(1,vmin) #avoid log(0)
        vmax = max(1,vmax) #avoid empty norm
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        kwargs['norm'] = norm
        vmin = None; vmax = None
    else:
        norm = None

    if pl_type=='contour':
        plot = ax.contour(nc[x_dim],nc[y_dim],nc,transform=transform,
                          extend=extend,**kwargs)
    elif pl_type=='contourf':
        if nc.ndim == 2:
            plot = ax.contourf(nc[x_dim],nc[y_dim],nc,transform=transform,
                            extend=extend,**kwargs)
        elif nc.ndim == 1:
            plot = ax.tricontourf(nc[x_dim],nc[y_dim],nc,transform=transform,
                            extend=extend,**kwargs)
    elif pl_type=='bool_field':
        #create bool colormap
        if 'cmap' in kwargs:
            cmap = ListedColormap(['white',kwargs['cmap']])
            # cmap = plt.cm.get_cmap(kwargs['cmap']).copy()
            cmap.set_under('white',alpha=0)
            kwargs['cmap'] = cmap
        if not nc.max()==0:
            plot = ax.pcolormesh(nc[x_dim],nc[y_dim],nc,vmax=vmax,shading='auto',
                                 transform=transform,vmin=0.5,**kwargs)
    else:
        plot = ax.pcolormesh(nc[x_dim],nc[y_dim],nc,vmax=vmax,vmin=vmin,
                             transform=transform,shading='auto',**kwargs)
    if cbar:
        add_cbar(ax,plot,cbar_lbl,extend=extend,horizontal=cbar_horizontal)

    if borders:
        ax.add_feature(cf.BORDERS,edgecolor=border_color)
    if title:
        try:
            ax.set_title(title) #+nc.name+nc.time)
        except:
            ax.set_title(title+nc.name)
    if canton:
        plot_canton(ax,canton=canton)


    # axis.set_title("\n".join(wrap(tit)))
    # if fontsize:
        # cbar.ax.tick_params(labelsize=fontsize)
        # cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
        # for item in [axis.title, cbar.ax.xaxis.label, cbar.ax.yaxis.label]:
            # item.set_fontsize(fontsize)
    # else:
        # plt.colorbar(plot,ax=ax,shrink = cbar_shrink)
    if return_plot:
        return ax, plot
    else:
        return ax

def add_cbar(ax,plot,cbar_lbl='',extend=None,horizontal=False,pad=0.1,**kwgs):
        if horizontal:
            cbax = make_axes_locatable(ax).append_axes(
                'bottom', size="6.5%", pad=pad, axes_class=plt.Axes)
            cbar = plt.colorbar(plot, cax=cbax, orientation='horizontal',
                                extend=extend,**kwgs)
        else:
            cbax = make_axes_locatable(ax).append_axes(
                'right', size="6.5%", pad=pad, axes_class=plt.Axes)
            cbar = plt.colorbar(plot, cax=cbax, orientation='vertical',
                                extend=extend,**kwgs)
        cbar.set_label(cbar_lbl)

def add_cbar_v2(fig,ax,c_mappable,x0=0.3,y0=0,dx=0.7,dy=0.08,label="",ticks=None,
                labelsize=8,labelpad=None,orientation='horizontal',**kwargs):
    """add new colorbar to figure in a specific position"""

    #Define position of colorbar
    pos = ax.get_position()
    x0_ = pos.x0 + x0*pos.width
    y0_ = pos.y0 + y0*pos.height
    width = dx*pos.width
    height = dy*pos.height

    #Add new axis
    cbax2 = fig.add_axes((x0_,y0_,width,height))

    #Add colorbar
    cbar2 = plt.colorbar(c_mappable, cax=cbax2, orientation=orientation,**kwargs)#, extend=extend)
    cbar2.set_label(label, fontsize=labelsize,labelpad=labelpad)
    if ticks is not None:
        cbar2.set_ticks(ticks)
    cbar2.ax.tick_params(labelsize=labelsize)

def cut_cmap(cmap, minval=0.0, maxval=1.0, n=-1,return_weak=False):

    if n == -1:
        n = cmap.N
    new_cmap = LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n))**1.1)
    new_cmap_light = LinearSegmentedColormap.from_list(
         'truncLight({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n))**0.3)
    if return_weak:
        return new_cmap,new_cmap_light
    else:
        return new_cmap


def add_ax_arr(fig,ax,x0=0.9,y0=0.5,dx=0.2,dy=0,mutation_scale=100,
                color='lightgrey',lw=4,ec='grey',**kwargs):
    """Add an arrow to the axes (from the axes position)

    Args:
        ax (matplotlib.axis): b
        x0 (float, optional): x0. Defaults to 0.9.
        y0 (float, optional): y0. Defaults to 0.5.
        dx (float, optional): length in x direction as fraction of axis. Defaults to 0.2.
        dy (int, optional): length in y direction as fraction of axis. Defaults to 0.
        mutation_scale (int, optional): kwarg for FancyArrowPatch. Defaults to 100.
        color (str, optional): Color. Defaults to 'lightgrey'.
        lw (int, optional): linewidth. Defaults to 4.
        ec (str, optional): edgecolor. Defaults to 'grey'.
    """

    pos = ax.get_position()
    x0 = pos.x0 + x0*pos.width
    y0 = pos.y0 + y0*pos.height
    arrowstyle = kwargs.pop('arrowstyle',ArrowStyle('simple',head_length=0.4, head_width=1, tail_width=0.5))
    arrow = FancyArrowPatch((x0, y0), (x0+dx*pos.width, y0+dy*pos.height),
                            mutation_scale=mutation_scale,color=color,
                            lw=lw,ec=ec,arrowstyle=arrowstyle,**kwargs)

    fig.add_artist(arrow)


def scale_bar(ax,point,length,crs='EPSG:2056',dy_label=4e3,fontsize=10,line_kwargs=None):
    """plot scale bar on map

    Args:
        ax (plt.Axes): figure axes
        point (tuple): starting point of scale (in fig coordinates)
        length (int): length in coord units (m by default)
        crs (str, optional): CRS. Defaults to 'EPSG:2056'.
    """
    #get the geographic coordinates from the figure coordinates
    if not crs=='EPSG:2056':
        raise NotImplementedError('Only implemented for EPSG:2056')

    p_a_disp = ax.transAxes.transform(point)
    xy = ax.transData.inverted().transform(p_a_disp)

    scaleBar = Line2D((xy[0],xy[0]+length),(xy[1],xy[1]),
                      transform = ccrs.epsg(2056),color='black',**line_kwargs)
    ax.add_line(scaleBar)
    ax.text(xy[0]+length, xy[1]+dy_label, f'{length/1000:.0f} km',
            transform=ccrs.epsg(2056),fontsize=fontsize,ha='right',va='bottom')

def set_up_broken_axis(ylim1,ylim2,hspace=0.05,height_ratio=(1,3),figsize=None,
                       figAxAx=None):

    if figAxAx is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                    gridspec_kw={'height_ratios': height_ratio,'hspace':hspace})
    else:
        fig,ax1,ax2 = figAxAx
    # fig.subplots_adjust(hspace=hspace)  # adjust space between axes

    # plot the same data on both axes
    # ax1.plot(pts)
    # ax2.plot(pts)

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(ylim2)  # outliers only
    ax2.set_ylim(ylim1)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    if ylim1[1] == ylim2[0] and hspace==0: #do not draw break if axis is continous
        ax2.axhline(ylim1[1],color='k',lw=1,linestyle='--')
    else:
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    return fig, (ax1, ax2)

# %% Specific plotting functions

########################### Subproject E #####################################
def plot_crowdsourced_overview(crowd_sel,res_km = 2,norm = LogNorm(vmin=1,vmax=1e3),figsize=(6,4)):

    #Create plot
    prj = ccrs.epsg(2056) #ccrs.AlbersEqualArea(8.222665776, 46.800663464)
    fig,ax = plt.subplots(1,1,figsize=figsize,subplot_kw={'projection':prj},dpi=200)

    #Plot crowdsourced data (2d histogram)
    ext = sc.constants.CH_EXTENT_EPSG2056
    step = res_km*1e3
    bins = [np.arange(ext[0],ext[1],step=step),np.arange(ext[2],ext[3],step=step)]
    cmap = cut_cmap(plt.cm.get_cmap('Purples'),0.1,1)
    cmap.set_under('white',alpha=0)
    pl = ax.hist2d(crowd_sel['chx'],crowd_sel['chy'],bins=bins,cmap=cmap,norm=norm,transform=ccrs.epsg(2056))
    sc.plot_funcs.add_cbar(ax,pl[3],cbar_lbl='Number of reports')

    #add cantons
    col='brown'
    sc.plot_canton(ax,canton=['Zürich','Aargau','Luzern','Bern'],
                facecolor='none',edgecolor=col,zorder=2,ch_border='black',lakeEdgeColor='steelblue')
    #Add canton names
    canton_names = ['Zurich','Aargau','Lucerne','Berne']
    canton_locs = [(8.4,47.72),(7.85,47.63),(8.19,46.9),(7.1,47.35)]

    transform = ccrs.PlateCarree()#._as_mpl_transform(ax)
    for i in range(len(canton_names)):
        ax.annotate(canton_names[i], xy = canton_locs[i], xytext = canton_locs[i],
                    xycoords=transform,fontsize=10,zorder=3,color=col,
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

    #add scalebar
    y= 1090e3
    x = 2733e3
    scaleBar = Line2D((x,x+100e3),(y,y),transform = ccrs.epsg(2056),color='black',linewidth=2.5,zorder=6)
    ax.add_line(scaleBar)
    ax.text(x+50e3, y+5e3, '100 km', transform=ccrs.epsg(2056),fontsize=9,ha='center',va='bottom',weight='bold')

    # Add line from (6, 46.1) to (10, 47.5)
    pts_1 = [(6.5,45.7),(6.85,46.15),(8.1,46.75),(9.8,47.2),(11,47.3)]
    line = Line2D([x[0] for x in pts_1],[x[1] for x in pts_1],transform=ccrs.PlateCarree(),color='grey',linewidth=2)
    pts_2 = [(8,45.8),(8.35,46.3),(9.3,46.33),(10.5,45.9)]

    line2 = Line2D([x[0] for x in pts_2],[x[1] for x in pts_2],transform=ccrs.PlateCarree(),color='grey',linewidth=2)
    ax.add_line(line)
    ax.add_line(line2)

    ax.annotate('Alps', xy = (8.7,46.5), xytext = (8.7,46.5),xycoords=transform,fontsize=14,zorder=3,color='grey', rotation=10,
                path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])


    # #set figure options and save the figure
    gl=ax.gridlines(draw_labels=["bottom","left"], x_inline=False, y_inline=False,linestyle='--',
                    linewidth=0.5,color='darkgrey',xlabel_style={'color':'grey','size':8,'rotation':'horizontal'},
                    ylabel_style={'color':'grey','size':8})
    ax.set_extent(sc.constants.CH_EXTENT_EPSG2056,crs = ccrs.epsg(2056))
    return fig,ax

def plot_imp_comp(date,ds_KGV,ds_haz,second='radar',ds_haz_second=None,
                  var='PAA',canton_list=None):
    """
    Plot comparison of observed, crowdsourced and radar-based impacts

    Args:
        date (str): Date of event (format: 'YYYY-MM-DD')
        ds_KGV (xr.Dataset): Building insurance dataset (KGV)
        ds_haz (xr.Dataset): Hazard data (gridded crowdsourced reports)
        second (str, optional): Whether the second column contains radar data
            or an alternative version of crowdsourced reports. Default: 'radar'
        ds_haz_second (xr.Dataset, optional): alternative hazard data. Defaults to None.
        var (str, optional): PAA or MDR . Defaults to 'PAA'.
    """
    #Define variables for percentage of assets affected (PAA) or mean damage ratio (MDR)
    if var == 'PAA':
        var_name = "Percentage of assets affected"
        abs_str = 'n_count'
        abs_name = "No. of claims"
        ds_KGV['n_count_MESHS'] = ds_KGV['n_buildings_MESHS'].copy()
        vmin=0.01

    elif var == 'MDR':
        var_name = "Mean damage ratio"
        abs_str = "imp"
        abs_name = "Monetary damages [CHF]"
        ds_KGV['imp'] = ds_KGV['imp_observed'].copy()

        vmin=0.0001
    if canton_list is None:
        canton_list = ['Zürich','Aargau','Luzern','Bern']

    #Set up figure
    fig,axes = plt.subplots(3,3,figsize=(15,15),subplot_kw={'projection':ccrs.epsg(2056)},
                            dpi=80,gridspec_kw={'hspace':-0.15,'wspace':0.05,'width_ratios':[1,1,1]})



    # ----------- Plot Exposure -----------
    ax,plot = sc.plot_nc(ds_KGV['n_count_exposure'],ax=axes[0,0], title="Exposure & hazard",
               borders=False,cmap='Greys',vmin=1,return_plot=True,cbar=False)
    sc.plot_funcs.add_cbar_v2(fig,axes[0,0],plot,label="No. of buildings",y0=-0.021,dy=0.07)
    axes[0,0].text(-0.02,0.5,f"Observations",transform=axes[0,0].transAxes,ha='right',va='center',fontsize=12,rotation='vertical')

    #Alternative for first plot: Crowdsourced reports
    # _ = sc.crowd_process.plot_crowd(
    #     crowd_data[(crowd_data.hailday==date_dt) & (crowd_data.maxMZC_t15_r4>0 )].sort_values('size'),
    #     fig_ax=(fig,axes[0,0]),legend=False,gridlines=False
    #     )

    # ----------- Plot aboslute impacts -----------
    #Select the 95th percentile of the reported damage as max. value for colorbar
    q95_noNan = ds_KGV[abs_str].sel(date=date).where(ds_KGV[abs_str].sel(date=date)>0).quantile(0.95)
    cmap = sc.cut_cmap(get_cmap("Reds").copy(),minval=0.02)
    cmap.set_under('white')
    kwargs = {"cmap":cmap,"vmin":1,"vmax":q95_noNan,'borders':False}

    #Observed
    sc.plot_nc(ds_KGV[abs_str].sel(date=date),ax=axes[0,2],title=abs_name,**kwargs,cbar=False)
    #Crowdsourced
    sc.plot_nc(ds_KGV[f"{abs_str}_crowd"].sel(date=date),ax=axes[1,2],**kwargs,cbar=False)


    # ----------- Plot relative impacts -----------
    #Select 95th percentile of the relative impact as max. value for colorbar
    q95_noNan = ds_KGV[var].sel(date=date).where(ds_KGV[var].sel(date=date)>0).quantile(0.95)
    cmap2 = sc.cut_cmap(get_cmap("Purples").copy(),minval=0.05)
    cmap2.set_under('white')
    kwargs2 = {"cmap":cmap2,"vmax":q95_noNan,"vmin":vmin,'borders':False}

    #Define formatter for colorbar
    if q95_noNan>0.05:
        fmt = lambda x, pos: '{:.0%}'.format(x)
    elif q95_noNan>0.005:
        fmt = lambda x, pos: '{:.1%}'.format(x)
    else:
        fmt = lambda x, pos: '{:.2%}'.format(x)
    fmter = FuncFormatter(fmt)

    #Observed
    sc.plot_nc(ds_KGV[var].sel(date=date),ax=axes[0,1],title=var_name,**kwargs2,cbar=False)
    #Crowdsourced
    sc.plot_nc(ds_KGV[f'{var}_crowd'].sel(date=date),ax=axes[1,1],**kwargs2,cbar=False)


    # ----------- Plot hazard -----------
    # Convert date to datetime object
    date_dt = dt.datetime.strptime(date,'%Y-%m-%d').date()

    #Plot gridded crowdsourced data
    _,plot = sc.plot_nc(ds_haz.sel(time=date).h_smooth.where(ds_haz.sel(time=date).h_smooth>0),
                        ax=axes[1,0],borders=False,cbar_lbl='Hail size [mm]',cbar=False,return_plot=True)

    sc.plot_funcs.add_cbar_v2(fig,axes[1,0],plot,label="Hail size [mm]",
                              extend='neither',y0=-0.18) #ticks=[1,20,40,60],

    axes[1,0].text(-0.02,0.5,f"Crowdsourced",transform=axes[1,0].transAxes,
                   ha='right',va='center',fontsize=12,rotation='vertical')



    # ----------- Plot second hazard data (radar or crowdsourced) -----------
    if second == 'radar': #Radar-based
        # Absolute impacts
        _,plot=sc.plot_nc(ds_KGV[f"{abs_str}_MESHS"].sel(date=date),ax=axes[2,2],
                          cbar=False,return_plot=True,**kwargs)
        sc.plot_funcs.add_cbar_v2(fig,axes[2,2],plot,label=f"{abs_name}",
                                  x0=0,y0=-0.14,dx=1,dy=0.08,labelsize=10)

        #Relative impacts
        _,plot = sc.plot_nc(ds_KGV[f"{var}_MESHS"].sel(date=date),ax=axes[2,1],
                            cbar=False,return_plot=True,**kwargs2)
        sc.plot_funcs.add_cbar_v2(fig,axes[2,1],plot,label=f"{var}",
                                  x0=0,y0=-0.14,dx=1,dy=0.08,labelsize=10,format=fmter)

        # Hazard
        meshs=xr.open_dataset(f'{data_dir}/V5/MZC/MZC_X1d66_{date_dt.year}.nc').sel(time=date).MZC.clip(0,100)
        _,plot = sc.plot_nc(meshs.where(meshs>0),ax=axes[2,0],cmap='viridis',
                            borders=False,cbar_lbl='MESHS [mm]',cbar=False,return_plot=True)
        sc.plot_funcs.add_cbar_v2(fig,axes[2,0],plot,label="MESHS [mm]",y0=-0.18) #ticks=[20,50,75,100]

        axes[2,0].text(-0.02,0.5,f"Radar-based",transform=axes[2,0].transAxes,
                       ha='right',va='center',fontsize=12,rotation='vertical')

    elif type(second) == int: #Alternative crowdsourced version
        assert ds_haz_second is not None, "Must provide the second hazard dataset"
        # Absolute impacts
        sc.plot_nc(ds_KGV[f'{abs_str}_crowd_V{second}'].sel(date=date),
                   ax=axes[2,2],title=f"V{second} ",**kwargs)
        #Relative impacts
        sc.plot_nc(ds_KGV[f'{var}_crowd_V{second}'].sel(date=date),
                   ax=axes[2,1],title=f"V{second} ",**kwargs2)
        # Hazard
        sc.plot_nc(ds_haz_second.sel(time=date).h_smooth,ax=axes[2,0],
                   borders=False,title = f'Crowd gridded: V{second} ')


    labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    for ax,lbl in zip(axes.flatten(),labels):
        ax.set_extent(sc.constants.SUB_CH_EXTENT_2056_TIGHT, crs=ccrs.epsg(2056))
        sc.plot_canton(ax,canton=canton_list,edgecolor='grey',facecolor='none',
                       zorder=6)
        txt = ax.text(0.02,0.97,f"{lbl}", transform=ax.transAxes,ha='left',
                    va='top',fontsize=14,weight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    return fig


def plot_imp_comp_V3(date, ds_KGV, ds_haz, canton_list=None):
    """
    Plot comparison of observed, crowdsourced and radar-based impacts

    Args:
        date (str): Date of event (format: 'YYYY-MM-DD')
        ds_KGV (xr.Dataset): Building insurance dataset (KGV)
        ds_haz (xr.Dataset): Hazard data (gridded crowdsourced reports)
        var (str, optional): PAA or MDR. Defaults to 'PAA'.
        canton_list (list, optional): List of cantons to plot. Defaults to None.
    """
    text_kw = dict(ha='right', va='center', fontsize=12, rotation='vertical')
    if canton_list is None:
        canton_list = ['Zürich', 'Aargau', 'Luzern', 'Bern']

    # Set up figure
    fig, axes = plt.subplots(5, 3, figsize=(15, 25), subplot_kw={'projection': ccrs.epsg(2056)},
                             dpi=80, gridspec_kw={'hspace': -0.35, 'wspace': 0.05, 'height_ratios': [1, 1, 1, 1, 1]})


    # ----------- Plot Exposure -----------
    ax, plot = sc.plot_nc(ds_KGV['n_count_exposure'], ax=axes[0, 0], title=f"Observations",
                          borders=False, cmap='Greys', vmin=1, return_plot=True, cbar=False)
    sc.plot_funcs.add_cbar_v2(fig, axes[0, 0], plot, label="No. of buildings", y0=0.09,labelpad=0) #dy=0.07)
    rect = Rectangle((0.3, 0), 0.75, 0.16,  # (x, y, width, height) in axes coordinates
                     facecolor='white', edgecolor='grey', alpha=1, zorder=10,
                        transform=axes[0, 0].transAxes)
    axes[0, 0].add_patch(rect)

    axes[0, 0].text(-0.02, 0.5, "Exposure & hazard", transform=axes[0, 0].transAxes, **text_kw)

    # ----------- Plot hazard -----------
    # Convert date to datetime object
    date_dt = dt.datetime.strptime(date, '%Y-%m-%d').date()

    # Plot gridded crowdsourced data
    _, plot = sc.plot_nc(ds_haz.sel(time=date).h_smooth.where(ds_haz.sel(time=date).h_smooth > 0),
                         ax=axes[0, 1], borders=False, cbar_lbl='Hail size [mm]',
                           cbar=False, return_plot=True,title='Crowdsourced')

    sc.plot_funcs.add_cbar_v2(fig, axes[0, 1], plot, label="Hail size [mm]",
                              extend='neither', y0=-0.04,labelpad=0)  # ticks=[1,20,40,60],
    rect = Rectangle((0.3, 0), 0.75, 0.16,  # (x, y, width, height) in axes coordinates
                     facecolor='white', edgecolor='grey', alpha=1, zorder=10,
                     transform=axes[0,1].transAxes)
    axes[0,1].add_patch(rect)

    meshs = xr.open_dataset(f'{data_dir}/V5/MZC/MZC_X1d66_{date_dt.year}.nc').sel(time=date).MZC.clip(0, 100)
    _, plot = sc.plot_nc(meshs.where(meshs > 0), ax=axes[0, 2], cmap='viridis',
                         borders=False, cbar_lbl='MESHS [mm]', cbar=False,
                         return_plot=True,title='Radar-based')
    sc.plot_funcs.add_cbar_v2(fig, axes[0, 2], plot, label="MESHS [mm]", y0=-0.04,labelpad=0)  # ticks=[20,50,75,100]
    rect = Rectangle((0.3, 0), 0.75, 0.16,  # (x, y, width, height) in axes coordinates
                     facecolor='white', edgecolor='grey', alpha=1, zorder=10,
                        transform=axes[0,2].transAxes)
    axes[0,2].add_patch(rect)

    # Define variables for percentage of assets affected (PAA) or mean damage ratio (MDR)
    for i, var in enumerate(['PAA', 'MDR']):
        if var == 'PAA':
            var_name = "Percentage of assets affected"
            abs_str = 'n_count'
            abs_name = "No. of claims"
            ds_KGV['n_count_MESHS'] = ds_KGV['n_buildings_MESHS'].copy()
            vmin = 0.01
            cmaps = ["Reds", "Purples"]

        elif var == 'MDR':
            var_name = "Mean damage ratio"
            abs_str = "imp"
            abs_name = "Monetary damages [CHF]"
            ds_KGV['imp'] = ds_KGV['imp_observed'].copy()
            vmin = 0.0001
            cmaps = ["Oranges","Greens"]

        # ----------- Plot absolute impacts -----------
        # Select the 95th percentile of the reported damage as max. value for colorbar
        q95_noNan = ds_KGV[abs_str].sel(date=date).where(ds_KGV[abs_str].sel(date=date) > 0).quantile(0.95)
        cmap = sc.cut_cmap(get_cmap(cmaps[0]).copy(), minval=0.02)
        cmap.set_under('white')
        kwargs = {"cmap": cmap, "vmin": 1, "vmax": q95_noNan, 'borders': False}

        # Observed
        sc.plot_nc(ds_KGV[abs_str].sel(date=date), ax=axes[2 + 2 * i, 0], **kwargs, cbar=False)
        axes[2+2*i,0].text(-0.02, 0.5, abs_name, transform=axes[2+2*i,0].transAxes, **text_kw)
        # Crowdsourced
        sc.plot_nc(ds_KGV[f"{abs_str}_crowd"].sel(date=date), ax=axes[2 + 2 * i, 1], **kwargs, cbar=False)

        # ----------- Plot relative impacts -----------
        # Select 95th percentile of the relative impact as max. value for colorbar
        q95_noNan = ds_KGV[var].sel(date=date).where(ds_KGV[var].sel(date=date) > 0).quantile(0.95)
        cmap2 = sc.cut_cmap(get_cmap(cmaps[1]).copy(), minval=0.05)
        cmap2.set_under('white')
        kwargs2 = {"cmap": cmap2, "vmax": q95_noNan, "vmin": vmin, 'borders': False}

        # Define formatter for colorbar
        if q95_noNan > 0.05:
            fmt = lambda x, pos: '{:.0%}'.format(x)
        elif q95_noNan > 0.005:
            fmt = lambda x, pos: '{:.1%}'.format(x)
        else:
            fmt = lambda x, pos: '{:.2%}'.format(x)
        fmter = FuncFormatter(fmt)

        # Observed
        sc.plot_nc(ds_KGV[var].sel(date=date), ax=axes[1+2*i,0], **kwargs2, cbar=False)
        axes[1+2*i,0].text(-0.02, 0.5, var_name, transform=axes[1+2*i,0].transAxes, **text_kw)
        # Crowdsourced
        sc.plot_nc(ds_KGV[f'{var}_crowd'].sel(date=date), ax=axes[1 + 2 * i, 1], **kwargs2, cbar=False)

        # ----------- Plot second hazard data (radar) -----------
        cbar_kw = dict(x0=1.02, y0=-0.02, dx=0.07, dy=1.04, labelsize=10,orientation='vertical')
        # Absolute impacts
        _, plot = sc.plot_nc(ds_KGV[f"{abs_str}_MESHS"].sel(date=date), ax=axes[2 + 2 * i, 2],
                             cbar=False, return_plot=True, **kwargs)
        sc.plot_funcs.add_cbar_v2(fig, axes[2 + 2 * i, 2], plot, label=f"{abs_name}",
                                  **cbar_kw
                                  )

        # Relative impacts
        _, plot = sc.plot_nc(ds_KGV[f"{var}_MESHS"].sel(date=date), ax=axes[1 + 2 * i, 2],
                             cbar=False, return_plot=True, **kwargs2)
        sc.plot_funcs.add_cbar_v2(fig, axes[1 + 2 * i, 2], plot, label=f"{var}",
                                  format=fmter,**cbar_kw)

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)',
              '(k)', '(l)', '(m)', '(n)', '(o)']
    for ax, lbl in zip(axes.flatten(), labels):
        ax.set_extent(sc.constants.SUB_CH_EXTENT_2056_TIGHT, crs=ccrs.epsg(2056))
        sc.plot_canton(ax, canton=canton_list, edgecolor='grey', facecolor='none',
                       zorder=6)
        txt = ax.text(0.02, 0.97, f"{lbl}", transform=ax.transAxes, ha='left',
                      va='top', fontsize=14, weight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    return fig


def plot_filtering(date,crowd_data,crowd_raw,ds_haz,fs=None,return_fig=False,
                   title=True):
    """
    Plot the filtering process of crowd data: 1) MESHS/POH based filter, 2) DBSCAN

    """
    crowd_plot = crowd_data[(crowd_data.hailday==date) & (crowd_data.cluster!=-1)]
    crowd_excl = crowd_data[(crowd_data.hailday==date) & (crowd_data.cluster==-1)]

    #Plot sizes on only clustered points
    fig,axes = plt.subplots(2,3,figsize=(18,10),subplot_kw={'projection': ccrs.epsg(2056)},
                            dpi=80,gridspec_kw={'hspace':0.1,'wspace':-0.02})

    #Raw data
    _ = sc.crowd_process.plot_crowd(crowd_raw[(crowd_raw.hailday==date)],fig_ax=(fig,axes[0,0]),gridlines=False,legend=False)

    #MESHS/POH filtered data
    _ = sc.crowd_process.plot_crowd(crowd_data[(crowd_data.hailday==date) ],fig_ax=(fig,axes[0,1]),gridlines=False,legend=False)

    #Clustering
    axes[1,1].scatter(crowd_plot.chx,crowd_plot.chy,c=crowd_plot.cluster,cmap='tab20c',
                    s=10,transform=ccrs.epsg(2056),zorder=30)
    axes[1,1].scatter(crowd_excl.chx,crowd_excl.chy,c='black',s=10,
                    transform=ccrs.epsg(2056),zorder=30)
    cols= ['tab:blue','tab:orange','tab:green']
    patchList = [None,None]
    patchList[0] = [Line2D([], [], color=c, marker='o', linestyle='None',markersize=5) for c in cols]
    patchList[1] = Line2D([], [], color='black', marker='o', linestyle='None',markersize=None)
    axes[1,1].legend(handles=patchList, labels=['Clustered reports','Outliers'],
            handler_map = {list: HandlerTuple(None)},loc='lower right',fontsize=fs)


    #Plot sizes on only clustered points
    _ = sc.crowd_process.plot_crowd(crowd_plot,fig_ax=(fig,axes[0,2]),
                                    gridlines=False,loc='lower right',
                                    bbox_to_anchor=(1.02, -0.02),all_lables=True)


    #Plot POH and MESHS
    year = date.year
    meshs=xr.open_dataset(f'{data_dir}/V5/MZC/MZC_X1d66_{year}.nc').sel(time=date.strftime("%Y-%m-%d")).MZC
    poh=xr.open_dataset(f'{data_dir}/V5/BZC/BZC_X1d66_{year}.nc').sel(time=date.strftime("%Y-%m-%d")).BZC
    sc.plot_nc(poh,ax=axes[1,0],cmap='lightblue',discrete=0,pl_type='bool_field',cbar=False,borders=False)
    ax,plot = sc.plot_nc(meshs.where(meshs>20),ax=axes[1,0],cmap='viridis',cbar=False,borders=False,return_plot=True,vmin=20,vmax=80)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0+0.05*pos.width, pos.y0 - 0.25 * pos.height, pos.width*0.9, 0.1 * pos.height])
    cb = plt.colorbar(plot,cax=cax,orientation='horizontal',label='MESHS [mm]')
    cb.set_label('MESHS [mm]',fontsize=fs)
    cb.ax.tick_params(labelsize=fs)

    #Plot poh legend
    patch = Patch(color="lightblue",alpha=1, label=f'Probability of Hail > 0% ')
    legend=ax.legend(handles=[patch],loc='lower right',fontsize=fs)
    legend.get_frame().set_alpha(1)
    legend.set_zorder(8)

    # #Plot contours within first plot
    # sc.plot_nc((sc.verification.maximum_filter_nc(poh,radius=4)>0),ax = axes[0,0],pl_type='bool_field',zorder=0.5,cmap='lightgreen',cbar=False)
    # sc.plot_nc((sc.verification.maximum_filter_nc(meshs,radius=4)>20),ax = axes[0,0],pl_type='bool_field',zorder=0.5,cmap='green',cbar=False)

    #Plot gridded hazard data in last subplot
    ax,plot = sc.plot_nc(ds_haz['h_smooth'].where(ds_haz['h_smooth']>0).sel(time=date.strftime("%Y-%m-%d")),
                         ax=axes[1,2],cbar=False,borders=False,cmap='viridis',
                         return_plot=True,zorder=0.5)
    pos = ax.get_position()



    cax = fig.add_axes([pos.x0+0.05*pos.width, pos.y0 - 0.25 * pos.height, pos.width*0.9, 0.1 * pos.height])
    cb2=plt.colorbar(plot,cax=cax,orientation='horizontal')
    cb2.set_label('Gridded hail size [mm]',fontsize=fs)
    cb2.ax.tick_params(labelsize=fs)

    labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
    for ax,lbl in zip(axes.flatten(),labels):
        ax.set_extent(sc.constants.SUB_CH_EXTENT_2056, crs=ccrs.epsg(2056))
        sc.plot_canton(ax,canton=['Bern','Zürich','Luzern','Aargau'],edgecolor='grey',lakeEdgeColor='royalblue',lakeFaceColor='lightblue')
        txt = ax.text(0.02,0.97,f"{lbl}", transform=ax.transAxes,ha='left',
                    va='top',fontsize=14,weight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    #Add connecting arrows
    sc.plot_funcs.add_ax_arr(fig,axes[0,0],x0=0.94,dx=0.2)
    sc.plot_funcs.add_ax_arr(fig,axes[0,1],x0=0.94,dx=0.2)
    # sc.plot_funcs.add_ax_arr(fig,axes[0,1],x0=0.5,y0=0.02,dx=0,dy=-0.2)
    sc.plot_funcs.add_ax_arr(fig,axes[0,2],x0=0.5,y0=0.05,dx=0,dy=-0.23)

    sc.plot_funcs.add_ax_arr(fig,axes[1,0],x0=0.5,y0=0.95,dx=0,dy=0.24,arrowstyle=ArrowStyle("<|-|>",head_length=0.2, head_width=0.2))
    sc.plot_funcs.add_ax_arr(fig,axes[1,1],x0=0.5,y0=0.95,dx=0,dy=0.24,arrowstyle=ArrowStyle("<|-|>",head_length=0.2, head_width=0.2)) #x0=0.93,y0=0.96,dx=0.15,dy=0.15)
    if title:
        fig.suptitle(f"{date.strftime('%Y-%m-%d')}",y=0.95,weight='bold',fontsize=fs)
    axes[0,0].set_title('Radar-based filter',fontsize=fs)
    axes[0,1].set_title('4D-DBSCAN clustering',fontsize=fs)
    axes[0,2].set_title('Gridding of filtered reports',fontsize=fs)

    axes[1,2].spines['geo'].set_linewidth(3)
    if return_fig:
        return fig,axes


def plot_filtering_V2(date,crowd_data,crowd_raw,ds_haz,fs=None,return_fig=False,
                   title=True):
    """
    Plot the filtering process of crowd data: 1) MESHS/POH based filter, 2) DBSCAN

    """
    crowd_plot = crowd_data[(crowd_data.hailday==date) & (crowd_data.cluster!=-1)]
    crowd_excl = crowd_data[(crowd_data.hailday==date) & (crowd_data.cluster==-1)]

    #Plot sizes on only clustered points
    fig,axes = plt.subplots(2,3,figsize=(18,10),subplot_kw={'projection': ccrs.epsg(2056)},
                            dpi=80,gridspec_kw={'hspace':0.1,'wspace':-0.02})

    #Raw data
    _ = sc.crowd_process.plot_crowd(crowd_raw[(crowd_raw.hailday==date)],fig_ax=(fig,axes[0,0]),gridlines=False,legend=False)

    #MESHS/POH filtered data
    _ = sc.crowd_process.plot_crowd(crowd_data[(crowd_data.hailday==date) ],fig_ax=(fig,axes[0,1]),gridlines=False,legend=False)

    #Clustering
    axes[1,1].scatter(crowd_plot.chx,crowd_plot.chy,c=crowd_plot.cluster,cmap='tab20c',
                    s=10,transform=ccrs.epsg(2056),zorder=30)
    axes[1,1].scatter(crowd_excl.chx,crowd_excl.chy,c='black',s=10,
                    transform=ccrs.epsg(2056),zorder=30)
    cols= ['tab:blue','tab:orange','tab:green']
    patchList = [None,None]
    patchList[0] = [Line2D([], [], color=c, marker='o', linestyle='None',markersize=5) for c in cols]
    patchList[1] = Line2D([], [], color='black', marker='o', linestyle='None',markersize=None)
    axes[1,1].legend(handles=patchList, labels=['Clustered reports','Outliers'],
            handler_map = {list: HandlerTuple(None)},loc='lower right',fontsize=fs)


    #Plot sizes on only clustered points
    _ = sc.crowd_process.plot_crowd(crowd_plot,fig_ax=(fig,axes[0,2]),
                                    gridlines=False,loc='lower right',
                                    bbox_to_anchor=(1.02, -0.02),all_lables=True)


    #Plot POH and MESHS
    year = date.year
    meshs=xr.open_dataset(f'{data_dir}/V5/MZC/MZC_X1d66_{year}.nc').sel(time=date.strftime("%Y-%m-%d")).MZC
    poh=xr.open_dataset(f'{data_dir}/V5/BZC/BZC_X1d66_{year}.nc').sel(time=date.strftime("%Y-%m-%d")).BZC
    sc.plot_nc(poh,ax=axes[1,0],cmap='lightblue',discrete=0,pl_type='bool_field',cbar=False,borders=False)
    ax,plot = sc.plot_nc(meshs.where(meshs>20),ax=axes[1,0],cmap='viridis',cbar=False,borders=False,return_plot=True,vmin=20,vmax=80)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0+0.05*pos.width, pos.y0 - 0.25 * pos.height, pos.width*0.9, 0.1 * pos.height])
    cb = plt.colorbar(plot,cax=cax,orientation='horizontal',label='MESHS [mm]')
    cb.set_label('MESHS [mm]',fontsize=fs)
    cb.ax.tick_params(labelsize=fs)

    #Plot poh legend
    patch = Patch(color="lightblue",alpha=1, label=f'Probability of Hail > 0% ')
    legend=ax.legend(handles=[patch],loc='lower right',fontsize=fs)
    legend.get_frame().set_alpha(1)
    legend.set_zorder(8)

    # #Plot contours within first plot
    # sc.plot_nc((sc.verification.maximum_filter_nc(poh,radius=4)>0),ax = axes[0,0],pl_type='bool_field',zorder=0.5,cmap='lightgreen',cbar=False)
    # sc.plot_nc((sc.verification.maximum_filter_nc(meshs,radius=4)>20),ax = axes[0,0],pl_type='bool_field',zorder=0.5,cmap='green',cbar=False)

    #Plot gridded hazard data in last subplot
    ax,plot = sc.plot_nc(ds_haz['h_smooth'].where(ds_haz['h_smooth']>0).sel(time=date.strftime("%Y-%m-%d")),
                         ax=axes[1,2],cbar=False,borders=False,cmap='viridis',
                         return_plot=True,zorder=0.5)
    pos = ax.get_position()



    cax = fig.add_axes([pos.x0+0.05*pos.width, pos.y0 - 0.25 * pos.height, pos.width*0.9, 0.1 * pos.height])
    cb2=plt.colorbar(plot,cax=cax,orientation='horizontal')
    cb2.set_label('Gridded hail size [mm]',fontsize=fs)
    cb2.ax.tick_params(labelsize=fs)

    labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
    for ax,lbl in zip(axes.flatten(),labels):
        ax.set_extent(sc.constants.SUB_CH_EXTENT_2056, crs=ccrs.epsg(2056))
        sc.plot_canton(ax,canton=['Bern','Zürich','Luzern','Aargau'],edgecolor='grey',lakeEdgeColor='royalblue',lakeFaceColor='lightblue')
        txt = ax.text(0.02,0.97,f"{lbl}", transform=ax.transAxes,ha='left',
                    va='top',fontsize=14,weight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    #Add connecting arrows
    # sc.plot_funcs.add_ax_arr(fig,axes[0,0],x0=0.94,dx=0.2,color='lightblue',ec='tab:blue')
    # sc.plot_funcs.add_ax_arr(fig,axes[0,1],x0=0.94,dx=0.2)
    # sc.plot_funcs.add_ax_arr(fig,axes[0,1],x0=0.5,y0=0.02,dx=0,dy=-0.2)
    # sc.plot_funcs.add_ax_arr(fig,axes[0,2],x0=0.5,y0=0.05,dx=0,dy=-0.23)

    # sc.plot_funcs.add_ax_arr(fig,axes[1,0],x0=0.5,y0=0.95,dx=0,dy=0.24,arrowstyle=ArrowStyle("<|-|>",head_length=0.2, head_width=0.2))
    # sc.plot_funcs.add_ax_arr(fig,axes[1,1],x0=0.5,y0=0.95,dx=0,dy=0.24,arrowstyle=ArrowStyle("<|-|>",head_length=0.2, head_width=0.2)) #x0=0.93,y0=0.96,dx=0.15,dy=0.15)

    if title:
        fig.suptitle(f"{date.strftime('%Y-%m-%d')}",y=0.95,weight='bold',fontsize=fs)

    axes[0,0].text(0.8,1.02,f"Step 1: Radar-based filter",transform=axes[0,0].transAxes,fontsize=fs)#,color='tab:blue')
    axes[0,1].text(0.8,1.02,f"Step 2: 4D-DBSCAN clustering",transform=axes[0,1].transAxes,fontsize=fs)
    axes[0,2].text(1.02, 0, f"Step 3: Gridding of filtered reports",
                  transform=axes[0,2].transAxes, fontsize=fs,
                  rotation=-90, ha='left', va='center')

    # Make frame of axes[1,0] blue
    # axes[1,0].spines['geo'].set_edgecolor('tab:blue')
    # axes[1,0].spines['geo'].set_linewidth(2)

    # axes[0,0].set_title('Radar-based filter',fontsize=fs,)
    # axes[0,1].set_title('4D-DBSCAN clustering',fontsize=fs)
    # axes[0,2].set_title('Gridding of filtered reports',fontsize=fs)

    axes[1,2].spines['geo'].set_linewidth(3)
    if return_fig:
        return fig,axes


def plot_removed_reports(crowd_data,removed_tennisballs,col='red',s=10,fig_ax=None,dates=None):
    """
    Helper function to plot removed tennis ball reports (if threshold-based method is chosen)
    """
    if dates is None:
        dates = [dt.date(2021,6,28)]
    for date in dates: #:,dt.date(2021,6,21),dt.date(2021,7,12)]:
        if fig_ax is None:
            fig,ax = sc.crowd_process.plot_crowd(crowd_data[(crowd_data.hailday==date)].sort_values('size'),dpi=140)
        else:
            fig,ax = fig_ax
        ax.set_extent(sc.constants.SUB_CH_EXTENT_2056, crs=ccrs.epsg(2056))
        ax.scatter(x=[2650e3,2650e3,2660e3,2660e3],y=[1200e3,1210e3,1200e3,1210e3],transform=ccrs.epsg(2056),color='green',s=4,zorder=20)
        removed_at_date = removed_tennisballs[removed_tennisballs.hailday==date]
        ax.scatter(removed_at_date.chx,removed_at_date.chy,transform=ccrs.epsg(2056),color=col,s=s,zorder=21,marker='x')
        ax.set(title=f'Removed tennis balls on {date}')
    return fig,ax

def plot_poh_sel(date,haz_poh,poh,exp_dmg,extent=None,poh_level=80):
    """
    Plot POH (+-2d) and reported damages per
    """
    raise DeprecationWarning('This function is deprecated. See event_definition.py for pre-processing plots')


def plot_event(date,imp,haz,act_dmgs,exp=None,dur_path=None,projection=None,extent=None,
    pl_type='default',canton=None,ch2056_filter = None,draw_grid=False,haz_var=None,
    vmax=0,vmax_haz=None,cmap_log=False,sel_dates=None,haz2=None,**kwargs):
    """Plot various plots per event

    Parameters
    ----------
    date : datetime
        date
    imp : Impact
        CLIMADA impact
    haz : Hazard
        CLIMADA hazard (MESHS)
    act_dmgs : dict of CLIMADA Exposure OR CLIMADA impact object
        Dictionary of actual damages read into a CLIMADA Exposure object
        The keys are the dates, formated as strings "ddmmYYYY"
    exp : CLIMADA Exposure
        Exposure
    dur_path : str
        path to duration files with YYYY=year and XXX = variable
    projection : cartopy.crs
        default = ccrs.PlateeCaree()
    extent : list
        [lon_min,lon_max,lat_min, lat_max]
    pl_type : str
        'default' to plot 1d, including duration
        'temp_ev' to ploat also +- 1 day, but no duration
        'sel_dates' to plot only the selected dates
    canton : str
        which (if any) canton borders to plot
    vmax: float
        maximum value for colorbar
        if vmax=0: set vmax as maximum of observed and modelled damages (default)
    Returns
    -------
    plot : matplotlib.figure.Figure
        Figure with plots
    """

    #Determine event IDs
    event_id = haz.event_id[haz.date==date.toordinal()]
    if not imp.event_id[haz.date==date.toordinal()] == event_id:
        ValueError("Event id is not consistent between Hazard and Impact")

    #set Projection
    if not projection:
        projection = ccrs.PlateCarree()

    #turn off borders if canton is plotted explicitly
    shapes = True # shapes = False if canton else True

    #determine unit
    assert(imp.unit==act_dmgs.unit)

    #set n plot columns
    pl_cols = 3 if haz2 is None else 4

    #Set vmin_for logatithmic impact scale
    if cmap_log:
        if imp.unit=='CHF' or imp.unit=='USD':
            vmin_log = 1e2
        elif imp.unit=='':
            vmin_log = 1
        else:
            raise ValueError('Impact Unit not recognized')

    #Initialize figure
    if pl_type == 'default':
        fig,axes = plt.subplots(2,pl_cols,gridspec_kw={'hspace':-0.1,'wspace':0.35},
                                subplot_kw={'projection':projection},figsize=(4*pl_cols,8))
        #plot duration
        if dur_path:
            for var,v_name, col in zip(['MZC','BZC'],['Duration of MESHS>20mm','Duration of POH>80%'],[1,2]):
                duration = dur_path.replace('YEAR',str(date.year)).replace('XXX',var)
                if os.path.exists(duration):
                    dur_da = xr.open_dataarray(duration).sel(time=date)*5
                    if ch2056_filter:
                        dur_da=dur_da.sel(chx=slice(ch2056_filter[0],ch2056_filter[1]),
                                          chy=slice(ch2056_filter[2],ch2056_filter[3]))
                    plot = axes[0,col].pcolormesh(dur_da.lon,dur_da.lat,dur_da)
                    axes[0,col].add_feature(cf.BORDERS)
                    axes[0,col].set_title(v_name)
                    add_cbar(axes[0,col],plot,'Duration (min)')
                    #plt.colorbar(plot,ax=axes[0,col],shrink = 0.6)
                    dur_da.close()
                else:
                    axes[0,col].annotate(var+ " duration files\ndo not exist",
                    xy=(0.1, 0.5), xycoords='axes fraction')
        #plot expsure
        if exp:
            exp.plot_hexbin(gridsize = 40,axis = axes[0,0],linewidths=0.05,extent=extent)
            axes[0,0].set_title('Exposure')
        #determine event ids
        ev_ids = [event_id]
        if isinstance(act_dmgs,Impact):
            ev_ids_observed = [act_dmgs.event_id[act_dmgs.date==date.toordinal()]]
        row = 1
        str_dates = [dt.datetime.strftime(date, "%d%m%Y")]

    elif pl_type == 'temp_ev':
        #plot Hazard at +- 1 day
        #haz.plot_intensity(event=event_id-1,axis=axes[0,0])
        #haz.plot_intensity(event=event_id+1,axis=axes[2,0])

        ev_ids = [imp.event_id[imp.date==(date+dt.timedelta(days=inc_day)).toordinal()]
                  for inc_day in [-1,0,1]]
        str_dates = [dt.datetime.strftime(date+dt.timedelta(days=inc_day), "%d%m%Y")
                     for inc_day in [-1,0,1]]
        if isinstance(act_dmgs,Impact):
            ev_ids_observed = [act_dmgs.event_id[act_dmgs.date==(date+dt.timedelta(days=inc_day)).toordinal()]
                               for inc_day in [-1,0,1]]

        #remove day +1 and day -1 if there is neither modelled nor reported damages
        for i in [2,0]:

            if isinstance(act_dmgs,Impact):
                rep_imp_is_zero = len(ev_ids_observed[i])==0
            elif isinstance(act_dmgs,dict):
                rep_imp_is_zero = str_dates[i] not in act_dmgs.keys()

            mod_imp_is_zero = (len(ev_ids[i])==0) or (imp.at_event[imp.event_id==ev_ids[i]][0] == 0)

            if mod_imp_is_zero and rep_imp_is_zero:
                del ev_ids[i]
                del str_dates[i]
                if isinstance(act_dmgs,Impact):
                    del ev_ids_observed[i]

        fig,axes = plt.subplots(len(ev_ids),pl_cols,gridspec_kw={'hspace':0.1,'wspace':0.3},
                                subplot_kw={'projection':projection},
                                figsize=(14/3*pl_cols,4*len(ev_ids)))
        if len(ev_ids)==1: axes=np.expand_dims(axes,axis=0)
        row = 0

    elif pl_type == 'sel_dates':
        assert(date in sel_dates)
        ev_ids = [haz.event_id[haz.date==date.toordinal()] for date in sel_dates]
        ev_ids_observed = [act_dmgs.event_id[act_dmgs.date==date.toordinal()] for date in sel_dates]
        fig,axes = plt.subplots(len(ev_ids),pl_cols,gridspec_kw={'hspace':-0.2,'wspace':0.3},
                                subplot_kw={'projection':projection},figsize=(14/3*pl_cols,4*len(ev_ids)))
        row = 0

    for i,ev_id in enumerate(ev_ids):



        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            #plot Hazard
            haz.plot_intensity(event=ev_id,axis=axes[row+i,0],vmin=1e-5,cmap = CMAP_VIR,vmax=vmax_haz)

            #plot second hazard
            if haz2 is not None:
                haz2.plot_intensity(event=ev_id,axis=axes[row+i,3],vmin=1e-5,cmap = CMAP_VIR,vmax=vmax_haz)

        #set label
        if haz_var: fig.axes[-1].set_ylabel(haz_var)

        # gl = axes[row+i,0].gridlines(draw_labels=True,transform=projection,
          # #xlocs=lon_grid,ylocs=lat_grid,
          # x_inline=False,y_inline=False,
          # color='k',linestyle='dotted')

        #plot impact (only if it is available)
        if vmax==0:
            vmax_ = None
            n_loop = 2
        else:
            vmax_ = vmax
            n_loop = 1

        for loop in range(n_loop):


            #clear plots if looping through the second time
            if loop==1:
                axes[row+i,1].clear()
                axes[row+i,2].clear()
                if 'cax1' in locals(): cax1.remove()
                if 'cax2' in locals(): cax2.remove()
            try:
                if imp.at_event[imp.event_id==ev_id][0] == 0:
                    raise ValueError("No impact at this date -> continue to except statement")
                if cmap_log:
                    norm = colors.LogNorm(vmin=vmin_log, vmax=vmax_)
                else:
                    norm = colors.Normalize(vmin=1, vmax=vmax_)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    imp.plot_hexbin_impact_exposure(event_id=ev_id,gridsize=50,
                                                    axis=axes[row+i,1],linewidths=0.05,
                                                    norm=norm,extent=extent,cmap=CMAP_IMPACT_CC,
                                                    shapes=shapes,**kwargs)
                axes[row+i,1].set_title(f'Modelled dmg:{imp.at_event[imp.event_id==ev_id][0]:.1e} {imp.unit}')
                cax1 = fig.get_children()[-1]
                vmax1= cax1.dataLim.get_points()[1,1]
            except:
                axes[row+i,1].annotate('Modelled damages are zero\nor unknown',
                xy=(0.1, 0.5), xycoords='axes fraction')
                vmax1=0
                #raise TypeError("imp argument is not CLIMADA Impact class")
                print("Warning: imp argument is not CLIMADA Impact class")
            try:
                ev_id_obs = ev_ids_observed[i]
                if cmap_log:
                    norm1 = colors.LogNorm(vmin=vmin_log, vmax=vmax_)
                else:
                    norm1 = colors.Normalize(vmin=1, vmax=vmax_)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    act_dmgs.plot_hexbin_impact_exposure(event_id=ev_id_obs,gridsize=50,
                                                        axis=axes[row+i,2],linewidths=0.05,
                                                        norm=norm1,extent=extent,cmap=CMAP_IMPACT_CC,
                                                        shapes=shapes,**kwargs)
                axes[row+i,2].set_title(f'Actual dmg:{act_dmgs.at_event[act_dmgs.event_id==ev_id_obs][0]:.1e} {act_dmgs.unit}')
                #TEST EQUAL AXIS
                cax2 = fig.get_children()[-1]
                vmax2= cax2.dataLim.get_points()[1,1]
            except:
                axes[row+i,2].annotate('Actual damages are zero\nor unknown',
                xy=(0.1, 0.5), xycoords='axes fraction')
                vmax2=0

            if  (vmax1!=0 and vmax2!=0 and min(vmax1,vmax2)/max(vmax1,vmax2)<0.1
                 and not cmap_log): #more than 1 OOM difference and not log scale
                vmax_ = max(vmax1,vmax2)/3
            else:
                vmax_ = max(vmax1,vmax2)

    if extent:
        for ax in axes.flatten():
            ax.set_extent(extent)
    if draw_grid:
        for ax in axes.flatten():
            gl=ax.gridlines(draw_labels=True)
            gl.xlabels_top = gl.ylabels_right = False
            gl.xlabel_style = gl.xlabel_style = {'color':'grey'}
    if canton:
        for ax in axes.flatten():
            plot_canton(ax,canton=canton)
    return fig,ax


#define function to convert numpy array data to xr format
def npz_t_xr(date,ncfile,var):
    str_date2 = date.strftime('%Y%m%d')
    npz = 'C:/Users/timo_/Documents/PhD/data/radar_dBZ/CZC_6_6_%s.npy'%str_date2
    arr = np.flip(np.load(npz),axis=[0])
    # arr = np.load(npz)
    ds = xr.Dataset({var: (("chy","chx"),arr)},
                    coords = ncfile.coords).drop('time')
    return ds

#Plotting function for model skill plot
def scatter_from_imp_df(imp_df,unit,xmin,dmg_thresh,eval_dict):
    """plotting function for skill plot from hail_main.py

    Args:
        imp_df (pd.DataFrame): dataframe with modelled and observed damages
        unit (str): unit of impacts
        xmin (float): minimum displayed values
        dmg_thresh (float): threshold used for the skill metrics calculation
        eval_dict (dict): Dictionary of evaluation metrics and hazard,
        exposure and impact info

    Returns:
        plt.Figure: Pyplot figure
    """

    #Set up figure
    fig,axes = plt.subplots(2,2,figsize = (7,7),
                            gridspec_kw={'height_ratios':[5,1],
                                         'width_ratios':[1,5]})
    ax = axes[0,1]
    ax_0model = axes[1,1]
    ax_0model.sharex(ax)
    ax_0rep = axes[0,0]
    ax_0rep.sharey(ax)
    fig.delaxes(axes[1,0])

    #Scatter plot wit label and annotations
    if 'prePost2012' in imp_df.columns:
        col = imp_df.prePost2012
        labels = ['after 2012'] if sum(~imp_df.prePost2012)==0 else ['until 2012','after 2012']
    else:
        col = np.zeros(len(imp_df))
        labels= [None]
    scat=ax.scatter(imp_df.imp_obs,imp_df.imp_modelled,marker='o',
                    c = col,facecolors = 'none', edgecolors = 'black')

    l1=ax.legend(scat.legend_elements()[0],labels,loc='lower right',bbox_to_anchor=(1,0.17))
    ax.add_artist(l1)
    neither_zero = (imp_df.imp_modelled>0) & (imp_df.imp_obs>0)
    ax.annotate('n=%d\ndmg=%.1e%s'%(sum(neither_zero),
                                    imp_df.imp_modelled[neither_zero].sum(),unit),
                xy=(0.05, 0.9), xycoords='axes fraction',color='darkblue')

    #Plot events where the modelled damage is zero
    ax_0model.scatter(imp_df.imp_obs[imp_df.imp_modelled==0],
                      imp_df.imp_modelled[imp_df.imp_modelled==0],marker = 'o',
                      facecolors = 'none',edgecolors = 'black',
                      c = col[imp_df.imp_modelled==0])
    tot_0model = sum(imp_df.imp_obs[imp_df.imp_modelled==0])
    str_0model = 'n=%d\ndmg=%.1e%s\n(%.2f%%)'%(sum((imp_df.imp_modelled==0)),
                                               tot_0model,unit,
                                               tot_0model/sum(imp_df.imp_obs))
    ax_0model.annotate(str_0model, xy=(0.68, 0.35), xycoords='axes fraction',
                       color='darkblue')

    #plot events where the reported damage is zero (false alarms)
    ax_0rep.scatter(imp_df.imp_obs[imp_df.imp_obs==0],
                    imp_df.imp_modelled[imp_df.imp_obs==0],marker = 'o',
                    facecolors = 'none',edgecolors = 'black',
                    c = col[imp_df.imp_obs==0])
    tot_0rep = sum(imp_df.imp_modelled[imp_df.imp_obs==0])
    str_0rep = 'n=%d\ndmg=%.1e%s\n(%.2f%%)'%(sum((imp_df.imp_obs==0)),tot_0rep,
                                             unit,tot_0rep/sum(imp_df.imp_modelled))
    ax_0rep.annotate(str_0rep, xy=(0.01, 0.85), xycoords='axes fraction',color='darkblue')

    ylim = [min(imp_df.imp_modelled[imp_df.imp_modelled!=0])*0.5,
            max(imp_df.imp_modelled)*2]
    ax.set(yscale = 'log',xscale = 'log',ylim = ylim)
    ax_0model.set(xscale = 'log',xlabel = 'reported dmgs [%s]'%unit,
                  xlim=[xmin,imp_df.imp_obs.max()*2])
    ax_0rep.set(yscale = 'log',ylabel = 'modeled dmgs [%s]'%unit,
                ylim=[xmin,imp_df.imp_obs.max()*2])
    for adj,col,lbl in zip([-1,0,1],['salmon','red','salmon'],
                           [None,'1:1 line','+- one order of\nmagnitude']):
        ax.plot([xmin,1e9],[10**(np.log10(xmin)+adj),10**(9+adj)],color = col,label = lbl)

    ax.legend(loc = 'lower right')
    if dmg_thresh:
        ax.add_patch(Rectangle((0, 0), dmg_thresh, dmg_thresh, color="grey", alpha=0.5))
        ax_0model.add_patch(Rectangle((0, ax_0model.get_ylim()[0]), dmg_thresh,
                                      dmg_thresh, color="grey", alpha=0.5))
        ax_0rep.add_patch(Rectangle((ax_0rep.get_xlim()[0], 0), dmg_thresh,
                                    dmg_thresh, color="grey", alpha=0.5))

    fig.suptitle(f'Hazard: {eval_dict["haz_var"]}, Exposure: {eval_dict["exposure"]}, Impf: {eval_dict["impf"]}\n\
                RMSE: {eval_dict["rmse"]:.2e},  RMSF: {eval_dict["rmsf"]:.2f},  weigthed RMSF: {eval_dict["rmsf_weighted"]:.2f}\n\
                POD: {eval_dict["POD"]*100:.1f}%, FAR: {eval_dict["FAR"]*100:.1f}%, within 1 OOM: {eval_dict["p_within_OOM"]*100:.1f}%')
    return fig


def plot_pre_process(exp_dmg,date,poh,haz_poh,ds_PH,event_index,plot_dBZ_dates,
                     poh_level,extent=None):

    #definitions
    poh_cols =['khaki','wheat','darkred','plum','lightsteelblue']
    lines = [mlines.Line2D([], [], color=color) for color in poh_cols]
    labels = ['D-2','D-1','D0','D+1','D+2']
    lines_pt = [mlines.Line2D([], [], color=color, marker=marker, linewidth=0)
                for color,marker in zip(['black','blue','red'],['.','.','*'])]
    labels_pt = ['keep','move','delete']

    #initialize gdf
    if type(exp_dmg) == geopandas.geodataframe.GeoDataFrame:
        exp_gdf = exp_dmg
    else:
        exp_gdf = exp_dmg.gdf
    # if not type(exp_dmg) == climada.entity.exposures.base.Exposures:



    #initialize figure
    fig_crs = ccrs.epsg(2056)
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':fig_crs},figsize = (10,6))

    #plot countours
    for i,day_adj in enumerate([-2,-1,0,1,2]):
        #check that the day exists in the hazard data
        if (0 <= event_index+day_adj < haz_poh.size and
            haz_poh.date[event_index]+ day_adj == haz_poh.date[event_index+day_adj]):

            ax.contour(poh.lon,poh.lat,poh.sel(time=date+pd.Timedelta(day_adj,'d')).fillna(0),
                       transform = ccrs.PlateCarree(),levels = [poh_level],colors = poh_cols[i])
            ax.contour(poh.lon,poh.lat,ds_PH.possible_hail.sel(time=date+pd.Timedelta(day_adj,'d')),
                       transform = ccrs.PlateCarree(),levels = [0.5],
                       colors = poh_cols[i],linestyles = '--',alpha = 0.8)

    shift_str = 'shifted' if 'shifted' in exp_gdf['category'].unique() else 'shif'
    color = exp_gdf['category'].map({'none':'red',shift_str:'blue','day0':'black'})
    size = exp_gdf['category'].map({'none':10, shift_str:1, 'day0':1})
    exp_gdf.to_crs(fig_crs).plot(ax=ax,color = color,#cmap='viridis',cax=cax,legend=True,
                      markersize = size,zorder=10,marker='*')

    if date.strftime('%d%m%Y') in plot_dBZ_dates:
        try:
            #contour_npz(date,poh,ax)
            ds=npz_t_xr(date,poh,"dBZ")
            ax.contourf(ds.lon,ds.lat,ds.dBZ.fillna(0),transform = ccrs.PlateCarree(),
                        levels = [35,40,45,100],colors = ['lightsalmon','red','darkred','none'],alpha=0.2)
        except:
            ValueError("cannot plot dBZ contours, check (hardcoded) path for files")

    if extent: ax.set_extent(extent,crs=fig_crs)
    ax.set_title('%s, #claims: %d, within POH%d: %d%%'%
                  (dt.datetime.strftime(date, "%Y-%m-%d"),exp_gdf.shape[0],poh_level,
                  sum(exp_gdf.POH>=poh_level)/exp_gdf.shape[0]*100))
    plot_canton(ax,canton=['Zürich','Bern','Luzern','Aargau'])
    fig.legend(lines,labels,ncol=2,bbox_to_anchor=(0.5,0.1),title = 'POH %d%%'%poh_level)
    fig.legend(lines_pt,labels_pt,ncol=1,bbox_to_anchor=(0.7,0.1),title ='Damage reports')
    return fig

def plot_haz_hist(haz,bins=30,density=False,**kwargs):

    fig,ax=plt.subplots(1,1)
    ax.hist((haz.intensity>0).sum(axis=1).getA1(),bins=bins,density=density,**kwargs)
    ax.set(yscale='log',xlabel='Area [km$^2$]',ylabel='count')
    if density:
        ax.set_ylabel('density')

    return fig


def plot_skill_plot(ds_KGV_per_event,PAA_lim,MDR_lim,PAA_thresh=1e2,
                    MDR_thresh=1e5,axis0lim=1e-2,imp_mod_names=None,case_studies=None,labels=None,
                    color2='tab:blue',gs_fig=None,ret_ax=False,exp_str='',
                    unique_color=None,legend_loc=1,year_colors=None,**kwargs):
    #by default the plot is not existing yet
    plot_exists = False

    if year_colors is None:
        year_colors = [2013]
    else:
        #assert that year_colors are increasing integers
        assert(year_colors == sorted(year_colors))

    # map years to color categories
    def get_year_category(year,year_colors=year_colors):

        for count,year_color in enumerate(year_colors):
            if year < year_color:
                return count
        return len(year_colors)+1


    mapped_years = [get_year_category(year) for year in ds_KGV_per_event.date.dt.year.values]
    ds_KGV_per_event['year_category'] = xr.DataArray(mapped_years, dims='date',
                                                     coords={'date': ds_KGV_per_event.date})


    #create labels
    colors_years_w_startEnd = ([ds_KGV_per_event.date.dt.year.values.min()]+
                               year_colors+
                               [ds_KGV_per_event.date.dt.year.values.max()+1]) #+1 to include the last year
    year_col_lbl = [str(colors_years_w_startEnd[i])+'-'+str(colors_years_w_startEnd[i+1]-1) for i in range(len(year_colors)+1)]

    n_unique_colors = len(np.unique(ds_KGV_per_event['year_category']))
    if not n_unique_colors == len(year_col_lbl) == len(year_colors)+1:
        raise ValueError('Number of unique colors does not match number of valid year categories')

    #define cmap
    if unique_color is None:
        if isinstance(color2,str):
            colors = ["beige",color2]
        else:
            colors = color2
        # cmap_post2012 = ListedColormap(["beige"]+color2)
    else:
        colors = [unique_color for i in range(len(year_colors)+1)]
        # cmap_post2012 = ListedColormap([unique_color, unique_color])
    cmap_scatter = ListedColormap(colors)

    # ds_KGV_per_event = ds_KGV_per_event.assign(mapped_years=mapped_years)

    ds_KGV_per_event['post2012']=ds_KGV_per_event.date.dt.year>=2013

    #initialize figure
    if gs_fig is None:
        fig,axes = plt.subplots(2,5,figsize=(10,4),
                                gridspec_kw={'height_ratios':[1,0.15],
                                             'width_ratios':[0.15,1,0.3,0.15,1],
                                             'wspace':0.00,'hspace':0.00})
    elif isinstance(gs_fig[0],gridspec.SubplotSpec):
        gs_in, fig = gs_fig
        gs = gridspec.GridSpecFromSubplotSpec(2,5, subplot_spec=gs_in,wspace=0.0,
                                              hspace=0,width_ratios=[0.15,1,0.3,0.15,1],
                                              height_ratios=[1,0.15])
        axes=np.array([[0,0,0,0,0],[0,0,0,0,0]],dtype=object)
        for x in range(5):
            for y in range(2):
                axes[y,x]= fig.add_subplot(gs[y,x])
    elif isinstance(gs_fig[0],np.ndarray):
        axes, fig = gs_fig
        plot_exists = True

    for ax in axes[:,2]:
        ax.set_visible(False)

    axes1 = axes[:,:2]
    axes2 = axes[:,3:]
    ax1 = axes[0,1]
    ax2 = axes[0,4]
    ax3 = axes[1,1]


    if imp_mod_names is None:
        imp_mod_names = ['n_buildings_MESHS','imp_MESHS']

    for x,y,xlimFull,ylimFull,axes_sel,dmg_thresh in zip(['n_count','imp_observed'],
                                imp_mod_names, # default:['n_buildings_MESHS','imp_MESHS'],
                                [PAA_lim,MDR_lim,],
                                [PAA_lim,MDR_lim],
                                (axes1,axes2),
                                [PAA_thresh,MDR_thresh]):
        [[a,b],[c,d]] = axes_sel
        for ax,ylim,yscale,xlim,xscale in zip([a,b,c,d],
                            [ylimFull,ylimFull,(-axis0lim,axis0lim),(-axis0lim,axis0lim)],
                            ['log','log','linear','linear'],
                            [(-axis0lim,axis0lim),xlimFull,(-axis0lim,axis0lim),xlimFull],
                            ['linear','log','linear','log']):
            if ax == c:
                ax.set(xlim=xlim,ylim=ylim) #do not plot the 0,0 axis, as days with neither reported nor observed damage are not included
            else:
                add_legend = None# True if ax == b else False #only works with fillna(0)
                dsNow = ds_KGV_per_event.where((ds_KGV_per_event[x]>xlim[0])&(ds_KGV_per_event[x]<xlim[1]))
                dsNow = dsNow.where((ds_KGV_per_event[y]>ylim[0])&(ds_KGV_per_event[y]<ylim[1]))



                artist=dsNow.plot.scatter(x=x,y=y,hue='year_category',hue_style='discrete',
                                          add_colorbar=False,cmap=cmap_scatter,edgecolor='black',
                                          alpha=1,xscale=xscale,yscale=yscale,ylim=ylim,
                                          xlim=xlim,ax=ax,add_title=False,add_legend=add_legend,**kwargs)
        if case_studies is not None:
            #add case studies
            ds_KGV_per_event.sel(date=case_studies).plot.scatter(x=x,y=y,ax=b,marker='x',
                                                                 color='red',s=50,zorder=10,
                                                                 add_title=False)

        #Format the cut axes correctly
        sc.E.make_cut_axis(a,b,c,d)
        if not plot_exists:
            #grey out dmg_tresh
            sc.E.grey_out_dmg_thresh(a,b,c,d,dmg_thresh=dmg_thresh)
            #add skill background colors
            sc.E.skill_background_cols(a,b,c,d,min_plot_val=xlimFull[0],
                                    max_plot_val=xlimFull[1],
                                    dmg_thresh=dmg_thresh,alpha=0.4)
            transFigure = fig.transFigure.inverted()
            # for startEnd in [(-0.1,1e9),]
            # for factor,col in zip([0.1,1,10],['grey','black','grey']):
            coord1 = transFigure.transform(b.transData.transform([xlimFull[1],xlimFull[1]]))
            coord2 = transFigure.transform(c.transData.transform([-axis0lim,-axis0lim]))
            line1 = Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                        transform=fig.transFigure,
                                        ls='--',
                                        color='black')
            fig.lines.extend([line1])

    for factor,col in zip([0.1,10],['grey','grey']):
        ax1.axline(xy1=(1,1*factor),xy2=(10,10*factor), color=col,ls='--')
        ax2.axline(xy1=(1,1*factor),xy2=(10,10*factor), color=col,ls='--')

    if labels is not None:
        assert(len(labels)==2)
        for ax,label in zip([axes1[0,0],axes2[0,0]],labels):
            txt = ax.text(0.15,0.97,label,transform=ax.transAxes,fontsize=11,
                          fontweight='bold',ha='left',va='top')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    if not plot_exists:
        #create legend
        if n_unique_colors >1 and unique_color is None: #only create if multiple colors are used
            handles = [Line2D([],[],color='none',markerfacecolor=colors[i],
                            markeredgecolor='black',marker='o', label=year_col_lbl[i])
                            for i in range(n_unique_colors)]
            #invert handles so most recent data are in the first line
            handles = handles[::-1]
            if case_studies is not None:
                handles.append(Line2D([],[],color='none',markerfacecolor='red',
                                    markeredgecolor='red',marker='x', label='Case studies'))
            ax_leg = axes[0,1] if legend_loc==1 else axes[0,4]
            legend = ax_leg.legend(handles=handles,loc='upper left')#bbox_to_anchor=(0.5, 0),ncol=1)


    #set labels
    axes[0,0].set_ylabel(f'Modelled No. of {exp_str} damages')
    axes[1,1].set_xlabel(f'Reported No. of {exp_str} damages')
    axes[0,3].set_ylabel(f'Modelled {exp_str} damage [CHF]')
    axes[1,4].set_xlabel(f'Reported {exp_str} damage [CHF]')

    if ret_ax:
        return fig, axes
    else:
        return fig

#impact function plots (from empirical calibration)
def fill_quantile(ds, var, q, col, ax,haz_var, binned=False, dim='b_sample',
                  alpha=0.5,cut_off=None,label=None):
    if cut_off is not None:
        ds=ds.where(ds[haz_var]<=cut_off)
    if label is None:
        label = '%s, Q %.2f-%.2f' % (var, q[0], q[1])

    ax.fill_between(ds[haz_var][1:], ds.quantile(q[0], dim=dim)[var][1:],
                    ds.quantile(q[1], dim=dim)[var][1:],  # step='pre',
                    color=col, alpha=alpha, label=label)

def fill_df_quantile(df,q1,q2,ax,label=None,**kwargs):
    q1_arr=df.quantile(q=q1,axis=1,numeric_only=True)*100
    q2_arr=df.quantile(q=q2,axis=1,numeric_only=True)*100
    if label is None:
        label = 'Q %.2f-%.2f' % (q1, q2)
    color = kwargs.pop('color','black')
    alpha = kwargs.pop('alpha',0.2)
    ax.fill_between(df.index,q1_arr,q2_arr,color=color,alpha=alpha,
                    label = label,**kwargs)

def impf_plot(df_all,df_roll,df_roll_cut,ds_boot_roll,ds_boot_roll_cut,haz_var,
              impf_emanuel,cut_off,plot_var='PAA',title='',dmg_bin_size=5,
              intensity_label = 'Intensity [?]',color='green'):

    intensity_range = df_roll.index.values
    xlim = [min(intensity_range)-2, max(intensity_range)]

    # plot rolling PAA
    if df_roll.index.values.max() > cut_off:
        paa_ylim = (0,max(df_roll[plot_var][np.isfinite(df_roll[plot_var])].max(),
                        df_roll.loc[cut_off,plot_var]*1.5)*100)
    else:
        paa_ylim = (0,df_roll[plot_var][np.isfinite(df_roll[plot_var])].max()*100*1.03)

    fig, ax = plt.subplots()
    ax.plot(df_roll.index[1:], df_roll[plot_var].iloc[1:]*100, color=color, label=plot_var)
    if intensity_range[0] == 0:
        ax.scatter(df_roll.index[0], df_roll[plot_var][0]*100, color=color, alpha=0.5, marker='.')
    fill_quantile(ds_boot_roll*100, plot_var, (0.05, 0.95), f'light{color}', ax,haz_var)

    sc.E.plot_monotone_fit(df_roll_cut*100, plot_var, ax, ls='--')
    sc.E.plot_monotone_fit(ds_boot_roll_cut.quantile(0.95, dim='b_sample').to_dataframe()*100,
                           plot_var, ax, ls='--', color='grey', label='Q 0.05-0.95')
    sc.E.plot_monotone_fit(ds_boot_roll_cut.quantile(0.05, dim='b_sample').to_dataframe()*100,
                           plot_var, ax, ls='--', color='grey', label=None)
    ax.scatter(impf_emanuel.intensity,impf_emanuel.mdd*100,marker='o',
               label='Sigmoidal fit',edgecolor='black',facecolor='none',s=10)
    sc.E.plot_dmg_bin(df_all, ax, color='red', alpha=0.1,bin_size=dmg_bin_size)

    fig.legend(loc='upper left', bbox_to_anchor=ax.get_position())
    ax.set_title(title)
    ax.set(ylabel=f'{plot_var} [%]', xlim=xlim,ylim=paa_ylim, xlabel=intensity_label)
    ax.add_patch(Rectangle((cut_off, ax.get_ylim()[0]), cut_off*3,
                           ax.get_ylim()[1]-ax.get_ylim()[0],
                           color="grey", alpha=0.5))
    return fig,ax

def impf_plot2(df_all,df_roll,df_roll_cut,ds_boot_roll,ds_boot_roll_cut,haz_var,
               impf_emanuel,cut_off,plot_var='PAA',title='',dmg_bin_size=5,
               intensity_label = 'Intensity [?]',color='green',df_boot_emanuel=None,
               quantile = (0.05,0.95),relative_bins=False,figAx=None):

    intensity_range = df_roll.index.values
    xlim = [min(intensity_range)-2, max(intensity_range)]
    # plot rolling PAA
    if ds_boot_roll[haz_var].max().item() > cut_off:
        paa_ylim = (0,max(df_roll[plot_var][np.isfinite(df_roll[plot_var])].max(),
                        ds_boot_roll[plot_var].sel({haz_var:cut_off}).quantile(quantile[1]),#df_roll.loc[cut_off,plot_var]*1.5,
                        impf_emanuel.mdd.max())*100)
        iloc_cut_off = np.where(df_roll.index.values==cut_off)[0][0]
    else:
        paa_ylim = (0,df_roll[plot_var][np.isfinite(df_roll[plot_var])].max()*100*1.03)
        iloc_cut_off = len(df_roll.index)-1

    if figAx is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figAx
    # density plot of bootstrap samples

    ax.plot(df_roll.index[1:iloc_cut_off+1],
            df_roll[plot_var].iloc[1:iloc_cut_off+1]*100,
            color=color, label=plot_var)
    fill_quantile(ds_boot_roll*100, plot_var, quantile, f'light{color}', ax,
                  haz_var,alpha=0.8,cut_off=cut_off,label='bootstrap CI (5-95%)')

    ax.plot(impf_emanuel.intensity,impf_emanuel.mdd*100,label='Sigmoidal fit',color='black')
    if df_boot_emanuel is not None:
        fill_df_quantile(df_boot_emanuel,0.05,0.95,ax,label='bootstrap CI (5-95%)')
    ax2=sc.E.plot_dmg_bin(df_all, ax, color='tomato', alpha=0.2,
                          bin_size=dmg_bin_size,relative=relative_bins)

    #plot exposed assets
    # sc.E.plot_dmg_bin(df_all, ax, color='blue', alpha=0.1,new_axis=False,bin_size=dmg_bin_size,pl_var='count_all')
    # ax.set(xlim=xlim, xlabel=intensity_label)
    # ax.set(ylabel=f'Proportion of affected assets [%]', xlim=xlim,ylim=paa_ylim, xlabel=intensity_label)
    hdl,lbl = ax.get_legend_handles_labels()
    hdl2,lbl2 = ax2.get_legend_handles_labels()
    fig.legend(hdl+hdl2,lbl+lbl2,loc='upper left', bbox_to_anchor=ax.get_position(),
               framealpha=0.5)
    ax.set_title(title)
    ax.set(ylabel=f'{plot_var} [%]', xlim=xlim,ylim=paa_ylim, xlabel=intensity_label)
    ax.add_patch(Rectangle((cut_off, ax.get_ylim()[0]), cut_off*3,
                           ax.get_ylim()[1]-ax.get_ylim()[0], color="grey",
                           alpha=0.7))
    return fig,ax


def plot_impf_pixel(ds_roll_gc,df_roll,ds_boot_roll,haz_var,cut_off,q1,q2,
                    impf_emanuel=None,plot_var='PAA',title='',color='green',
                    intensity_label='Intensity [?]',pl_type=1):
    ds1 = get_emp_quantile(ds_roll_gc, q1)
    ds2 = get_emp_quantile(ds_roll_gc, q2)
    fig, ax = plt.subplots()

    #Type 1 plot
    if pl_type == 1:
        ax.plot(df_roll.index[1:], df_roll[plot_var][1:]*100, color=color,
                label=plot_var)

        ax.fill_between(ds1[haz_var], ds1[plot_var]*100, ds2[plot_var]* 100,
                        color='orange', alpha=0.3,
                        label='pixel-wise Q%d-Q%d' % (q1*100, q2*100))
        fill_quantile(ds_boot_roll*100, plot_var, (0.05, 0.95), f'light{color}',
                      ax,haz_var,alpha=0.9)

        weighted_PAA = ds_roll_gc[['PAA', 'MDR']].weighted(weights=ds_roll_gc.exp_val.fillna(0)).mean(dim='index')
        ax.plot(df_roll.index[1:], weighted_PAA[plot_var]*100, color='black', label=plot_var)


    #Type 2 plot
    elif pl_type == 2:
        ax.plot(impf_emanuel.intensity,impf_emanuel.mdd*100,
                label='Sigmoidal fit',color='black')

        dsQ0 = get_emp_quantile(ds_roll_gc, 0)
        for quantile in np.flip([0.8,0.85,0.9,0.95]):
            color = (1-(quantile-0.7)*3,0,(quantile-0.7)*3)
            dsQ = get_emp_quantile(ds_roll_gc, quantile)
            ax.fill_between(dsQ0[haz_var], dsQ0[plot_var]*100, dsQ[plot_var]
                            * 100,color=color, alpha=1, label='pixel-wise Q%d' % (quantile*100))

        weighted_PAA = ds_roll_gc[['PAA', 'MDR']].weighted(weights=ds_roll_gc.exp_val.fillna(0)).mean(dim='index')

    #Type 3 plot
    elif pl_type == 3:
        dsMean = get_emp_mean(ds_roll_gc)
        ax.plot(impf_emanuel.intensity,impf_emanuel.mdd*100,
                label='Sigmoidal fit',color='black')
        ax.plot(df_roll.index[1:], df_roll[plot_var][1:]*100, color=color,
                label=plot_var)
        ax.fill_between(ds1[haz_var], dsMean[plot_var]*100, dsMean[plot_var]*100,
                        color=color, alpha=1, label='pixel-wise mean')

        ax.fill_between(ds1[haz_var], dsMean[plot_var]*100/2, dsMean[plot_var]*100*2,
                        color=color, alpha=0.5, label='pixel-wise mean')

        weighted_PAA = ds_roll_gc[['PAA', 'MDR']].weighted(weights=ds_roll_gc.exp_val.fillna(0)).mean(dim='index')
        ax.plot(df_roll.index[1:], weighted_PAA[plot_var]*100, color='black', label=plot_var)


    ax.legend()
    ax.add_patch(Rectangle((cut_off, 0), cut_off*3, ax.get_ylim()[1], color="grey", alpha=0.5))
    ax.set(title=title,ylabel=plot_var,ylim=[0,min(100,ax.get_ylim()[1])],xlabel=intensity_label)
    return fig,ax

def get_emp_quantile(ds_roll_gc, q):
    ds_roll_gc['PAA'] = ds_roll_gc.n_dmgs/ds_roll_gc.n_exp
    ds_roll_gc['MDR'] = ds_roll_gc.dmg_val/ds_roll_gc.exp_val
    ds = ds_roll_gc[['PAA', 'MDR']]
    quant = ds.quantile(q, dim='index')
    return quant

def get_emp_mean(ds_roll_gc):
    ds_roll_gc['PAA'] = ds_roll_gc.n_dmgs/ds_roll_gc.n_exp
    ds_roll_gc['MDR'] = ds_roll_gc.dmg_val/ds_roll_gc.exp_val
    ds = ds_roll_gc[['PAA', 'MDR','exp_val']]
    mean = ds.weighted(weights=ds_roll_gc.exp_val.fillna(0)).mean(dim='index')
    return mean

########################### Subproject D #####################################
def plot_POH(date, hazard, axis):
    """Plot POH
    Parameters
    ----------
    date : str or pandas.Timestamp
        datestring of event in YYYY-MM-DD or pandas.Timestamp
    haz : Hazard
        CLIMADA hazard (POH)
    axis : axis to plot
       matplotlib.axis

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot

    """
    # make colormap
    cmap = colors.ListedColormap(['r', 'g', 'y'])
    cmap.set_under('w')
    # bounds = [40,60,80,100]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    if type(date) == pd.Timestamp:
        datestring=date.strftime('%Y-%m-%d')
    elif type(date) == type(""):
        datestring=date
    else:
        raise ValueError(f"data type of {date} not compatible.")
    # create event_name from datestring
    event_name = 'ev_' + datestring

    # plot data
    subplot = hazard.plot_intensity(event=event_name, axis=axis, cmap=cmap,
                                    vmin=40, vmax=100, alpha=1)

    return subplot


def plot_MESHS(date, hazard, axis,cmap):
    """Plot MESHS
    Parameters
    ----------
    date : str or pandas.Timestamp
        datestring of event in YYYY-MM-DD
    haz : Hazard
        CLIMADA hazard (MESHS)
    axis : axis to plot
       matplotlib.axis


    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot

    """
    # make colormap
    if not cmap:
        cmap = colors.ListedColormap(['r', 'g', 'y'])
    cmap.set_under('w')
    cmap.set_over('pink')
    # bounds = [10,20,30,40]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    if type(date) == pd.Timestamp:
        datestring=date.strftime('%Y-%m-%d')
    elif type(date) == type(""):
        datestring=date
    else:
        raise ValueError(f"data type of {date} not compatible.")

    # create event_name vom datestring
    event_name = 'ev_' + datestring

    # plot data
    subplot = hazard.plot_intensity(event=event_name, axis=axis, cmap=cmap, vmin=20, vmax=80, alpha=1)

    # to do: extend colorbar with colors below and above the range
    return subplot


def plot_CH_map(ncols=2,nrows=2, proj=ccrs.epsg(2056),stamen_map=True,
                figsize=(15,10),lakes=True,cantons=False,pads=0.1,lakesalpha=1,lakeszorder=0.5,edgecolor='black'):
    """Plot a Figure of a map of Switzerland optionally using the Stamen terrain background map (http://maps.stamen.com/#toner/12/37.7706/-122.3782)
    Parameters
    ----------

    stamen_map: boolean
            if True, plot Stamen terrain map, otherwise no background (default is True)
    figsize: tuple
            the figsize keyword argument for matplotlib.figure

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot

    """

    ch_shp_path = str(CONFIG.ch_shp_path)

    def rect_from_bound(xmin, xmax, ymin, ymax):
        """Returns list of (x,y)'s for a rectangle"""
        xs = [xmax, xmin, xmin, xmax, xmax]
        ys = [ymax, ymax, ymin, ymin, ymax]
        return [(x, y) for x, y in zip(xs, ys)]

    # request data for use by geopandas
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'

    shpfilename = shpreader.natural_earth(resolution, category, name)

    df = geopandas.read_file(shpfilename)

    # get geometry of a country
    poly = [df.loc[df['ADMIN'] == 'Switzerland']['geometry'].values[0]]

    stamen_terrain = cimgt.Stamen('terrain-background')

    # projections that involved
    if stamen_map == True:
        st_proj = stamen_terrain.crs  #projection used by Stamen images
        ll_proj = ccrs.PlateCarree()
    else:
        ll_proj = ccrs.AlbersEqualArea(central_longitude=8) #proj #ccrs.PlateCarree() #ccrs.AlbersEqualArea(central_longitude=8) #ccrs.PlateCarree() #  #CRS for raw long/lat
        st_proj = ll_proj

    if isinstance(pads,list):
        if len(pads)==4:
            exts = [poly[0].bounds[0] - pads[0], poly[0].bounds[2] + pads[1],
                    poly[0].bounds[1] - pads[2], poly[0].bounds[3] + pads[3]];
        else:
            raise ValueError('Argument pads needs to be sequence of length 4 or float.')
    else:
        pad1=pads
        exts = [poly[0].bounds[0] - pad1, poly[0].bounds[2] + pad1,
                poly[0].bounds[1] - pad1, poly[0].bounds[3] + pad1];

    # create fig and axes using intended projection
    fig = plt.figure(figsize = figsize)

    axes=[]

    subplot=0

    for row in range(nrows):
        for col in range(ncols):
            subplot+=1
            ax = fig.add_subplot(nrows,ncols,subplot, projection=st_proj) #st_proj)
            ax.add_geometries(poly, crs=ll_proj, facecolor='none', edgecolor=edgecolor)

            #pad1 = .1  #padding, degrees unit
            #exts = [poly[0].bounds[0] - pad1, poly[0].bounds[2] + pad1, poly[0].bounds[1] - pad1, poly[0].bounds[3] + pad1];
            ax.set_extent(exts, crs=proj)
            ax.spines['geo'].set(edgecolor=edgecolor)

            # make a mask polygon by polygon's difference operation
            # base polygon is a rectangle, another polygon is simplified switzerland
            msk = Polygon(rect_from_bound(*exts)).difference( poly[0].simplify(0.01) )
            msk_stm  = st_proj.project_geometry (msk, ll_proj)  # project geometry to the projection used by stamen

            if stamen_map == True:
                # get and plot Stamen images
                ax.add_image(stamen_terrain, 8) # this requests image, and plot

                # plot the mask using semi-transparency (alpha=0.65) on the masked-out portion
                ax.add_geometries( msk_stm, st_proj, zorder=12, facecolor='white', edgecolor='none', alpha=0.65)

            #ax.gridlines(draw_labels = False)
            #ax.add_feature(cf.LAKES)
            ax.add_feature(cf.BORDERS)

            if lakes == True:
                reader2 = shpreader.Reader("%s/Hydrography/swissTLMRegio_Lake.shp"%ch_shp_path)
                geometry = reader2.geometries()
                geometry = np.array([g for g in geometry])
                lakesize = np.array([a.area for a in reader2.geometries()])
                geometry = geometry[lakesize>2e7]
                shape_feature2 = cf.ShapelyFeature(geometry,
                                                   ccrs.epsg(2056), edgecolor='none',facecolor = "lightblue")
                ax.add_feature(shape_feature2,zorder=lakeszorder,alpha=lakesalpha)#0.5,2
            if cantons == True:
                reader = shpreader.Reader("%s/swissTLMRegio_KANTONSGEBIET_LV95.shp"%ch_shp_path)
                sel_cantons = [place for place in reader.records() if place.attributes["OBJEKTART"]=='Kanton' and place.attributes["ICC"]=='CH']
                for sel_canton in sel_cantons:
                    shape_feature = cf.ShapelyFeature([sel_canton.geometry],
                                                           ccrs.epsg(2056), edgecolor = 'dimgrey',facecolor = "none")
                    ax.add_feature(shape_feature,linewidth=0.5,zorder=0.5)

            axes.append(ax)


    return fig, axes,exts


def plot_modeled_vs_observed(impact,damages,event_name,variable):

     import scClim.plot_funcs as scplot
     from scClim.constants import  CH_EXTENT_EPSG2056


     fig,axes,extent=scplot.plot_CH_map(subplots=(1,2),stamen_map=False,figsize=(20,10)) #, extent=[lonmin-0.25,lonmax+0.25,latmin-0.1,latmax+0.1])
     plt.rcParams.update({'font.size': 15})
     print(event_name)

     # MODELED
     impday=impact.select(event_names=event_name)

     if impday.imp_mat.getnnz()>0:
         impday.plot_hexbin_impact_exposure(event_id=impday.event_id[0],extent=CH_EXTENT_EPSG2056,axis=axes[0],gridsize=80,vmin=1,vmax=25,extend='both',linewidths=0.0)
     cbax=fig.gca()
     if variable == 'PVA':
         cbax.set_ylabel('affected area (ha)')
     elif variable == 'PAA':
         cbax.set_ylabel('affected number of fields')

     total_impact=np.int(np.round(impday.imp_mat.sum(), decimals = 0))

     axes[0].set_title('modeled')
     #axes[0].gridlines(draw_labels=['left','bottom'])
     axes[0].set_extent(CH_EXTENT_EPSG2056, crs=ccrs.epsg(2056))
     axes[0].text(0.1,0.9,'total: '+ str(total_impact), transform=axes[0].transAxes)

     # REPORTED
     dmgday=damages.select(event_names=event_name)

     if dmgday.imp_mat.getnnz()>0:
         dmgday.plot_hexbin_impact_exposure(event_id=dmgday.event_id[0],extent=CH_EXTENT_EPSG2056,axis=axes[1],gridsize=80,vmin=1,vmax=25,extend='both',linewidths=0.0)
     cbax=fig.gca()
     if variable == 'PVA':
         cbax.set_ylabel('affected area (ha)')
     elif variable == 'PAA':
         cbax.set_ylabel('affected number of fields')

     total_impact=np.int(np.round(dmgday.imp_mat.sum(), decimals = 0))

     axes[1].set_title('reported')
     #axes[0].gridlines(draw_labels=['left','bottom'])
     axes[1].set_extent(CH_EXTENT_EPSG2056, crs=ccrs.epsg(2056))
     axes[1].text(0.1,0.9,'total: '+ str(total_impact), transform=axes[1].transAxes)

     return fig,axes

########################### Subproject A #####################################
def plot_sandbox_RADAR_IMPACTS(day,haz_ds,imp,var,varname,unit,img_dir,
                               asset='buildings',croptype=None):
    """
    Parameters
    ----------
    day : datetime.datetime
        day to be plotted
    haz_ds : xarray.Dataset
        hazard dataset containing variable *varname*
    imp : climada.Impact
        climada impact object to be plotted
    var : str
        variable name (full name that appears on the plot, e.g. MESHS)
    varname : str
        variable name in the hazard dataset (e.g. MZC)
    unit : str
        variable unit
    img_dir : str
        directory to store output figures
    asset : asset type, optional
        asset type to be plotted. Currently supports "bulidings" and "fields".
        The default is 'buildings'.
    croptype : str, optional
        name of the crop for which impact is plotted.
        Only relevant if asset is "fields". The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
         Figure with all content
    axes : matplotlib.axes.Axes
         Figure axes

    """

    #hardcoded variables (MESHS specific)
    vmin = 20
    vmax = min(haz_ds[varname].max().values,60)

    #format titles
    tomorrow=day+dt.timedelta(days=1)
    haz_name=r"$\bf{Maximum~Expected~Severe~Hail~Size~(MESHS)}$"
    date_text=f'VALID: {day.strftime("%d/%m/%y 06UTC")} - {tomorrow.strftime("%d/%m/%y 06UTC")}'

    #default cbar label and cmap and impact scale (log vs linear)
    cbar_label = f'Number of damaged {asset}'
    cmap = CMAP_IMPACT_CC

    if asset == 'fields':
        fig_dir = f'{img_dir}/agro/{croptype}_'
        imp_norm = Normalize(vmin=1, vmax=25)
        if croptype in ['wheat', 'farmland']:
            text=f'CLIMADA~IMPACT:~Number~of~damaged~{croptype}~fields'
            damage_name=r"$\bf{text}$".format(text=text)
        elif croptype == 'grapevine':
            text='CLIMADA~IMPACT:~Number~of~damaged~vineyards'
            damage_name=r"$\bf{text}$".format(text=text)
    elif asset == 'buildings':
        imp_norm = LogNorm(vmin=100, vmax=1e6)

        fig_dir = f'{img_dir}/buildings/'
        text='CLIMADA~IMPACT:~Estimated~damage~to~buildings'
        damage_name=r"$\bf{text}$".format(text=text)
        cbar_label = 'Building damage [CHF]'
        cmap = CMAP_IMPACT_CHF
    elif asset == 'n_buildings':
        imp_norm = Normalize(vmin=1, vmax=80)
        fig_dir = f'{img_dir}/buildings/PAA_'
        text='CLIMADA~IMPACT:~Number~of~damaged~buildings'
        damage_name=r"$\bf{text}$".format(text=text)
        cbar_label = 'Number of damaged buildings'
    elif asset == 'cars':
        imp_norm = LogNorm(vmin=100, vmax=1e6)
        fig_dir = f'{img_dir}/cars/'
        text='CLIMADA~IMPACT:~Estimated~damage~to~cars:'
        damage_name=r"$\bf{text}$".format(text=text)
        cbar_label = 'Car damage [CHF]'
        cmap = CMAP_IMPACT_CHF
    elif asset == 'n_cars':
        imp_norm = Normalize(vmin=1, vmax=80)
        fig_dir = f'{img_dir}/cars/PAA_'
        text='CLIMADA~IMPACT:~Number~of~damaged~cars:'
        damage_name=r"$\bf{text}$".format(text=text)
        cbar_label = 'Number of damaged cars'

    #set colormap lower color to transparent
    cmap.set_under('white',alpha=0)


    fig, axes = plt.subplots(1,2,subplot_kw={'projection':ccrs.epsg(2056)},
                            gridspec_kw={'wspace':0.15},
                            figsize=(12,7),constrained_layout=True)#,figsize=(6,8)


    # plot radar data
    ax=axes[0]
    sc.plot_nc(haz_ds[varname].where(haz_ds[varname]>0),ax=ax,vmin=vmin,vmax=vmax,borders=False,cbar_lbl='%s [%s]'%(var.upper(),unit),zorder=10,cmap=cc.cm.bgy)
    ax.set_title(haz_name+'\n'+date_text, loc='left')

    # plot impact data
    ax=axes[1]
    # ax.add_feature(cf.BORDERS)
    if imp.aai_agg>0:
        imp.plot_hexbin_impact_exposure(event_id=1,gridsize=70,axis=ax,linewidths=0,pop_name=False,
                                        adapt_fontsize=False,norm=imp_norm,cmap=cmap,shapes=False,
                                        extent=CH_EXTENT_EPSG2056,buffer=0)
        imp_tot = imp.at_event[0]
        fig.axes[3].set(ylabel=cbar_label)
    else:
        cbax = make_axes_locatable(ax).append_axes(
        'right', size="6.5%", pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(ScalarMappable(norm=None, cmap=cmap), cax=cbax, orientation='vertical')#, extend=extend)
        cbar.set_label(cbar_label)#cbar.set_label(f'Estimated damage = 0 {asset}')
        cbar.set_ticks([])
        imp_tot = 0

    ax.set_extent(CH_EXTENT_EPSG2056,crs=ccrs.epsg(2056))
    ax.set_title(damage_name+f' {imp_tot:.0f} {imp.unit}\n'+date_text, loc='left')

    #plot cantons
    for ax in axes:
        sc.plot_canton(ax,canton='all',edgecolor='lightgrey',lakes=True,ch_border='black')

    # set hspace
    fig.set_constrained_layout_pads(wspace=0.15)  # for matplolib version 3.3
    # fig.get_layout_engine().set(wspace=0.15) #version 3.7

    fig.savefig(f"{fig_dir}{day.strftime('%Y-%m-%d')}.jpg", dpi=300,bbox_inches='tight')
    print(f"Saved figure for {asset} for {day.strftime('%Y-%m-%d')}")

    return fig, axes

def plot_PAA(ds_KGV,ax,date_dt,min_exp_count=10,cbar=False,**pl_kwargs):

    paa = ds_KGV.sel(date=date_dt)['PAA']*100
    q99 = np.quantile(paa.values.flatten()[~np.isnan(paa.values.flatten())],0.99)
    if cbar:
        sc.plot_funcs.plot_nc(paa,ax=ax,cmap='Blues',vmax=q99,cbar_lbl='Affected Assets [%]',**pl_kwargs)
    else:
        sc.plot_funcs.plot_nc(paa,ax=ax,cmap='Blues',vmax=q99,cbar=False,**pl_kwargs)
    exp_count=ds_KGV.sel(date=date_dt)['n_count_exposure']; density=6
    #grey out areas without enough exposure
    plot_nc(exp_count.fillna(0)<min_exp_count,ax=ax,pl_type='bool_field',cmap='lightgrey',**pl_kwargs)
    ax.set(title=date_dt.strftime('%Y-%m-%d'))


def plot_dmg_haz(date_dt,ds_KGV,meshs,dBZ,pl_type='',min_exp_count=10,**pl_kwargs):

    fig,(ax0,ax1,ax2)=plt.subplots(1,3,subplot_kw={'projection':ccrs.epsg(2056)},figsize=(10,3),
                                  gridspec_kw={'wspace':0.32})
    #plot PAA
    plot_PAA(ds_KGV,ax0,date_dt,min_exp_count=min_exp_count,**pl_kwargs)

    #plot MEHSHS, dBZ
    plot_nc(meshs.MZC.sel(time=date_dt),ax=ax1,title=date_dt.strftime('%Y-%m-%d: '),cmap='viridis',vmax=100,cbar_lbl='MESHS [mm]',**pl_kwargs)
    plot_nc(dBZ.CZC.sel(time=date_dt),ax=ax2,title=date_dt.strftime('%Y-%m-%d: '),cmap='magma_r',vmin=45,vmax=70,cbar_lbl='Reflectivity [dBZ]',**pl_kwargs)
    if date_dt.year>=2013:
        ax2.set(title='Reflectivity')
    else:
        ax2.set(title='Reflectivity \n(w/o reliable clutter filter)')
    ax0.set(title='PAA')

    if pl_type == '_dBZ':
        levels = [55,60]; colors = ['salmon','darkred']
        dBZ['dBZ_smooth'] = (("chy","chx"),g_filter(dBZ.CZC.sel(time=date_dt),sigma=0.5))
        sc.plot_nc(dBZ.dBZ_smooth,ax=ax0,borders=False,pl_type='contour',levels=levels,colors=colors,**pl_kwargs)
        # ax0.annotate('dBZ=55 (lightred),dBZ=60 (red)', xy=(0, -0.05), xycoords='axes fraction',verticalalignment='top',horizontalalignment='left')
        lines = [mlines.Line2D([], [], color=color) for color in colors]
        fig.legend(lines,[f'{l} dBZ' for l in levels],bbox_to_anchor=(0.11,0.7),title='Reflectivity')

    if pl_type == '_meshs':
        levels = [20,40]; colors=['lightgreen','darkgreen']
        if date_dt.strftime('%Y-%m-%d')=='2021-06-28': levels = [20,60]
        sc.plot_funcs.plot_nc(meshs.sel(time=date_dt).MZC.fillna(0),ax=ax0,pl_type='contour',levels=levels,colors=colors,**pl_kwargs)
        lines = [mlines.Line2D([], [], color=color) for color in colors]
        fig.legend(lines,[f'{l} mm' for l in levels],bbox_to_anchor=(0.12,0.7),title='MESHS')
    for ax in [ax0,ax1,ax2]:
        sc.plot_canton(ax,canton=['Bern','Luzern','Zürich','Aargau'],edgecolor='black',lakes=True,zorder=2)

    return fig

# def plot_dmg_haz2(date_dt,ds_KGV,meshs,dBZ,E_kin,pl_type='',min_exp_count=10,**pl_kwargs):

    # fig,axes=plt.subplots(2,3,subplot_kw={'projection':ccrs.epsg(2056)},figsize=(10,6),
    #                               gridspec_kw={'wspace':0.32})


    # #PAA
    # paa = ds_KGV.sel(date=date_dt)['PAA']*100
    # q99 = np.quantile(paa.values.flatten()[~np.isnan(paa.values.flatten())],0.99)
    # sc.plot_funcs.plot_nc(paa,ax=ax0,title=date_dt.strftime('%Y-%m-%d: '),cmap='Blues',vmax=q99,cbar_lbl='Assets Affected [%]',**pl_kwargs)
    # exp_count=ds_KGV.sel(date=date_dt)['n_count_exposure']; density=6
    # #grey out areas without enough exposure
    # plot_nc(exp_count.fillna(0)<min_exp_count,ax=ax0,pl_type='bool_field',cmap='lightgrey',**pl_kwargs)


    # #plot MEHSHS, dBZ, E_kin
    # sc.plot_funcs.plot_nc(meshs.MZC.sel(time=date_dt),ax=ax1,title=date_dt.strftime('%Y-%m-%d: '),cmap='viridis',vmax=100,cbar_lbl='MESHS [mm]',**pl_kwargs)
    # sc.plot_funcs.plot_nc(dBZ.CZC.sel(time=date_dt),ax=ax2,title=date_dt.strftime('%Y-%m-%d: '),cmap='magma_r',vmin=45,vmax=70,cbar_lbl='Reflectivity [dBZ]',**pl_kwargs)
    # sc.plot_funcs.plot_nc(E_kin.sel(time=date_dt),ax=ax3,title=date_dt.strftime('%Y-%m-%d: '),cmap='magma_r',vmin=1,cbar_lbl='E$_{kin}$ [Jm$^{-2}$]',**pl_kwargs)

    # return fig