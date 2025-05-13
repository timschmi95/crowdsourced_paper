# -*- coding: utf-8 -*-
"""
Customized colormaps for plotting

"""

# modules
import matplotlib.colors
import numpy as np
import colorcet as cc
import copy

# COLORMAP DAMAGES TRUNCATED / PERCEPTUALLY UNIFORM
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

CMAP_IMPACT_CC = truncate_colormap(cc.cm.fire_r, 0.1, 1)
CMAP_IMPACT_CC.set_under('white',alpha=0)
CMAP_IMPACT_CHF = truncate_colormap(cc.cm.rainbow4, 0.5, 1)
CMAP_IMPACT_CC.set_under('white',alpha=0)
CMAP_HAZARD_HAILCAST = copy.copy(cc.cm.bmy)