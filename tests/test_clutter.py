#!/usr/bin/env python
# coding: utf-8


import unittest
import sys
sys.path.insert(1, '../visual_clutter/')
from clutter import *

# make visual clutter object and load test map and set parameters
clt = Vlc('test.jpg', numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=3, color_pool_sigma=3)

# get Feature Congestion clutter of a test map:
clutter_scalar_fc, clutter_map_fc = clt.getClutter_FC(p=1, pix=1)

# get Subband Entropy clutter of the test map:
clutter_scalar_se = clt.getClutter_SE(wlevels=3, wght_chrom=0.0625)

print(f'clutter_scalar_fc: {clutter_scalar_fc}')
print(f'clutter_scalar_se: {clutter_scalar_se}')

# just compute and display color clutter map(s)
color_clutter = clt.colorClutter(color_pix=1)

# just compute and display contrast clutter map(s)
contrast_clutter = clt.contrastClutter(contrast_pix=1)

# just compute and display orientation clutter map(s)
orientation_clutter = clt.orientationClutter(orient_pix=1)