#!/usr/bin/env python
# coding: utf-8


import unittest
import sys
sys.path.insert(1, '../visual_clutter/')
from clutter import *

clt = Clutter('test.jpg', numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=3, color_pool_sigma=3)
color_clutter = clt.colorClutter(color_pix=1)
contrast_clutter = clt.contrastClutter(contrast_pix=1)
orientation_clutter = clt.orientationClutter(orient_pix=1)
color_clutter, contrast_clutter, orientation_clutter = clt.computeClutter(color_pix=1, contrast_pix=1, orient_pix=1)
clutter_scalar_fc, clutter_map_fc = clt.getClutter_FC(p=1, pix=1)

print(f'clutter_scalar_fc: {clutter_scalar_fc}')
print(f'clutter_scalar_se: {clt.getClutter_SE(wlevels=3, wght_chrom=0.0625)}')
