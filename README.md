# visual_clutter
Python Implementation of two measures of visual clutter (Feature Congestion and Subband Entropy), [Matlab Version](https://dspace.mit.edu/handle/1721.1/37593), [+ library dependency](https://nl.mathworks.com/matlabcentral/fileexchange/52571-matlabpyrtools).


## Pre-requisites
* This utility is written in Python 3. You will need a Python 3 interpreter installed or you will have to package this into a self contained executable. 

* This utility uses [Pyrtools](https://pyrtools.readthedocs.io/en/latest/). So you will need to run it on Linux or on OSX. Windows is NOT supported because of issues with the C compiler (gcc isn't necessarily installed).


## Dependencies
- opencv-python >= 4.5.3
- numpy >= 1.20.2
- scipy == 1.7.0
- Pillow >= 8.3.1
- pyrtools == 1.0.0
- scikit-image >= 0.16.2

## How to Install visual_clutter

```
pip3 install git+https://github.com/kargaranamir/visual-clutter
```

## How to Upgrade visual_clutter

```
pip3 install --upgrade git+https://github.com/kargaranamir/visual-clutter
```



## How to use (Examples)
```
from visual_clutter import Vlc

# make visual clutter object and load test map and set parameters
clt = Vlc('./tests/test.jpg', numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=3, color_pool_sigma=3)

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

# just compute and display contrast clutter map(s)
orientation_clutter = clt.orientationClutter(orient_pix=1)

```

## Reference
```
Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano. "Measuring Visual Clutter". 
Journal of Vision, 7(2), 2007. http://www.journalofvision.com/7/2/

Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano, May 2007.
```

## Related Repositories
- [Piranhas](https://github.com/ArturoDeza/Piranhas)
- [Aalto Interface Metrics (AIM)](https://github.com/aalto-ui/aim)
- [pyrtools: tools for multi-scale image processing](https://github.com/LabForComputationalVision/pyrtools)

