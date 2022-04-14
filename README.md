# visual_clutter
Python Implementation of two measures of visual clutter (Feature Congestion and Subband Entropy), [Matlab Version](https://dspace.mit.edu/handle/1721.1/37593), [+ library dependency](https://nl.mathworks.com/matlabcentral/fileexchange/52571-matlabpyrtools).


## Pre-requisites
* This utility is written in Python 3. You will need a Python 3 interpreter installed or you will have to package this into a self contained executable. 

* This utility uses [Pyrtools](https://pyrtools.readthedocs.io/en/latest/). So you will need to run it on Linux or on OSX. Windows is NOT supported because of issues with the C compiler (gcc isn't necessarily installed).



## How to Install visual_clutter

```
pip install visual-clutter
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

# just compute and display orientation clutter map(s)
orientation_clutter = clt.orientationClutter(orient_pix=1)

```

## Reference
```
Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano. "Measuring Visual Clutter". 
Journal of Vision, 7(2), 2007. http://www.journalofvision.com/7/2/

Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano, May 2007.
```

## Citing
visual_clutter python package is now part of [AIM2](https://github.com/aalto-ui/aim). AIM2 developed and supported by User Interfaces Research Group at Aalto University. If you use any part of this library in your research, please cite it using the following BibTex entry.
```
@inproceedings{oulasvirta2018aalto,
  title={Aalto Interface Metrics (AIM) A Service and Codebase for Computational GUI Evaluation},
  author={Oulasvirta, Antti and De Pascale, Samuli and Koch, Janin and Langerak, Thomas and Jokinen, Jussi and Todi, Kashyap and Laine, Markku and Kristhombuge, Manoj and Zhu, Yuxi and Miniukovich, Aliaksei and others},
  booktitle={The 31st Annual ACM Symposium on User Interface Software and Technology Adjunct Proceedings},
  pages={16--19},
  year={2018}
}

@misc{visual_clutter,
  author = {Kargaran, Amir Hossein},
  title = {Visual Clutter Python Library},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/kargaranamir/visual_clutter}},
}
```

## Related Repositories
- [Piranhas](https://github.com/ArturoDeza/Piranhas)
- [Aalto Interface Metrics (AIM)](https://github.com/aalto-ui/aim)
- [pyrtools: tools for multi-scale image processing](https://github.com/LabForComputationalVision/pyrtools)
