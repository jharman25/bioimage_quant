Simple python-based quantification software for simple biological images. Currently implemented for protein and DNA gels/blots and yeast spotting assays. Takes an input image which gets cropped by the user. Then splits image into user-specified vertical lanes, which are temporarily saved. These lanes are then converted to numpy arrays, and the average RGB intensity of every pixel is calculated. 

For gels, intensities have the option to be gaussian-weighted so that the middle of the gel band contributes more to the average than the outer edges, as other lanes in gels sometimes bleed into the lane of interest. Data for each slice is baselined given a user-specified baselining region. This data is plotted and can be quantified for band intensities.

See notebook-example.ipynb and gel-example.png for details/example usage.

To install a development version, clone this repo and pip install:

pip install -e .

Dependencies: matplotlib, numpy, PIL, pandas, os, natsort, scipy
