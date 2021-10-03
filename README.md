Simple python-based quantification software for simple biological images. Currently implemented for protein and DNA gels/blots and yeast spotting assays. Takes an input image which gets cropped by the user and converts to a grayscale numpy array using the Python Image Library (PIL). Then splits image into user-specified lanes/boxes, and the average intensity of every pixel per lane/band/blot/spot is calculated. 

For gels, intensities have the option to be gaussian-weighted so that the middle of the gel band contributes more to the average than the outer edges, as other lanes in gels sometimes bleed into the lane of interest. Data for each slice is baselined given a user-specified baselining region. This data is plotted and can be quantified for band intensities.

Note: updates needed for SDS-PAGE gel analysis (spotting assay approach is much faster/better implemented). The pipeline is functional and accurate but slow; old code is preserved to reflect analysis in https://elifesciences.org/articles/54100. 

See example .ipynb notebooks for details/example usage.

To install a development version, clone this repo and pip install:

pip install -e .

Dependencies: matplotlib, numpy, PIL, pandas, os, natsort, scipy
