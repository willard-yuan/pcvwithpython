 # -*- coding: utf-8 -*-
from pylab import *
from PIL import Image

from PCV.localdescriptors import harris

"""
Example of detecting Harris corner points (Figure 2-1 in the book).
"""

# open image
im = array(Image.open('../data/empire.jpg').convert('L'))

# detect corners and plot
harrisim = harris.compute_harris_response(im)
# the Harris response function
harrisim1 = 255-harrisim
figure()
gray()
imshow(harrisim1)
axis('off')
filtered_coords = harris.get_harris_points(harrisim, 6, threshold=0.01)
#filtered_coords = harris.get_harris_points(harrisim, 6, threshold=0.05)
#filtered_coords = harris.get_harris_points(harrisim, 6, threshold=0.1)
harris.plot_harris_points(im, filtered_coords)

# plot only 200 strongest
# harris.plot_harris_points(im, filtered_coords[:200])