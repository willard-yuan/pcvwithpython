# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from scipy.ndimage import filters
import numpy

im = array(Image.open('../data/empire.jpg').convert('L'))
gray()

subplot(1, 4, 1)
axis('off')
title('(a)')
imshow(im)

imx = zeros(im.shape)
filters.sobel(im, 1, imx)
subplot(1, 4, 2)
axis('off')
title('(b)')
imshow(imx)

imy = zeros(im.shape)
filters.sobel(im, 0, imy)
subplot(1, 4, 3)
axis('off')
title('(c)')
imshow(imy)

mag = numpy.sqrt(imx**2 + imy**2)
subplot(1, 4, 4)
title('(d)')
axis('off')
imshow(mag)

show()
