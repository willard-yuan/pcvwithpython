from PIL import Image
from pylab import *
from scipy.ndimage import filters
import numpy

im = array(Image.open('../data/empire.jpg').convert('L'))
gray()
subplot(1, 4, 1)
axis('off')
imshow(im)

sigma = 5

imx = zeros(im.shape)
filters.gaussian_filter(im, sigma, (0, 1), imx)
subplot(1, 4, 2)
axis('off')
imshow(imx)

imy = zeros(im.shape)
filters.gaussian_filter(im, sigma, (1, 0), imy)
subplot(1, 4, 3)
axis('off')
imshow(imy)

# there's also gaussian_gradient_magnitude()
mag = numpy.sqrt(imx**2 + imy**2)
subplot(1, 4, 4)
axis('off')
imshow(mag)

show()
