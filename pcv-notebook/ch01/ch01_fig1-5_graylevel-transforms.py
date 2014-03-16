from PIL import Image
from numpy import *
from pylab import *

im = array(Image.open('../data/empire.jpg').convert('L'))
im2 = 255 - im  # invert image
im3 = (100.0/255) * im + 100  # clamp to interval 100...200
im4 = 255.0 * (im/255.0)**2  # squared

figure()
gray()
subplot(1, 3, 1)
imshow(im2)
axis('off')

subplot(1, 3, 2)
imshow(im3)
axis('off')

subplot(1, 3, 3)
imshow(im4)
axis('off')

show()