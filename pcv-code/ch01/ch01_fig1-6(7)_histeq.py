from PIL import Image
from pylab import *
from PCV.tools import imtools

#im = array(Image.open('../data/empire.jpg').convert('L'))
im = array(Image.open('../data/AquaTermi_lowcontrast.JPG').convert('L'))
im2, cdf = imtools.histeq(im)

figure()
subplot(2, 2, 1)
axis('off')
gray()
title('original')
imshow(im)

subplot(2, 2, 2)
axis('off')
title('histogram-equalized')
imshow(im2)

subplot(2, 2, 3)
axis('off')
title('original hist')
#hist(im.flatten(), 128, cumulative=True, normed=True)
hist(im.flatten(), 128, normed=True)

subplot(2, 2, 4)
axis('off')
title('equalized hist')
#hist(im2.flatten(), 128, cumulative=True, normed=True)
hist(im2.flatten(), 128, normed=True)

show()
